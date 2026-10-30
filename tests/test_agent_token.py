"""Tests for Bearer agent-token verification (src/agent_token.py).

Uses a local JWKS server whose response can be changed mid-test, and an
injected clock so the refresh, staleness and claim-time windows are exact.
"""

import asyncio
import json
import ssl
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import ClassVar

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from jwt import algorithms as jwt_algorithms

from src.agent_token import (
    FETCH_RETRY_FLOOR_S,
    KEY_SET_MAX_STALE_S,
    KEY_SET_REFRESH_S,
    UNKNOWN_KID_COOLDOWN_S,
    AgentTokenUnavailableError,
    AgentTokenVerifier,
    InvalidAgentTokenError,
    check_agent_token_config,
)
from src.config import Settings
from src.telemetry.http import create_async_client

ISSUER = "https://support.example.test"
AUDIENCE = "https://qa.example.test"
AZP = "access-qa-bot"
AGENT_KID = "agent-kid"
XDMOD_KID = "xdmod-kid"


def _keypair():
    return ec.generate_private_key(ec.SECP256R1())


def _jwk(private_key, kid: str) -> dict:
    jwk = json.loads(
        jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(
            private_key.public_key()
        )
    )
    jwk.update({"kid": kid, "use": "sig", "alg": "ES256"})
    return jwk


AGENT_KEY = _keypair()
XDMOD_KEY = _keypair()
COOKIE_KEY = _keypair()
ROTATED_KEY = _keypair()
EXCHANGE_JWKS = {"keys": [_jwk(AGENT_KEY, AGENT_KID), _jwk(XDMOD_KEY, XDMOD_KID)]}


class _JWKSHandler(BaseHTTPRequestHandler):
    status: ClassVar[int] = 200
    body: ClassVar[bytes] = b""
    delay_s: ClassVar[float] = 0.0
    hits: ClassVar[int] = 0

    def do_GET(self):
        type(self).hits += 1
        if self.delay_s:
            time.sleep(self.delay_s)
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.body)

    def log_message(self, format, *args):
        pass


def _serve(body, status: int = 200) -> None:
    _JWKSHandler.status = status
    _JWKSHandler.body = body if isinstance(body, bytes) else json.dumps(body).encode()


@pytest.fixture(autouse=True)
def _reset_jwks_handler():
    _serve(EXCHANGE_JWKS)
    _JWKSHandler.delay_s = 0.0
    _JWKSHandler.hits = 0


@pytest.fixture
def jwks_url():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _JWKSHandler)
    Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/.well-known/exchange-jwks.json"
    server.shutdown()
    server.server_close()


class _Clock:
    def __init__(self) -> None:
        self.now = 1_800_000_000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock():
    return _Clock()


@pytest.fixture
async def verifier(jwks_url, clock):
    v = AgentTokenVerifier(
        issuer=ISSUER,
        jwks_url=jwks_url,
        audience=AUDIENCE,
        allowed_azp=frozenset({AZP}),
        clock=clock,
        cache_clock=clock,
    )
    yield v
    await v.aclose()


def _token(clock, *, key=AGENT_KEY, kid=AGENT_KID, header=None, drop=(), **claims) -> str:
    now = int(clock.now)
    payload = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "azp": AZP,
        "sub": "jsmith@access-ci.org",
        "jti": "jti-1",
        "scope": "agent:read agent:write",
        "iat": now,
        "exp": now + 900,
    }
    payload.update(claims)
    for name in drop:
        payload.pop(name)
    headers = {"kid": kid, "typ": "at+jwt"}
    headers.update(header or {})
    headers = {k: v for k, v in headers.items() if v is not None}
    return jwt.encode(payload, key, algorithm="ES256", headers=headers)


# --- A valid token ---


async def test_valid_token_returns_identity(verifier, clock):
    identity = await verifier.verify(_token(clock))
    assert identity.sub == "jsmith@access-ci.org"
    assert identity.scopes == frozenset({"agent:read", "agent:write"})
    assert identity.exp == int(clock.now) + 900


async def test_read_only_scope_is_accepted(verifier, clock):
    identity = await verifier.verify(_token(clock, scope="agent:read"))
    assert identity.scopes == frozenset({"agent:read"})


# --- Header checks ---


async def test_non_es256_alg_is_rejected(verifier, clock):
    token = jwt.encode(
        {"sub": "x"},
        "secret-long-enough-for-hs256-0123456789",
        algorithm="HS256",
        headers={"kid": AGENT_KID, "typ": "at+jwt"},
    )
    with pytest.raises(InvalidAgentTokenError, match="alg"):
        await verifier.verify(token)


@pytest.mark.parametrize("typ", [None, "JWT", "application/at+jwt"])
async def test_typ_other_than_at_jwt_is_rejected(verifier, clock, typ):
    with pytest.raises(InvalidAgentTokenError, match="typ"):
        await verifier.verify(_token(clock, header={"typ": typ}))


async def test_missing_kid_is_rejected(verifier, clock):
    with pytest.raises(InvalidAgentTokenError, match="kid"):
        await verifier.verify(_token(clock, header={"kid": None}))


async def test_garbage_is_rejected(verifier):
    with pytest.raises(InvalidAgentTokenError, match="malformed"):
        await verifier.verify("not-a-jwt")


async def test_signature_from_another_key_is_rejected(verifier, clock):
    with pytest.raises(InvalidAgentTokenError):
        await verifier.verify(_token(clock, key=COOKIE_KEY))


# --- Claim checks ---


@pytest.mark.parametrize(
    "claims",
    [
        {"iss": "https://other.example.test"},
        {"aud": "https://xdmod.dev.invalid"},
        {"aud": [AUDIENCE, "https://other.example.test"]},
        {"azp": "some-other-client"},
        {"scope": "agent:write"},
        {"scope": ""},
        {"sub": ""},
        {"jti": ""},
        {"exp": "soon"},
        {"exp": float("inf"), "iat": float("nan")},
        {"azp": [AZP]},
    ],
)
async def test_bad_claim_is_rejected(verifier, clock, claims):
    with pytest.raises(InvalidAgentTokenError):
        await verifier.verify(_token(clock, **claims))


@pytest.mark.parametrize("name", ["iss", "aud", "sub", "jti", "iat", "exp", "azp", "scope"])
async def test_missing_claim_is_rejected(verifier, clock, name):
    with pytest.raises(InvalidAgentTokenError):
        await verifier.verify(_token(clock, drop=(name,)))


async def test_expired_token_is_rejected(verifier, clock):
    now = int(clock.now)
    with pytest.raises(InvalidAgentTokenError, match="expired"):
        await verifier.verify(_token(clock, iat=now - 901 + 2, exp=now - 1))


async def test_token_expiring_now_is_rejected(verifier, clock):
    now = int(clock.now)
    with pytest.raises(InvalidAgentTokenError, match="expired"):
        await verifier.verify(_token(clock, iat=now - 100, exp=now))


async def test_fractional_times_are_rejected(verifier, clock):
    now = int(clock.now)
    with pytest.raises(InvalidAgentTokenError, match="integers"):
        await verifier.verify(_token(clock, iat=now + 0.5, exp=now + 900))


async def test_iat_up_to_60_seconds_ahead_is_accepted(verifier, clock):
    now = int(clock.now)
    await verifier.verify(_token(clock, iat=now + 59, exp=now + 59 + 900))


async def test_iat_more_than_60_seconds_ahead_is_rejected(verifier, clock):
    now = int(clock.now)
    with pytest.raises(InvalidAgentTokenError, match="iat"):
        await verifier.verify(_token(clock, iat=now + 61, exp=now + 61 + 900))


async def test_lifetime_over_900_seconds_is_rejected(verifier, clock):
    now = int(clock.now)
    with pytest.raises(InvalidAgentTokenError, match="lifetime"):
        await verifier.verify(_token(clock, iat=now, exp=now + 901))


async def test_future_nbf_is_rejected(verifier, clock):
    with pytest.raises(InvalidAgentTokenError, match="not yet valid"):
        await verifier.verify(_token(clock, nbf=int(clock.now) + 61))


async def test_xdmod_audience_token_is_rejected(verifier, clock):
    # Signed with a key that IS in exchange-jwks.json, so only the claims stop it.
    token = _token(clock, key=XDMOD_KEY, kid=XDMOD_KID, aud="https://xdmod.dev.invalid")
    with pytest.raises(InvalidAgentTokenError):
        await verifier.verify(token)


async def test_cookie_key_token_is_rejected(verifier, clock):
    # The cookie key is never in exchange-jwks.json.
    with pytest.raises(InvalidAgentTokenError, match="unknown kid"):
        await verifier.verify(_token(clock, key=COOKIE_KEY, kid="cookie-kid"))


# --- Key-set rules ---


async def test_empty_key_set_revokes(verifier, clock):
    await verifier.verify(_token(clock))
    _serve({"keys": []})
    clock.now += KEY_SET_REFRESH_S
    with pytest.raises(InvalidAgentTokenError, match="unknown kid"):
        await verifier.verify(_token(clock))


@pytest.mark.parametrize(
    ("body", "status"),
    [({}, 200), ({"keys": None}, 200), ([], 200), (b"not json", 200), (EXCHANGE_JWKS, 500)],
)
async def test_failed_first_fetch_is_unavailable(verifier, clock, body, status):
    _serve(body, status)
    with pytest.raises(AgentTokenUnavailableError):
        await verifier.verify(_token(clock))


@pytest.mark.parametrize(("body", "status"), [({}, 200), ({"keys": None}, 200), (b"", 503)])
async def test_failed_fetch_keeps_old_keys(verifier, clock, body, status):
    await verifier.verify(_token(clock))
    _serve(body, status)
    clock.now += KEY_SET_REFRESH_S
    await verifier.verify(_token(clock))
    assert _JWKSHandler.hits == 2


async def test_unreachable_issuer_keeps_old_keys(clock):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _JWKSHandler)
    Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/jwks"
    v = AgentTokenVerifier(
        issuer=ISSUER,
        jwks_url=url,
        audience=AUDIENCE,
        allowed_azp=frozenset({AZP}),
        clock=clock,
        cache_clock=clock,
    )
    try:
        await v.verify(_token(clock))
        server.shutdown()
        server.server_close()
        clock.now += KEY_SET_REFRESH_S
        await v.verify(_token(clock))
    finally:
        await v.aclose()


async def test_stale_set_is_unavailable_after_two_hours(verifier, clock):
    await verifier.verify(_token(clock))
    _serve(b"", 500)
    clock.now += KEY_SET_MAX_STALE_S
    await verifier.verify(_token(clock))
    clock.now += 1
    with pytest.raises(AgentTokenUnavailableError):
        await verifier.verify(_token(clock))


async def test_refetch_picks_up_removed_key(verifier, clock):
    await verifier.verify(_token(clock))
    _serve({"keys": [_jwk(XDMOD_KEY, XDMOD_KID)]})
    clock.now += KEY_SET_REFRESH_S - 1
    await verifier.verify(_token(clock))
    clock.now += 1
    with pytest.raises(InvalidAgentTokenError, match="unknown kid"):
        await verifier.verify(_token(clock))


async def test_unknown_kid_forces_one_refresh_that_finds_a_rotated_key(verifier, clock):
    await verifier.verify(_token(clock))
    _serve({"keys": [*EXCHANGE_JWKS["keys"], _jwk(ROTATED_KEY, "rotated")]})
    await verifier.verify(_token(clock, key=ROTATED_KEY, kid="rotated"))
    assert _JWKSHandler.hits == 2


async def test_unknown_kid_cooldown_holds(verifier, clock):
    await verifier.verify(_token(clock))
    for kid in ("bogus-1", "bogus-2", "bogus-3"):
        with pytest.raises(InvalidAgentTokenError, match="unknown kid"):
            await verifier.verify(_token(clock, kid=kid))
    assert _JWKSHandler.hits == 2

    clock.now += UNKNOWN_KID_COOLDOWN_S - 1
    with pytest.raises(InvalidAgentTokenError):
        await verifier.verify(_token(clock, kid="bogus-4"))
    assert _JWKSHandler.hits == 2

    clock.now += 1
    # The periodic refresh is due at the same moment; it is the only fetch.
    with pytest.raises(InvalidAgentTokenError):
        await verifier.verify(_token(clock, kid="bogus-5"))
    assert _JWKSHandler.hits == 3


async def test_failed_fetch_is_not_retried_within_the_floor(verifier, clock):
    _serve(b"", 500)
    for _ in range(3):
        with pytest.raises(AgentTokenUnavailableError):
            await verifier.verify(_token(clock))
    assert _JWKSHandler.hits == 1

    _serve(EXCHANGE_JWKS)
    clock.now += FETCH_RETRY_FLOOR_S - 1
    with pytest.raises(AgentTokenUnavailableError):
        await verifier.verify(_token(clock))
    clock.now += 1
    await verifier.verify(_token(clock))
    assert _JWKSHandler.hits == 2


async def test_concurrent_requests_share_one_fetch(verifier, clock):
    _JWKSHandler.delay_s = 0.2
    results = await asyncio.gather(*(verifier.verify(_token(clock)) for _ in range(5)))
    assert len(results) == 5
    assert _JWKSHandler.hits == 1


async def test_cancelled_waiter_does_not_cancel_the_shared_fetch(verifier, clock):
    _JWKSHandler.delay_s = 0.2
    first = asyncio.ensure_future(verifier.verify(_token(clock)))
    second = asyncio.ensure_future(verifier.verify(_token(clock)))
    await asyncio.sleep(0.05)
    first.cancel()
    assert (await second).sub == "jsmith@access-ci.org"
    assert _JWKSHandler.hits == 1


async def test_unusable_keys_are_skipped(verifier, clock):
    _serve(
        {
            "keys": [
                {"kty": "EC", "kid": "broken", "crv": "P-256", "x": "AA", "y": "AA"},
                {"kty": "oct", "kid": "symmetric", "k": "c2VjcmV0"},
                {"kty": "EC"},
                "not-an-object",
                *EXCHANGE_JWKS["keys"],
            ]
        }
    )
    await verifier.verify(_token(clock))
    with pytest.raises(InvalidAgentTokenError, match="unknown kid"):
        await verifier.verify(_token(clock, kid="symmetric"))


# --- Settings and startup ---


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _production(**overrides) -> Settings:
    base = {
        "ENVIRONMENT": "production",
        "LLM_PROVIDER": "vllm",
        "EVAL_JUDGE_BASE_URL": "https://judge.example.test/v1",
        "AGENT_TOKEN_ISSUER": ISSUER,
        "AGENT_TOKEN_JWKS_URL": f"{ISSUER}/.well-known/exchange-jwks.json",
    }
    base.update(overrides)
    return _settings(**base)


async def test_from_settings_uses_the_environment_defaults(jwks_url, clock):
    s = _settings(ENVIRONMENT="local", AGENT_TOKEN_ISSUER=ISSUER, AGENT_TOKEN_JWKS_URL=jwks_url)
    v = AgentTokenVerifier.from_settings(s)
    v._clock = v._cache_clock = clock
    try:
        identity = await v.verify(_token(clock, aud="http://localhost:8000"))
        assert identity.sub == "jsmith@access-ci.org"
    finally:
        await v.aclose()


async def test_from_settings_verifies_tls_outside_local_and_docker():
    v = AgentTokenVerifier.from_settings(_production())
    local = AgentTokenVerifier.from_settings(_settings(ENVIRONMENT="docker"))
    try:
        assert _ssl_context(v._client).verify_mode == ssl.CERT_REQUIRED
        assert _ssl_context(local._client).verify_mode == ssl.CERT_NONE
    finally:
        await v.aclose()
        await local.aclose()


def test_audience_defaults_per_environment():
    assert _settings(ENVIRONMENT="local").agent_token_audience == "http://localhost:8000"
    assert _production().agent_token_audience == "https://qa.access-ci.org"
    assert _settings(AGENT_TOKEN_AUDIENCE=AUDIENCE).agent_token_audience == AUDIENCE


def test_allowed_azp_is_a_comma_separated_list():
    assert _settings().agent_token_allowed_azp == frozenset({"access-qa-bot"})
    s = _settings(AGENT_TOKEN_ALLOWED_AZP=" a , b ,")
    assert s.agent_token_allowed_azp == frozenset({"a", "b"})


@pytest.mark.parametrize("timeout", [0, 121])
def test_turn_timeout_outside_1_to_120_is_refused(timeout):
    with pytest.raises(RuntimeError, match="AGENT_TURN_TIMEOUT_S"):
        check_agent_token_config(_settings(ENVIRONMENT="local", AGENT_TURN_TIMEOUT_S=timeout))


def test_turn_timeout_of_120_is_allowed():
    check_agent_token_config(_settings(ENVIRONMENT="local", AGENT_TURN_TIMEOUT_S=120))


@pytest.mark.parametrize("missing", ["AGENT_TOKEN_ISSUER", "AGENT_TOKEN_JWKS_URL"])
def test_production_refuses_to_start_without_issuer_settings(missing):
    with pytest.raises(RuntimeError, match=missing):
        check_agent_token_config(_production(**{missing: ""}))


def test_production_refuses_a_plain_http_jwks_url():
    with pytest.raises(RuntimeError, match="https"):
        check_agent_token_config(_production(AGENT_TOKEN_JWKS_URL="http://x.test/jwks"))


def test_production_starts_with_issuer_settings():
    check_agent_token_config(_production())


def test_local_starts_without_issuer_settings():
    check_agent_token_config(_settings(ENVIRONMENT="local"))


# --- HTTP client TLS option ---


def _ssl_context(client: httpx.AsyncClient) -> ssl.SSLContext:
    transport = client._transport
    transport = getattr(transport, "_transport", transport)
    return transport._pool._ssl_context


@pytest.mark.parametrize("otel", ["true", "false"])
async def test_create_async_client_can_skip_tls_verification(monkeypatch, otel):
    monkeypatch.setenv("OTEL_ENABLED", otel)
    unverified = create_async_client(verify=False)
    verified = create_async_client()
    try:
        assert _ssl_context(unverified).verify_mode == ssl.CERT_NONE
        assert _ssl_context(verified).verify_mode == ssl.CERT_REQUIRED
    finally:
        await unverified.aclose()
        await verified.aclose()


async def test_a_key_that_breaks_the_parser_counts_as_unusable(verifier, clock):
    bad = {"kid": "bad", "kty": "EC", "crv": "P-256", "x": 123, "y": 456}
    _serve({"keys": [bad, *EXCHANGE_JWKS["keys"]]})
    await verifier.verify(_token(clock))
    assert _JWKSHandler.hits == 1


async def test_unexpected_fetch_exception_still_applies_the_retry_floor(
    verifier, clock, monkeypatch
):
    calls = 0

    async def broken_get(url):
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(verifier._client, "get", broken_get)
    for _ in range(3):
        with pytest.raises(AgentTokenUnavailableError):
            await verifier.verify(_token(clock))
    assert calls == 1
