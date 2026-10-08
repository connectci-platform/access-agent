"""Tests for JWT cookie authentication (src/auth.py).

Uses ES256 (ECDSA P-256) key pairs — no shared secret.
"""

import json
import logging
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import ClassVar
from unittest.mock import MagicMock

import jwt
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from jwt import algorithms as jwt_algorithms

from src.auth import (
    JWKS_REFRESH_COOLDOWN_S,
    _jwks_clients,
    configure_trusted_issuers,
    get_acting_user_from_cookie,
)

# Generate a test EC P-256 key pair.
_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

# Build a JWKS response for the test public key.
_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "test-kid-001"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "test-kid-001"


# A minimal HTTP server that serves the JWKS endpoint for tests.
class _JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(JWKS_RESPONSE)

    def log_message(self, format, *args):
        pass  # Suppress server logs in test output.


def _start_jwks_server() -> tuple[HTTPServer, str]:
    """Start a local JWKS server and return (server, url)."""
    server = HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


def _make_jwt(sub: str, expired: bool = False, issuer: str = ISSUER) -> str:
    """Create an ES256-signed JWT for testing."""
    now = int(time.time())
    payload = {
        "iss": issuer,
        "sub": sub,
        "iat": now - 3600,
        "exp": (now - 10) if expired else (now + 3600),
    }
    return jwt.encode(
        payload,
        PRIVATE_PEM,
        algorithm="ES256",
        headers={"kid": KID},
    )


def _mock_request(cookies: dict | None = None):
    """Create a mock FastAPI Request with cookies."""
    request = MagicMock()
    request.cookies = cookies or {}
    return request


def _setup_issuers(jwks_url: str) -> None:
    """Configure trusted issuers with the test JWKS server."""
    configure_trusted_issuers({ISSUER: jwks_url})


# --- Valid JWT cookie ---


def test_valid_jwt_cookie():
    """Valid ES256 JWT cookie returns the ACCESS ID and cookie_present=True."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("jsmith@access-ci.org")
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user == "jsmith@access-ci.org"
        assert cookie_present is True
    finally:
        server.shutdown()


def test_valid_jwt_cookie_result_is_not_affected_by_body():
    """Cookie resolution is independent of body content (body not read)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("cookie-user@access-ci.org")
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user == "cookie-user@access-ci.org"
        assert cookie_present is True
    finally:
        server.shutdown()


# --- Expired JWT cookie ---


def test_expired_jwt_cookie_returns_none():
    """Expired JWT cookie returns (None, True) — cookie present but invalid."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("jsmith@access-ci.org", expired=True)
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


# --- Invalid JWT cookie ---


def test_invalid_jwt_cookie_returns_none():
    """Tampered/invalid JWT cookie returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        request = _mock_request(cookies={"SESSaccess_auth": "not-a-valid-jwt"})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_wrong_key_returns_none():
    """JWT signed with a different key returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        # Sign with a different key pair.
        other_key = ec.generate_private_key(ec.SECP256R1())
        other_pem = other_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        token = jwt.encode(
            {
                "iss": ISSUER,
                "sub": "jsmith@access-ci.org",
                "exp": int(time.time()) + 3600,
            },
            other_pem,
            algorithm="ES256",
            headers={"kid": "unknown-kid"},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_untrusted_issuer_returns_none():
    """JWT from an untrusted issuer returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("jsmith@access-ci.org", issuer="https://evil.com")
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_jwt_missing_sub_claim():
    """JWT without 'sub' claim returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        now = int(time.time())
        token = jwt.encode(
            {"iss": ISSUER, "email": "jsmith@access-ci.org", "exp": now + 3600},
            PRIVATE_PEM,
            algorithm="ES256",
            headers={"kid": KID},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_jwt_empty_sub_claim():
    """JWT with empty 'sub' claim returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        now = int(time.time())
        token = jwt.encode(
            {"iss": ISSUER, "sub": "", "exp": now + 3600},
            PRIVATE_PEM,
            algorithm="ES256",
            headers={"kid": KID},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


def test_jwt_missing_iss_claim():
    """JWT without 'iss' claim returns (None, True)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        now = int(time.time())
        token = jwt.encode(
            {"sub": "jsmith@access-ci.org", "exp": now + 3600},
            PRIVATE_PEM,
            algorithm="ES256",
            headers={"kid": KID},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


# --- Algorithm confusion attack ---


def test_hs256_jwt_rejected():
    """HS256-signed JWT is rejected even if payload looks valid (algorithm confusion)."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        now = int(time.time())
        token = jwt.encode(
            {"iss": ISSUER, "sub": "jsmith@access-ci.org", "exp": now + 3600},
            "some-shared-secret-that-should-not-work",
            algorithm="HS256",
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
    finally:
        server.shutdown()


# --- JWKS server unreachable ---


def test_jwks_server_unreachable():
    """When the JWKS server is unreachable, JWT returns (None, True)."""
    # Point at a port with nothing listening.
    configure_trusted_issuers({ISSUER: "http://127.0.0.1:1"})
    token = _make_jwt("jsmith@access-ci.org")
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request)

    assert user is None
    assert cookie_present is True


# --- No cookie ---


def test_no_cookie_returns_none_and_not_present():
    """No cookie returns (None, False) — caller decides on fallback."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        request = _mock_request()

        user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is False
    finally:
        server.shutdown()


# --- No issuers configured ---


def test_no_issuers_configured():
    """With no trusted issuers, any JWT returns (None, True)."""
    configure_trusted_issuers({})
    token = _make_jwt("jsmith@access-ci.org")
    request = _mock_request(cookies={"SESSaccess_auth": token})

    user, cookie_present = get_acting_user_from_cookie(request)

    assert user is None
    assert cookie_present is True


# --- Environment parameter on configure_trusted_issuers ---


def test_configure_trusted_issuers_docker_sets_ssl_context_without_hostname_check():
    """When environment='docker', PyJWKClient gets an ssl_context with check_hostname=False."""
    configure_trusted_issuers(
        {ISSUER: "https://example.com/.well-known/jwks.json"},
        environment="docker",
    )

    assert ISSUER in _jwks_clients
    client = _jwks_clients[ISSUER]

    ssl_ctx = getattr(client, "ssl_context", None)
    assert ssl_ctx is not None, "Expected an ssl_context on the PyJWKClient for docker environment"
    assert ssl_ctx.check_hostname is False


def test_configure_trusted_issuers_production_has_no_custom_ssl_context():
    """When environment='production' (default), PyJWKClient has no custom ssl_context."""
    configure_trusted_issuers(
        {ISSUER: "https://example.com/.well-known/jwks.json"},
        environment="production",
    )

    assert ISSUER in _jwks_clients
    client = _jwks_clients[ISSUER]

    ssl_ctx = getattr(client, "ssl_context", None)
    assert ssl_ctx is None, "Expected no custom ssl_context for production environment"


# --- Failure classification: user state vs infrastructure (issue #243) ---
#
# Every failure path returns the same (None, True), so the caller cannot tell
# an expired session from a JWKS outage. Operators must be able to: a wave of
# expired cookies is normal, a JWKS outage is an incident. These tests pin the
# log classification — the return value deliberately stays fail-open.


def test_expired_cookie_logs_as_user_state(caplog):
    """An expired session is the ordinary end of a session, not an incident."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = _make_jwt("jsmith@access-ci.org", expired=True)
        request = _mock_request(cookies={"SESSaccess_auth": token})

        with caplog.at_level(logging.INFO, logger="src.auth"):
            user, _ = get_acting_user_from_cookie(request)

        assert user is None
        assert any("AUTH_USER" in r.message for r in caplog.records)
        assert not any("AUTH_INFRA" in r.message for r in caplog.records)
    finally:
        server.shutdown()


def test_tampered_cookie_logs_as_user_state(caplog):
    """A malformed token is user state (or an attack), not an outage."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        request = _mock_request(cookies={"SESSaccess_auth": "this-is-not-a-jwt"})

        with caplog.at_level(logging.INFO, logger="src.auth"):
            user, _ = get_acting_user_from_cookie(request)

        assert user is None
        assert any("AUTH_USER" in r.message for r in caplog.records)
        assert not any("AUTH_INFRA" in r.message for r in caplog.records)
    finally:
        server.shutdown()


def test_jwks_outage_logs_as_infrastructure_at_error(caplog):
    """A JWKS outage is an ops incident, not a user who logged out.

    ``PyJWKClientConnectionError`` is not an ``InvalidTokenError`` subclass, so
    before this classification it fell into the bare ``except Exception`` and
    was indistinguishable from a bad token.
    """
    configure_trusted_issuers({ISSUER: "http://127.0.0.1:1"})
    token = _make_jwt("jsmith@access-ci.org")
    request = _mock_request(cookies={"SESSaccess_auth": token})

    with caplog.at_level(logging.INFO, logger="src.auth"):
        user, cookie_present = get_acting_user_from_cookie(request)

    # Fail-open contract unchanged: an upstream outage degrades to anonymous
    # rather than locking every authenticated user out.
    assert user is None
    assert cookie_present is True

    infra = [r for r in caplog.records if "AUTH_INFRA" in r.message]
    assert infra, "JWKS outage must be logged as AUTH_INFRA"
    assert all(r.levelno >= logging.ERROR for r in infra)
    assert not any("AUTH_USER" in r.message for r in caplog.records)


def test_untrusted_issuer_is_not_log_injectable(caplog):
    """The issuer is attacker-controlled; a newline in it must not forge log
    lines. Unquoted, an attacker could fabricate fake AUTH_INFRA outage
    records and defeat this classification."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        evil = "https://evil.example\nERROR:src.auth:AUTH_INFRA: FORGED OUTAGE"
        token = _make_jwt("jsmith@access-ci.org", issuer=evil)
        request = _mock_request(cookies={"SESSaccess_auth": token})

        with caplog.at_level(logging.INFO, logger="src.auth"):
            user, _ = get_acting_user_from_cookie(request)

        assert user is None
        # The payload text may appear *inside* the escaped issuer string —
        # that is fine. What must not happen is a raw newline reaching the
        # log, which is what would split one record into two and let the
        # attacker forge the second.
        rendered = [r.getMessage() for r in caplog.records]
        assert rendered, "the untrusted issuer must still be logged"
        assert all("\n" not in m for m in rendered), "issuer must be escaped, not raw"
        # And the forged text must be quoted as data, not standing alone as
        # its own record.
        assert all(not m.startswith("ERROR:") for m in rendered)
    finally:
        server.shutdown()


def test_unknown_kid_is_warning_not_error(caplog):
    """A bogus `kid` is attacker-controlled and expected during key rotation,
    so it must not let anonymous callers raise ERROR-level alerts."""
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)
        token = jwt.encode(
            {
                "iss": ISSUER,
                "sub": "jsmith@access-ci.org",
                "iat": int(time.time()) - 60,
                "exp": int(time.time()) + 3600,
            },
            PRIVATE_PEM,
            algorithm="ES256",
            headers={"kid": "no-such-key-id"},
        )
        request = _mock_request(cookies={"SESSaccess_auth": token})

        with caplog.at_level(logging.INFO, logger="src.auth"):
            user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
        assert caplog.records, "the kid miss must still be logged"
        assert all(r.levelno < logging.ERROR for r in caplog.records), (
            "an attacker-supplied kid must not generate ERROR alerts"
        )
    finally:
        server.shutdown()


def test_unexpected_error_logs_as_infrastructure_at_error(caplog, monkeypatch):
    """An unclassified failure is treated as infrastructure, not user state.

    We cannot know that an unrecognized exception means the user's token is
    bad, so it must not be filed alongside expired/tampered cookies.
    """
    server, jwks_url = _start_jwks_server()
    try:
        _setup_issuers(jwks_url)

        def _boom(*args, **kwargs):
            raise MemoryError("something unrecognized went wrong")

        # _decode_jwt resolves the signing key via get_signing_keys()/match_kid
        # (see _get_signing_key's rate-limited lookup), not
        # get_signing_key_from_jwt directly — patch the method actually on
        # the call path so this still exercises an unclassified failure
        # during key resolution.
        monkeypatch.setattr(_jwks_clients[ISSUER], "get_signing_keys", _boom)
        token = _make_jwt("jsmith@access-ci.org")
        request = _mock_request(cookies={"SESSaccess_auth": token})

        with caplog.at_level(logging.INFO, logger="src.auth"):
            user, cookie_present = get_acting_user_from_cookie(request)

        assert user is None
        assert cookie_present is True
        infra = [r for r in caplog.records if "AUTH_INFRA" in r.message]
        assert infra
        assert all(r.levelno >= logging.ERROR for r in infra)
        assert not any("AUTH_USER" in r.message for r in caplog.records)
    finally:
        server.shutdown()


def test_no_issuers_configured_logs_as_infrastructure_at_error(caplog):
    """An empty TRUSTED_JWKS_URLS silently anonymizes every logged-in user —
    a deploy problem that must not look like ordinary logged-out traffic."""
    configure_trusted_issuers({})
    token = _make_jwt("jsmith@access-ci.org")
    request = _mock_request(cookies={"SESSaccess_auth": token})

    with caplog.at_level(logging.INFO, logger="src.auth"):
        user, _ = get_acting_user_from_cookie(request)

    assert user is None
    infra = [r for r in caplog.records if "AUTH_INFRA" in r.message]
    assert infra
    assert all(r.levelno >= logging.ERROR for r in infra)


# --- Unknown-kid amplification DoS (forced JWKS refresh rate limit) ---
#
# `kid` is an attacker-controlled JWT header field. PyJWKClient.get_signing_key
# forces a live outbound fetch to the issuer's JWKS endpoint on ANY cache miss
# before raising. Without a per-issuer cooldown, an anonymous caller can send a
# fresh bogus `kid` on every request and force one outbound fetch per request —
# amplification against the real issuer's JWKS endpoint. These tests count hits
# against a local JWKS server to prove the refresh is bounded, not per-request,
# while still allowing a genuinely rotated key to be picked up promptly.


class _CountingJWKSHandler(BaseHTTPRequestHandler):
    """Same JWKS response as `_JWKSHandler`, but records a hit per GET."""

    hits: ClassVar[list[float]] = []

    def do_GET(self):
        type(self).hits.append(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(JWKS_RESPONSE)

    def log_message(self, format, *args):
        pass


def _start_counting_jwks_server() -> tuple[HTTPServer, str, list[float]]:
    """Start a local JWKS server that records request timestamps.

    Returns (server, url, hits) — `hits` is the handler class's own list, so
    the caller can inspect fetch counts without instance plumbing.
    """
    hits: list[float] = []

    class _Handler(_CountingJWKSHandler):
        pass

    _Handler.hits = hits
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}", hits


def _make_bogus_kid_jwt(kid: str, issuer: str = ISSUER) -> str:
    """An unsigned-relative-to-the-real-key JWT with an attacker-chosen kid.

    Only the header `kid`/`iss` need to be attacker-plausible to trigger a
    cache lookup — the signature never verifies, by construction, since the
    real private key is never used. That's enough to exercise the fetch path:
    PyJWKClient's forced refresh happens on the `kid` miss, before signature
    verification is reached.
    """
    now = int(time.time())
    return jwt.encode(
        {"iss": issuer, "sub": "attacker@evil.example", "exp": now + 3600},
        PRIVATE_PEM,  # arbitrary key; signature is never checked (invalid kid)
        algorithm="ES256",
        headers={"kid": kid},
    )


def test_unknown_kid_amplification_is_rate_limited(monkeypatch):
    """N requests with N DIFFERENT bogus kids must force ~1 refresh, not N.

    This is the amplification-DoS proof: before the fix, every unknown kid
    forces a live outbound JWKS fetch (PyJWKClient.get_signing_key refreshes
    on any cache miss). An anonymous attacker varying `kid` per request could
    force one outbound fetch per request indefinitely.
    """
    server, jwks_url, hits = _start_counting_jwks_server()
    try:
        _setup_issuers(jwks_url)
        # Prime the cache so the *first* unknown kid in the loop below isn't
        # conflated with "first ever lookup" bookkeeping.
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": _make_jwt("x")}))
        hits.clear()

        now = time.time()
        monkeypatch.setattr(time, "time", lambda: now)

        for i in range(10):
            token = _make_bogus_kid_jwt(kid=f"bogus-kid-{i}")
            request = _mock_request(cookies={"SESSaccess_auth": token})
            user, cookie_present = get_acting_user_from_cookie(request)
            assert user is None
            assert cookie_present is True

        assert len(hits) <= 1, (
            f"expected at most 1 forced refresh within the cooldown window, got {len(hits)}"
        )
    finally:
        server.shutdown()


def test_unknown_kid_refresh_allowed_again_after_cooldown(monkeypatch):
    """After the cooldown elapses, a new unknown kid is allowed to refresh again.

    Proves the rate limit doesn't permanently wedge refreshes shut — it only
    bounds how often an unknown kid can force one.
    """
    server, jwks_url, hits = _start_counting_jwks_server()
    try:
        _setup_issuers(jwks_url)
        # Prime the cache (an empty cache fetches once even without a forced
        # refresh) so the counts below isolate forced refreshes only.
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": _make_jwt("x")}))
        hits.clear()

        now = time.time()
        monkeypatch.setattr(time, "time", lambda: now)

        token1 = _make_bogus_kid_jwt(kid="bogus-kid-a")
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": token1}))
        assert len(hits) == 1

        # Still within cooldown — a second bogus kid must not refresh again.
        monkeypatch.setattr(time, "time", lambda: now + JWKS_REFRESH_COOLDOWN_S - 1)
        token2 = _make_bogus_kid_jwt(kid="bogus-kid-b")
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": token2}))
        assert len(hits) == 1

        # Cooldown elapsed — the next unknown kid is allowed to refresh.
        monkeypatch.setattr(time, "time", lambda: now + JWKS_REFRESH_COOLDOWN_S)
        token3 = _make_bogus_kid_jwt(kid="bogus-kid-c")
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": token3}))
        assert len(hits) == 2
    finally:
        server.shutdown()


def test_unknown_kid_suppressed_refresh_logs_at_debug_not_warning(caplog, monkeypatch):
    """The suppressed-refresh decision itself (still within cooldown) must
    log at DEBUG, not WARNING/ERROR — it's attacker-controlled input and must
    not become a log-spam or alert oracle. The pre-existing "unknown key id"
    WARNING from the eventual PyJWKClientError is unchanged (see
    test_unknown_kid_is_warning_not_error) and still fires; what must never
    happen is anything reaching ERROR, and the suppression line itself must
    be DEBUG."""
    server, jwks_url, hits = _start_counting_jwks_server()
    try:
        _setup_issuers(jwks_url)
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": _make_jwt("x")}))
        hits.clear()

        now = time.time()
        monkeypatch.setattr(time, "time", lambda: now)

        token1 = _make_bogus_kid_jwt(kid="bogus-kid-a")
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": token1}))
        assert len(hits) == 1

        with caplog.at_level(logging.DEBUG, logger="src.auth"):
            token2 = _make_bogus_kid_jwt(kid="bogus-kid-b")
            user, _ = get_acting_user_from_cookie(
                _mock_request(cookies={"SESSaccess_auth": token2})
            )

        assert user is None
        assert len(hits) == 1
        suppression_records = [r for r in caplog.records if "suppressing forced JWKS" in r.message]
        assert suppression_records, "expected a suppressed-refresh log line"
        assert all(r.levelno == logging.DEBUG for r in suppression_records)
        assert all(r.levelno < logging.ERROR for r in caplog.records), (
            "an attacker-supplied kid must never generate ERROR alerts"
        )
    finally:
        server.shutdown()


def test_legit_key_rotation_still_resolves_after_cooldown(monkeypatch):
    """A kid missing from the cached set but present after a refresh (real
    rotation), with cooldown elapsed, must still resolve the user — the rate
    limit must not break legitimate key rotation.
    """
    server, jwks_url, hits = _start_counting_jwks_server()
    try:
        _setup_issuers(jwks_url)
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": _make_jwt("x")}))
        hits.clear()

        now = time.time()
        monkeypatch.setattr(time, "time", lambda: now)

        # Burn the cooldown with an initial (bogus) unknown-kid lookup so the
        # rotated-key lookup below has to wait out a real cooldown window,
        # same as it would in production.
        burn_token = _make_bogus_kid_jwt(kid="bogus-kid-burn")
        get_acting_user_from_cookie(_mock_request(cookies={"SESSaccess_auth": burn_token}))
        assert len(hits) == 1

        # Rotate: a new key (new kid) appears at the JWKS endpoint.
        new_key = ec.generate_private_key(ec.SECP256R1())
        new_public_pem = new_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
        new_public_key = new_key.public_key()
        new_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(
            new_public_key
        )
        new_jwk = json.loads(new_json_key)
        new_jwk["kid"] = "rotated-kid"
        new_jwk["use"] = "sig"
        new_jwk["alg"] = "ES256"

        # Swap the module-level JWKS_RESPONSE the handler serves, mid-test.
        monkeypatch.setattr(
            sys.modules[__name__],
            "JWKS_RESPONSE",
            json.dumps({"keys": [_jwk_dict, new_jwk]}).encode(),
        )

        # Cooldown elapsed — the rotated kid should force exactly one more
        # refresh and resolve successfully.
        monkeypatch.setattr(time, "time", lambda: now + JWKS_REFRESH_COOLDOWN_S)
        rotated_token = jwt.encode(
            {
                "iss": ISSUER,
                "sub": "rotated-user@access-ci.org",
                "iat": int(now) - 60,
                "exp": int(now) + 3600,
            },
            new_public_pem,
            algorithm="ES256",
            headers={"kid": "rotated-kid"},
        )
        user, cookie_present = get_acting_user_from_cookie(
            _mock_request(cookies={"SESSaccess_auth": rotated_token})
        )

        assert user == "rotated-user@access-ci.org"
        assert cookie_present is True
        assert len(hits) == 2
    finally:
        server.shutdown()
