"""Tests for POST /threads/{id}/handoff and POST /handoff/exchange.

Harness copied from tests/test_thread_routes_history.py (JWKS server + ES256
JWT cookie minting, sqlite-backed thread-owner store) — do not invent a new
auth-mocking approach. The handoff token store itself is in-memory by
default (no REDIS_URL configured in tests), reset between tests via
src.api.handoff_tokens._reset_store_for_test().
"""

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from httpx import ASGITransport, AsyncClient
from jwt import algorithms as jwt_algorithms

from src.api.handoff_tokens import _reset_store_for_test
from src.auth import configure_trusted_issuers
from src.main import app
from src.thread_owners import get_thread_owner_store

# ---------------------------------------------------------------------------
# Test key pair and JWKS server (copied from tests/test_auth_e2e.py:44-97,
# mirrored in tests/test_thread_routes_history.py)
# ---------------------------------------------------------------------------

_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "handoff-test-kid"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "handoff-test-kid"


class _JWKSHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(JWKS_RESPONSE)

    def log_message(self, format, *args):
        pass


def _start_jwks_server() -> tuple[HTTPServer, str]:
    server = HTTPServer(("127.0.0.1", 0), _JWKSHandler)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{port}"


def _make_jwt(sub: str, expired: bool = False, issuer: str = ISSUER) -> str:
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _jwks_server():
    """Start a local JWKS server and configure trusted issuers for all tests."""
    server, jwks_url = _start_jwks_server()
    configure_trusted_issuers({ISSUER: jwks_url})
    yield
    server.shutdown()
    configure_trusted_issuers({})


@pytest.fixture(autouse=True)
def _sqlite_owner_store(monkeypatch, tmp_path):
    """Point the thread-owner store at a temp sqlite file (mirrors test_thread_owners.py)."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.thread_owners as m

    m._store = None  # reset singleton


@pytest.fixture(autouse=True)
def _reset_handoff_store():
    """In-memory handoff token store is module-global; reset between tests."""
    _reset_store_for_test()
    yield
    _reset_store_for_test()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def cookie_for():
    """Returns a dict cookie mapping for the given user sub."""

    def _make(sub: str) -> dict[str, str]:
        return {"SESSaccess_auth": _make_jwt(sub)}

    return _make


# ---------------------------------------------------------------------------
# POST /threads/{id}/handoff (mint)
# ---------------------------------------------------------------------------


async def test_mint_requires_cookie(client):
    r = await client.post("/api/v1/threads/some-thread/handoff")
    assert r.status_code == 401


async def test_mint_non_owner_gets_404(client, cookie_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    r = await client.post(
        "/api/v1/threads/t-owned/handoff",
        cookies=cookie_for("userB@x"),
    )
    assert r.status_code == 404


async def test_mint_anon_owned_thread_gets_404(client, cookie_for):
    """Anon-owned threads are never eligible for handoff — an anon thread_id is a
    credential that must never leak into a URL via a minted token.

    NOTE ON WHAT THIS PROVES: this asserts the observable behavior (anon thread →
    404), but it cannot isolate the `not owner.was_authenticated` guard from the
    owner-match check below it. An anon row has user_hash=None, so even without
    the anon guard the request would fall through to `_hash_user(caller) != None`
    and still 404. That is not a test gap that can be closed: the state the anon
    guard uniquely defends — was_authenticated=False WITH a non-null user_hash —
    is unreachable (only upgrade_owner writes a non-null hash, and it also flips
    was_authenticated=True). The guard is therefore fail-closed defense-in-depth,
    not the sole load-bearing check for this scenario. See mint_handoff's
    docstring in thread_routes.py."""
    get_thread_owner_store().claim_thread("t-anon", None)

    r = await client.post(
        "/api/v1/threads/t-anon/handoff",
        cookies=cookie_for("userA@x"),
    )
    assert r.status_code == 404


async def test_mint_owner_gets_token(client, cookie_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    r = await client.post(
        "/api/v1/threads/t-owned/handoff",
        cookies=cookie_for("userA@x"),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["expires_in"] == 60
    assert isinstance(body["token"], str) and body["token"]


# ---------------------------------------------------------------------------
# POST /handoff/exchange
# ---------------------------------------------------------------------------


async def test_exchange_requires_cookie(client):
    r = await client.post("/api/v1/handoff/exchange", json={"token": "whatever"})
    assert r.status_code == 401


async def test_exchange_same_identity_returns_thread_id(client, cookie_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    mint = await client.post(
        "/api/v1/threads/t-owned/handoff",
        cookies=cookie_for("userA@x"),
    )
    token = mint.json()["token"]

    r = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        cookies=cookie_for("userA@x"),
    )
    assert r.status_code == 200
    assert r.json() == {"thread_id": "t-owned"}


async def test_exchange_different_identity_gets_404(client, cookie_for):
    """Identity binding: a leaked token exchanged by a different caller must
    fail even though it is otherwise valid and unexpired."""
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    mint = await client.post(
        "/api/v1/threads/t-owned/handoff",
        cookies=cookie_for("userA@x"),
    )
    token = mint.json()["token"]

    r = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        cookies=cookie_for("userB@x"),
    )
    assert r.status_code == 404


async def test_exchange_is_single_use(client, cookie_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    mint = await client.post(
        "/api/v1/threads/t-owned/handoff",
        cookies=cookie_for("userA@x"),
    )
    token = mint.json()["token"]

    first = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        cookies=cookie_for("userA@x"),
    )
    second = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        cookies=cookie_for("userA@x"),
    )
    assert first.status_code == 200
    assert second.status_code == 404


async def test_exchange_response_contains_only_thread_id(client, cookie_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    mint = await client.post(
        "/api/v1/threads/t-owned/handoff",
        cookies=cookie_for("userA@x"),
    )
    token = mint.json()["token"]

    r = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        cookies=cookie_for("userA@x"),
    )
    body = r.json()
    assert set(body.keys()) == {"thread_id"}
    assert "messages" not in body
    assert "values" not in body
