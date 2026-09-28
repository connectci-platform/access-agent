"""Tests for POST /threads and POST /threads/{id}/history.

Harness copied from tests/test_auth_e2e.py (JWKS server + ES256 JWT cookie
minting) — do not invent a new signing harness. Ownership is seeded directly
via get_thread_owner_store().claim_thread(...) (Task 2's store), and the
checkpointer is a fake set directly on app.state.checkpointer (mirrors
tests/test_pooled_checkpointer.py's FakeCheckpointerCM pattern) so these
tests need no live Postgres.
"""

import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Any

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
from langchain_core.messages import HumanMessage

from src.auth import configure_trusted_issuers
from src.main import app
from src.thread_owners import get_thread_owner_store

# ---------------------------------------------------------------------------
# Test key pair and JWKS server (copied from tests/test_auth_e2e.py:44-97)
# ---------------------------------------------------------------------------

_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "thread-routes-test-kid"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "thread-routes-test-kid"


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


class FakeStateSnapshot:
    """Stands in for langgraph's StateSnapshot — only the fields the route reads."""

    def __init__(
        self,
        values: dict[str, Any],
        checkpoint_id: str,
        parent_checkpoint_id: str | None,
        metadata: dict[str, Any],
        created_at: str,
    ) -> None:
        self.values = values
        self.config = {"configurable": {"checkpoint_id": checkpoint_id}}
        self.parent_config = (
            {"configurable": {"checkpoint_id": parent_checkpoint_id}}
            if parent_checkpoint_id
            else None
        )
        self.metadata = metadata
        self.created_at = created_at


class FakeCheckpointer:
    """Fake graph-history source: app.state.checkpointer plus a monkeypatched
    create_checkpointed_graph that returns an object exposing aget_state_history.
    """


class FakeGraph:
    def __init__(self, snapshots: list[FakeStateSnapshot]) -> None:
        self._snapshots = snapshots

    async def aget_state_history(self, config, limit=10):
        for snap in self._snapshots[:limit]:
            yield snap


# ---------------------------------------------------------------------------
# POST /threads
# ---------------------------------------------------------------------------


async def test_create_thread_requires_cookie(client):
    r = await client.post("/api/v1/threads")
    assert r.status_code == 401


async def test_create_thread_returns_fresh_thread(client, cookie_for):
    r = await client.post("/api/v1/threads", cookies=cookie_for("userA@x"))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "idle"
    assert body["metadata"] == {}
    assert body["values"] == {}
    assert "thread_id" in body
    assert "created_at" in body


async def test_create_thread_returns_distinct_ids(client, cookie_for):
    cookies = cookie_for("userA@x")
    r1 = await client.post("/api/v1/threads", cookies=cookies)
    r2 = await client.post("/api/v1/threads", cookies=cookies)
    assert r1.json()["thread_id"] != r2.json()["thread_id"]


# ---------------------------------------------------------------------------
# POST /threads/{id}/history
# ---------------------------------------------------------------------------


async def test_history_requires_cookie(client):
    r = await client.post("/api/v1/threads/some-thread/history", json={"limit": 5})
    assert r.status_code == 401


async def test_history_non_owner_gets_404(client, cookie_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    r = await client.post(
        "/api/v1/threads/t-owned/history",
        json={"limit": 5},
        cookies=cookie_for("userB@x"),
    )
    assert r.status_code == 404


async def test_history_unknown_thread_gets_404_identical_body(client, cookie_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    non_owner = await client.post(
        "/api/v1/threads/t-owned/history",
        json={"limit": 5},
        cookies=cookie_for("userB@x"),
    )
    unknown = await client.post(
        "/api/v1/threads/does-not-exist/history",
        json={"limit": 5},
        cookies=cookie_for("userB@x"),
    )
    assert non_owner.status_code == unknown.status_code == 404
    assert non_owner.json() == unknown.json()


async def test_history_owner_gets_state_history(client, cookie_for, monkeypatch):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    snapshots = [
        FakeStateSnapshot(
            values={"messages": [HumanMessage(content="hello")]},
            checkpoint_id="cp-2",
            parent_checkpoint_id="cp-1",
            metadata={"step": 1},
            created_at="2026-09-28T00:00:00Z",
        ),
        FakeStateSnapshot(
            values={"messages": []},
            checkpoint_id="cp-1",
            parent_checkpoint_id=None,
            metadata={"step": 0},
            created_at="2026-09-27T23:59:00Z",
        ),
    ]
    fake_graph = FakeGraph(snapshots)

    app.state.checkpointer = object()  # non-None so the route proceeds
    monkeypatch.setattr(
        "src.api.thread_routes.create_checkpointed_graph",
        lambda checkpointer: fake_graph,
    )
    try:
        r = await client.post(
            "/api/v1/threads/t-owned/history",
            json={"limit": 10},
            cookies=cookie_for("userA@x"),
        )
    finally:
        app.state.checkpointer = None

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    assert body[0]["values"]["messages"][0]["content"] == "hello"
    assert body[0]["checkpoint"]["checkpoint_id"] == "cp-2"
    assert body[0]["parent_checkpoint"]["checkpoint_id"] == "cp-1"
    assert body[0]["metadata"] == {"step": 1}
    assert body[1]["parent_checkpoint"] == {}


async def test_history_no_checkpointer_returns_empty_list(client, cookie_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    app.state.checkpointer = None

    r = await client.post(
        "/api/v1/threads/t-owned/history",
        json={"limit": 10},
        cookies=cookie_for("userA@x"),
    )
    assert r.status_code == 200
    assert r.json() == []
