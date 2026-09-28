"""Tests for POST /threads/{id}/runs/stream (the gated, bounded streaming run)
and the widget-path ownership/lock sharing.

Harness copied from tests/test_thread_routes_history.py (JWKS server + ES256 JWT
cookie minting). stream_agent is mocked so no LLM/MCP is needed. The thread-owner
store is pointed at a temp sqlite file.
"""

import asyncio
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
from langchain_core.messages import AIMessageChunk

from src.auth import configure_trusted_issuers
from src.main import app
from src.thread_owners import get_thread_owner_store

# ---------------------------------------------------------------------------
# Test key pair and JWKS server (copied from tests/test_thread_routes_history.py)
# ---------------------------------------------------------------------------

_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "thread-run-test-kid"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "thread-run-test-kid"


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


def _make_jwt(sub: str, issuer: str = ISSUER) -> str:
    now = int(time.time())
    payload = {
        "iss": issuer,
        "sub": sub,
        "iat": now - 3600,
        "exp": now + 3600,
    }
    return jwt.encode(payload, PRIVATE_PEM, algorithm="ES256", headers={"kid": KID})


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _jwks_server():
    server, jwks_url = _start_jwks_server()
    configure_trusted_issuers({ISSUER: jwks_url})
    yield
    server.shutdown()
    configure_trusted_issuers({})


@pytest.fixture(autouse=True)
def _sqlite_owner_store(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}")
    import src.thread_owners as m

    m._store = None  # reset singleton


@pytest.fixture(autouse=True)
def _reset_run_registry():
    """Clear the in-process per-thread run registry between tests."""
    import src.api.thread_runs as tr

    tr._active_by_thread.clear()
    tr._task_by_run.clear()
    yield
    tr._active_by_thread.clear()
    tr._task_by_run.clear()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        app.state.checkpointer = None
        yield c


@pytest.fixture
def valid_cookie_for():
    def _make(sub: str) -> dict[str, str]:
        return {"SESSaccess_auth": _make_jwt(sub)}

    return _make


@pytest.fixture
def valid_cookie(valid_cookie_for):
    return valid_cookie_for("runner@x")


@pytest.fixture
def seed_owner():
    def _seed(thread_id: str, user: str) -> None:
        get_thread_owner_store().claim_thread(thread_id, user)

    return _seed


def _mock_stream_agent(monkeypatch, *, content: str = "hi", msg_id: str = "m1"):
    """Patch thread_routes.stream_agent to yield a canned messages chunk + updates."""

    async def _fake(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content=content, id=msg_id),
                {"langgraph_node": "tool_calling_loop"},
            ),
        )
        yield "updates", {"tool_calling_loop": {"final_answer": content}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)


# ---------------------------------------------------------------------------
# POST /threads/{id}/runs/stream
# ---------------------------------------------------------------------------


async def test_run_stream_emits_protocol_events(client, valid_cookie, monkeypatch):
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-new/runs/stream",
        cookies=valid_cookie,
        json={
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 200
    body = r.text
    assert "event: metadata" in body
    assert "event: messages/partial" in body
    assert "event: messages/complete" in body
    assert "event: values" in body
    assert "event: end" not in body  # protocol has no end terminator


async def test_run_requires_cookie(client):
    r = await client.post(
        "/api/v1/threads/t/runs/stream",
        json={"input": {"messages": []}},
    )
    assert r.status_code == 401


async def test_run_on_another_users_thread_404(client, valid_cookie_for, seed_owner, monkeypatch):
    _mock_stream_agent(monkeypatch)
    seed_owner("t-owned", "alice@x")
    r = await client.post(
        "/api/v1/threads/t-owned/runs/stream",
        cookies=valid_cookie_for("bob@x"),
        json={
            "input": {"messages": [{"role": "user", "content": "x"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 404


async def test_concurrent_run_rejected(client, valid_cookie, monkeypatch):
    release = asyncio.Event()

    async def _blocking_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "tool_calling_loop"},
            ),
        )
        await release.wait()
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _blocking_stream)

    body = {
        "input": {"messages": [{"role": "user", "content": "hi"}]},
        "if_not_exists": "create",
    }

    async def _fire():
        return await client.post(
            "/api/v1/threads/t-busy/runs/stream", cookies=valid_cookie, json=body
        )

    first_task = asyncio.create_task(_fire())
    # Give the first run time to acquire the lock before firing the second.
    for _ in range(50):
        await asyncio.sleep(0.01)
        import src.api.thread_runs as tr

        if "t-busy" in tr._active_by_thread:
            break

    second = await _fire()
    assert second.status_code == 409

    release.set()
    await first_task


# ---------------------------------------------------------------------------
# Step 6: widget path — ownership + shared lock
# ---------------------------------------------------------------------------


async def test_widget_and_fullscreen_share_lock(client, valid_cookie, monkeypatch):
    """A widget /query and a /runs/stream on the same session_id can't run at once."""
    release = asyncio.Event()

    async def _blocking_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "tool_calling_loop"},
            ),
        )
        await release.wait()
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    async def _fake_registry():
        from types import SimpleNamespace

        return SimpleNamespace(catalog={"servers": []})

    monkeypatch.setattr("src.api.thread_runs._active_by_thread", {}, raising=False)
    monkeypatch.setattr("src.api.routes.stream_agent", _blocking_stream)
    monkeypatch.setattr("src.api.routes.get_registry", _fake_registry)
    monkeypatch.setattr("src.api.thread_routes.stream_agent", _blocking_stream)

    async def _fire_widget():
        return await client.post(
            "/api/v1/query",
            cookies=valid_cookie,
            json={"query": "hi", "session_id": "shared-sess"},
        )

    widget_task = asyncio.create_task(_fire_widget())
    for _ in range(50):
        await asyncio.sleep(0.01)
        import src.api.thread_runs as tr

        if "shared-sess" in tr._active_by_thread:
            break

    fullscreen = await client.post(
        "/api/v1/threads/shared-sess/runs/stream",
        cookies=valid_cookie,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert fullscreen.status_code == 409

    release.set()
    await widget_task


async def test_widget_thread_owned_by_creator(client, valid_cookie_for, monkeypatch):
    """A widget turn claims ownership; a stranger's runs/stream on it → 404."""

    async def _fake_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "tool_calling_loop"},
            ),
        )
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    async def _fake_registry():
        from types import SimpleNamespace

        return SimpleNamespace(catalog={"servers": []})

    monkeypatch.setattr("src.api.routes.stream_agent", _fake_stream)
    monkeypatch.setattr("src.api.routes.get_registry", _fake_registry)
    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake_stream)

    # Alice runs a widget turn on a new session → she owns it.
    widget = await client.post(
        "/api/v1/query",
        cookies=valid_cookie_for("alice@x"),
        json={"query": "hi", "session_id": "widget-sess"},
    )
    assert widget.status_code == 200
    _ = widget.text  # drain the stream so the generator's finally releases the lock

    assert get_thread_owner_store().check_access("widget-sess", "alice@x") is True

    # Bob can't run Alice's widget-born thread.
    bob = await client.post(
        "/api/v1/threads/widget-sess/runs/stream",
        cookies=valid_cookie_for("bob@x"),
        json={"input": {"messages": [{"role": "user", "content": "x"}]}},
    )
    assert bob.status_code == 404


async def test_widget_registry_failure_does_not_leak_lock(client, valid_cookie, monkeypatch):
    """get_registry() failing inside the widget stream must release the per-thread lock.

    acquire_thread_run runs in the handler; if get_registry() (MCP catalog fetch,
    can fail on a cold/refreshing cache) raised OUTSIDE held_run, the lock would
    never be released and the session_id would be permanently 409-locked.
    """

    async def _boom_registry():
        raise RuntimeError("catalog cold")

    monkeypatch.setattr("src.api.routes.get_registry", _boom_registry)

    r = await client.post(
        "/api/v1/query",
        cookies=valid_cookie,
        json={"query": "hi", "session_id": "leak-sess"},
    )
    assert r.status_code == 200
    _ = r.text  # drain the stream; the error path runs and held_run's finally fires

    import src.api.thread_runs as tr

    assert "leak-sess" not in tr._active_by_thread  # lock released, not leaked
    # A subsequent acquire on the same session must NOT 409.
    tr.acquire_thread_run("leak-sess", "next-run")  # raises HTTPException(409) if leaked
    tr.release_thread_run("leak-sess", "next-run")
