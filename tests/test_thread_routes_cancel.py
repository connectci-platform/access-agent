"""Tests for POST /threads/{id}/runs/{run_id}/cancel (explicit protocol cancel).

Harness copied from tests/test_thread_routes_run.py (JWKS server + ES256 JWT
cookie minting, sqlite thread-owner store, in-process run registry reset).
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

from src.auth import configure_trusted_issuers
from src.main import app
from src.thread_owners import get_thread_owner_store

# ---------------------------------------------------------------------------
# Test key pair and JWKS server (copied from tests/test_thread_routes_run.py)
# ---------------------------------------------------------------------------

_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "thread-cancel-test-kid"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "thread-cancel-test-kid"


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
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
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


# ---------------------------------------------------------------------------
# POST /threads/{thread_id}/runs/{run_id}/cancel
# ---------------------------------------------------------------------------


async def test_cancel_anon_caller_unknown_thread_404(client):
    """Anon (no cookie) caller on a thread with no owner row: 404, not 401."""
    r = await client.post("/api/v1/threads/t/runs/r/cancel")
    assert r.status_code == 404


async def test_cancel_anon_caller_on_authed_owned_thread_404(client, seed_owner):
    """Anon (no cookie) caller hitting an existing AUTHED-owned thread's
    cancel is a non-owner: 404, not 401/403."""
    seed_owner("t-owned-authed", "alice@x")
    r = await client.post("/api/v1/threads/t-owned-authed/runs/whatever/cancel")
    assert r.status_code == 404


async def test_cancel_non_owner_404(client, valid_cookie_for, seed_owner):
    seed_owner("t-owned", "alice@x")
    r = await client.post(
        "/api/v1/threads/t-owned/runs/whatever/cancel",
        cookies=valid_cookie_for("bob@x"),
    )
    assert r.status_code == 404  # not bob's thread → indistinguishable from unknown


async def test_cancel_unknown_run_404(client, valid_cookie_for, seed_owner):
    seed_owner("t-mine", "me@x")
    r = await client.post(
        "/api/v1/threads/t-mine/runs/no-such-run/cancel",
        cookies=valid_cookie_for("me@x"),
    )
    assert r.status_code == 404  # owner, but no in-flight run with that id


async def test_cancel_owner_cancels_in_flight_run(client, valid_cookie_for, seed_owner):
    """Owner cancels a real in-flight run → 200 {"status": "cancelled"}.

    Registers a real asyncio task into the run registry (via held_run) so
    cancel_run(run_id) has something to find, then cancels it through the
    endpoint while the task is still alive.
    """
    seed_owner("t-live", "me@x")

    import src.api.thread_runs as tr

    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def _long_running_run() -> None:
        async with tr.held_run("t-live", "run-1"):
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

    tr.acquire_thread_run("t-live", "run-1")
    task = asyncio.create_task(_long_running_run())
    await started.wait()

    r = await client.post(
        "/api/v1/threads/t-live/runs/run-1/cancel",
        cookies=valid_cookie_for("me@x"),
    )
    assert r.status_code == 200
    assert r.json() == {"status": "cancelled"}

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert "t-live" not in tr._active_by_thread  # held_run's finally released the lock


async def test_cancel_cannot_cross_threads(client, valid_cookie_for, seed_owner):
    """Owner of thread A cannot cancel thread B's run even with B's real run_id.

    run_ids are disclosed to clients in the metadata SSE event. A caller who owns
    thread A and observed B's run_id must get a 404 (not a cancel) and B's task
    must keep running. Guards the cross-thread cancel (Finding I1).
    """
    seed_owner("thread-a", "attacker@x")  # attacker owns thread-a

    import src.api.thread_runs as tr

    started = asyncio.Event()

    async def _run_on_b() -> None:
        async with tr.held_run("thread-b", "run-b"):
            started.set()
            await asyncio.sleep(10)

    tr.acquire_thread_run("thread-b", "run-b")
    task = asyncio.create_task(_run_on_b())
    await started.wait()

    # Attacker authenticates as thread-a's owner, presents B's run_id under A.
    r = await client.post(
        "/api/v1/threads/thread-a/runs/run-b/cancel",
        cookies=valid_cookie_for("attacker@x"),
    )
    assert r.status_code == 404  # run-b is not active on thread-a
    assert not task.done()  # B's run is untouched

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
