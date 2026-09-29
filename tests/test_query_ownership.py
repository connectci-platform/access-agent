"""Ownership enforcement on the widget /api/v1/query resume path.

Covers defect #1 from the tenant-isolation audit: the widget path resolved
cookie-only identity and called claim_thread, but never called the existing
check_access gate — so an authenticated caller who was not the thread's owner
could resume/append to someone else's authed-owned thread by supplying its
session_id. Mirrors the runs/stream gate in src/api/thread_routes.py (404,
never 403, on mismatch — identical body to an unknown thread).

Harness mirrors tests/test_query_anon_session_security.py (sqlite-pinned
owner store, mocked stream_agent/get_registry) and tests/test_auth_e2e.py
(ES256 JWT cookie helpers).
"""

import json
import time

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

_ec_private_key = ec.generate_private_key(ec.SECP256R1())
_ec_public_key = _ec_private_key.public_key()

PRIVATE_PEM = _ec_private_key.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
PUBLIC_PEM = _ec_public_key.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

_ec_json_key = jwt_algorithms.ECAlgorithm(jwt_algorithms.ECAlgorithm.SHA256).to_jwk(_ec_public_key)
_jwk_dict = json.loads(_ec_json_key)
_jwk_dict["kid"] = "ownership-test-kid"
_jwk_dict["use"] = "sig"
_jwk_dict["alg"] = "ES256"

JWKS_RESPONSE = json.dumps({"keys": [_jwk_dict]}).encode()

ISSUER = "https://test-issuer.access-ci.org"
KID = "ownership-test-kid"

FAKE_AGENT_RESULT = {
    "final_answer": "Test response",
    "tools_used": [],
    "query_analysis": None,
    "query_classification": None,
}


def _make_jwt(sub: str) -> str:
    now = int(time.time())
    payload = {"iss": ISSUER, "sub": sub, "iat": now - 3600, "exp": now + 3600}
    return jwt.encode(payload, PRIVATE_PEM, algorithm="ES256", headers={"kid": KID})


@pytest.fixture(autouse=True)
def _jwks_server():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread

    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(JWKS_RESPONSE)

        def log_message(self, format, *args):
            pass

    from src.auth import configure_trusted_issuers

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    configure_trusted_issuers({ISSUER: f"http://127.0.0.1:{port}"})
    yield
    server.shutdown()
    configure_trusted_issuers({})


@pytest.fixture(autouse=True)
def _no_turnstile(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch, tmp_path):
    # Pin the owner store at a temp sqlite file — mirrors tests/test_thread_owners.py
    # and tests/test_query_anon_session_security.py's autouse pin. Without this the
    # widget claim_thread/check_access calls hit whatever DATABASE_URL is exported
    # in the dev shell.
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.thread_owners as owners

    owners._store = None
    yield
    owners._store = None


@pytest.fixture(autouse=True)
def _reset_run_registry():
    import src.api.thread_runs as tr

    tr._active_by_thread.clear()
    tr._task_by_run.clear()
    yield
    tr._active_by_thread.clear()
    tr._task_by_run.clear()


@pytest.fixture
def mock_agent():
    from unittest.mock import patch

    async def fake_stream(**kwargs):
        yield "updates", {"__end__": FAKE_AGENT_RESULT}

    with patch("src.api.routes.stream_agent", side_effect=fake_stream) as mock:
        yield mock


@pytest.fixture
def mock_registry():
    from unittest.mock import AsyncMock, patch

    mock_reg = AsyncMock()
    mock_reg.catalog = {"tools": [], "quick_lookup": {}}
    with patch("src.api.routes.get_registry", return_value=mock_reg):
        yield mock_reg


@pytest.fixture
async def client():
    from src.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def valid_cookie_for():
    def _make(sub: str) -> dict[str, str]:
        token = _make_jwt(sub)
        return {"cookie": f"SESSaccess_auth={token}"}

    return _make


@pytest.fixture
def seed_owner():
    def _seed(thread_id: str, user: str | None) -> None:
        from src.thread_owners import get_thread_owner_store

        get_thread_owner_store().claim_thread(thread_id, user)

    return _seed


async def test_authed_nonowner_cannot_resume_widget_thread(
    client, valid_cookie_for, seed_owner, mock_agent, mock_registry
):
    """Bob presenting alice's authed-owned thread id must be denied, 404 (not 403)."""
    seed_owner("t-A", "alice@access-ci.org")

    r = await client.post(
        "/api/v1/query",
        headers=valid_cookie_for("bob@access-ci.org"),
        json={"query": "hi", "session_id": "t-A"},
    )

    assert r.status_code == 404
    mock_agent.assert_not_called()


async def test_owner_can_resume_own_widget_thread(
    client, valid_cookie_for, seed_owner, mock_agent, mock_registry
):
    seed_owner("t-mine", "me@access-ci.org")

    r = await client.post(
        "/api/v1/query",
        headers=valid_cookie_for("me@access-ci.org"),
        json={"query": "hi", "session_id": "t-mine"},
    )

    assert r.status_code == 200
    mock_agent.assert_called_once()


async def test_anon_thread_resumable_by_id(client, seed_owner, mock_agent, mock_registry):
    """Capability model: presenting the unguessable id is itself the credential."""
    seed_owner("t-anon", None)

    r = await client.post("/api/v1/query", json={"query": "hi", "session_id": "t-anon"})

    assert r.status_code == 200
    mock_agent.assert_called_once()


async def test_nonowner_denial_body_matches_other_404s(
    client, valid_cookie_for, seed_owner, mock_agent, mock_registry
):
    """The non-owner denial must be a standard 404 body (not a distinguishable
    'forbidden' shape), consistent with the runs/stream gate's identical-body
    contract in src/api/thread_routes.py. (Unlike that endpoint, an *unknown*
    session_id on this widget path legitimately lazy-creates a new thread via
    claim_thread and returns 200 — the widget's capability model, not a leak.)
    """
    seed_owner("t-B", "alice@access-ci.org")

    r = await client.post(
        "/api/v1/query",
        headers=valid_cookie_for("bob@access-ci.org"),
        json={"query": "hi", "session_id": "t-B"},
    )

    assert r.status_code == 404
    assert r.json() == {"detail": "Thread not found"}


@pytest.fixture(autouse=True)
def _sqlite_turn_reporter(monkeypatch, tmp_path):
    # Same DATABASE_URL setting drives both stores; pin turn_reporter's
    # singleton to the same temp sqlite file the owner-store fixture uses.
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.turn_reporter as reporter_mod

    reporter_mod._turn_reporter = None
    yield
    reporter_mod._turn_reporter = None


async def test_authed_caller_upgrades_anon_thread_and_appears_in_sidebar(
    client, valid_cookie_for, seed_owner, mock_agent, mock_registry
):
    """An authed caller resuming an anon-owned widget thread flips ownership
    and backfills turn_reports, so the thread shows up in their sidebar via
    list_threads_for_user."""
    from src.thread_owners import get_thread_owner_store
    from src.turn_reporter import get_turn_reporter

    seed_owner("t-upgrade", None)  # anon-owned

    reporter = get_turn_reporter()
    reporter.log_turn_report(
        final_state={"final_answer": "hi", "tools_used": []},
        session_id="t-upgrade",
        turn_index=1,
        question_id="q-anon-1",
        query_text="anon turn before login",
        duration_ms=10.0,
        acting_user=None,
        success=True,
        capabilities=[],
    )

    r = await client.post(
        "/api/v1/query",
        headers=valid_cookie_for("me@access-ci.org"),
        json={"query": "hi again", "session_id": "t-upgrade"},
    )

    assert r.status_code == 200
    mock_agent.assert_called_once()

    owner = get_thread_owner_store().resolve_owner("t-upgrade")
    assert owner is not None
    assert owner.was_authenticated is True

    import hashlib

    expected_hash = hashlib.sha256(b"me@access-ci.org").hexdigest()[:16]
    assert owner.user_hash == expected_hash

    threads = reporter.list_threads_for_user(expected_hash, limit=10)
    assert any(t["session_id"] == "t-upgrade" for t in threads)
