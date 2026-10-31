"""Tests for POST /threads/{id}/handoff and POST /handoff/exchange.

Agent tokens come from the shared ``agent_tokens`` fixture, and the
thread-owner store is sqlite-backed — do not invent a new auth-mocking
approach. The handoff token store itself is in-memory by
default (no REDIS_URL configured in tests), reset between tests via
src.api.handoff_tokens._reset_store_for_test().
"""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.handoff_tokens import _reset_store_for_test
from src.main import app
from src.thread_owners import get_thread_owner_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def bearer_for(agent_tokens):
    return agent_tokens.headers


# ---------------------------------------------------------------------------
# POST /threads/{id}/handoff (mint)
# ---------------------------------------------------------------------------


async def test_mint_requires_token(client):
    r = await client.post("/api/v1/threads/some-thread/handoff")
    assert r.status_code == 401


async def test_mint_non_owner_gets_404(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    r = await client.post(
        "/api/v1/threads/t-owned/handoff",
        headers=bearer_for("userB@x"),
    )
    assert r.status_code == 404


async def test_mint_anon_owned_thread_gets_404(client, bearer_for):
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
        headers=bearer_for("userA@x"),
    )
    assert r.status_code == 404


async def test_mint_owner_gets_token(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    r = await client.post(
        "/api/v1/threads/t-owned/handoff",
        headers=bearer_for("userA@x"),
    )
    assert r.status_code == 200
    body = r.json()
    assert body["expires_in"] == 60
    assert isinstance(body["token"], str) and body["token"]


# ---------------------------------------------------------------------------
# POST /handoff/exchange
# ---------------------------------------------------------------------------


async def test_exchange_requires_token(client):
    r = await client.post("/api/v1/handoff/exchange", json={"token": "whatever"})
    assert r.status_code == 401


async def test_exchange_same_identity_returns_thread_id(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    mint = await client.post(
        "/api/v1/threads/t-owned/handoff",
        headers=bearer_for("userA@x"),
    )
    token = mint.json()["token"]

    r = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        headers=bearer_for("userA@x"),
    )
    assert r.status_code == 200
    assert r.json() == {"thread_id": "t-owned"}


async def test_exchange_different_identity_gets_404(client, bearer_for):
    """Identity binding: a leaked token exchanged by a different caller must
    fail even though it is otherwise valid and unexpired."""
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    mint = await client.post(
        "/api/v1/threads/t-owned/handoff",
        headers=bearer_for("userA@x"),
    )
    token = mint.json()["token"]

    r = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        headers=bearer_for("userB@x"),
    )
    assert r.status_code == 404


async def test_exchange_is_single_use(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    mint = await client.post(
        "/api/v1/threads/t-owned/handoff",
        headers=bearer_for("userA@x"),
    )
    token = mint.json()["token"]

    first = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        headers=bearer_for("userA@x"),
    )
    second = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        headers=bearer_for("userA@x"),
    )
    assert first.status_code == 200
    assert second.status_code == 404


async def test_exchange_response_contains_only_thread_id(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    mint = await client.post(
        "/api/v1/threads/t-owned/handoff",
        headers=bearer_for("userA@x"),
    )
    token = mint.json()["token"]

    r = await client.post(
        "/api/v1/handoff/exchange",
        json={"token": token},
        headers=bearer_for("userA@x"),
    )
    body = r.json()
    assert set(body.keys()) == {"thread_id"}
    assert "messages" not in body
    assert "values" not in body
