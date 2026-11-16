"""DELETE /api/v1/threads/{id} and DELETE /api/v1/me/conversations (D8-3126)."""

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from httpx import ASGITransport, AsyncClient

from src.api import delete_routes, thread_runs
from src.conversation_delete import DeletionUnavailableError
from src.main import app


@pytest.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        app.state.checkpointer = None
        yield c


@pytest.fixture
def bearer(agent_tokens):
    return agent_tokens.headers("runner@x")


@pytest.fixture
def deleted(monkeypatch):
    calls: list[dict] = []

    async def fake(thread_ids, *, user, scope, checkpointer):
        calls.append({"ids": list(thread_ids), "user": user, "scope": scope})
        return len(thread_ids)

    monkeypatch.setattr(delete_routes, "delete_conversations", fake)
    return calls


@pytest.fixture
def access(monkeypatch):
    allowed: set[str] = set()
    seen: list[tuple] = []

    def check(thread_id, caller, claim_key, **_):
        seen.append((thread_id, caller, claim_key))
        if thread_id not in allowed:
            raise HTTPException(status_code=404, detail="Thread not found")

    monkeypatch.setattr(delete_routes, "require_thread_access", check)
    return allowed, seen


# ── One conversation ─────────────────────────────────────────────────────


async def test_the_owner_deletes_one_conversation(client, bearer, deleted, access):
    allowed, seen = access
    allowed.add("t1")

    r = await client.delete("/api/v1/threads/t1", headers=bearer)

    assert r.status_code == 204
    assert seen == [("t1", "runner@x", None)]
    assert deleted == [{"ids": ["t1"], "user": "runner@x", "scope": "one"}]


async def test_a_signed_out_key_holder_deletes_their_conversation(client, deleted, access):
    allowed, seen = access
    allowed.add("t1")

    r = await client.delete("/api/v1/threads/t1", headers={"X-Thread-Claim-Key": "k" * 32})

    assert r.status_code == 204
    assert seen[0][1] is None and seen[0][2] == "k" * 32
    assert deleted == [{"ids": ["t1"], "user": None, "scope": "one"}]


async def test_someone_elses_conversation_is_404_and_untouched(client, bearer, deleted, access):
    r = await client.delete("/api/v1/threads/not-mine", headers=bearer)

    assert r.status_code == 404
    assert deleted == []


async def test_a_conversation_still_answering_is_409(client, bearer, deleted, access):
    allowed, _ = access
    allowed.add("t1")
    thread_runs.acquire_thread_run("t1", "run-1")
    try:
        r = await client.delete("/api/v1/threads/t1", headers=bearer)
    finally:
        thread_runs.release_thread_run("t1", "run-1")

    assert r.status_code == 409
    assert "still being written" in r.json()["detail"]
    assert deleted == []


async def test_the_run_lock_is_released_after_deleting(client, bearer, deleted, access):
    allowed, _ = access
    allowed.add("t1")
    await client.delete("/api/v1/threads/t1", headers=bearer)

    # A new run on the same id isn't blocked by the delete's lock.
    thread_runs.acquire_thread_run("t1", "run-after")
    thread_runs.release_thread_run("t1", "run-after")


async def test_no_database_is_503(client, bearer, monkeypatch, access):
    allowed, _ = access
    allowed.add("t1")
    monkeypatch.setattr(
        delete_routes,
        "delete_conversations",
        AsyncMock(side_effect=DeletionUnavailableError("no db")),
    )

    r = await client.delete("/api/v1/threads/t1", headers=bearer)

    assert r.status_code == 503
    assert "can't be deleted right now" in r.json()["error"]


# ── All conversations ────────────────────────────────────────────────────


async def test_delete_all_removes_every_conversation_the_caller_owns(
    client, bearer, deleted, monkeypatch
):
    seen: list[str] = []

    def owned(user):
        seen.append(user)
        return ["t1", "t2"]

    monkeypatch.setattr(delete_routes, "owned_thread_ids", owned)

    r = await client.delete("/api/v1/me/conversations", headers=bearer)

    assert r.status_code == 200
    assert r.json() == {"deleted": 2}
    assert r.headers["cache-control"] == "no-store"
    assert seen == ["runner@x"]
    assert deleted == [{"ids": ["t1", "t2"], "user": "runner@x", "scope": "all"}]


async def test_delete_all_needs_sign_in(client, deleted, monkeypatch):
    owned = AsyncMock()
    monkeypatch.setattr(delete_routes, "owned_thread_ids", owned)

    r = await client.delete("/api/v1/me/conversations")

    assert r.status_code == 401
    owned.assert_not_called()
    assert deleted == []


async def test_delete_all_waits_for_any_answer_still_streaming(
    client, bearer, deleted, monkeypatch
):
    monkeypatch.setattr(delete_routes, "owned_thread_ids", lambda user: ["t1", "t2"])
    thread_runs.acquire_thread_run("t2", "run-2")
    try:
        r = await client.delete("/api/v1/me/conversations", headers=bearer)
    finally:
        thread_runs.release_thread_run("t2", "run-2")

    assert r.status_code == 409
    assert deleted == []
    # t1's lock, taken before t2 refused, was let go.
    thread_runs.acquire_thread_run("t1", "run-after")
    thread_runs.release_thread_run("t1", "run-after")


async def test_delete_all_with_nothing_owned_deletes_nothing(client, bearer, deleted, monkeypatch):
    monkeypatch.setattr(delete_routes, "owned_thread_ids", lambda user: [])

    r = await client.delete("/api/v1/me/conversations", headers=bearer)

    assert r.json() == {"deleted": 0}


async def test_delete_all_without_a_database_is_503(client, bearer, monkeypatch):
    def unavailable(user):
        raise DeletionUnavailableError("no db")

    monkeypatch.setattr(delete_routes, "owned_thread_ids", unavailable)

    r = await client.delete("/api/v1/me/conversations", headers=bearer)

    assert r.status_code == 503
