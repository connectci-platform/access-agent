"""Tests for POST /threads and POST /threads/{id}/history.

Agent tokens come from the shared ``agent_tokens`` fixture in conftest.py —
do not invent a new signing harness. Ownership is seeded directly
via get_thread_owner_store().claim_thread(...) (Task 2's store), and the
checkpointer is a fake set directly on app.state.checkpointer (mirrors
tests/test_pooled_checkpointer.py's FakeCheckpointerCM pattern) so these
tests need no live Postgres.
"""

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import HumanMessage

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


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


@pytest.fixture
def bearer_for(agent_tokens):
    return agent_tokens.headers


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


async def test_create_thread_anon_accepts(client):
    r = await client.post("/api/v1/threads")
    assert r.status_code == 200


async def test_create_thread_returns_fresh_thread(client, bearer_for):
    r = await client.post("/api/v1/threads", headers=bearer_for("userA@x"))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "idle"
    assert body["metadata"] == {}
    assert body["values"] == {}
    assert "thread_id" in body
    assert "created_at" in body


async def test_create_thread_returns_distinct_ids(client, bearer_for):
    headers = bearer_for("userA@x")
    r1 = await client.post("/api/v1/threads", headers=headers)
    r2 = await client.post("/api/v1/threads", headers=headers)
    assert r1.json()["thread_id"] != r2.json()["thread_id"]


# ---------------------------------------------------------------------------
# POST /threads/{id}/history
# ---------------------------------------------------------------------------


async def test_history_anon_caller_on_authed_owned_thread_404(client):
    """An anon (no-token) caller on an AUTHED-owned thread's history is a
    non-owner: 404 (never 401, never 403, never requires_auth)."""
    get_thread_owner_store().claim_thread("t-owned-authed", "userA@x")

    r = await client.post(
        "/api/v1/threads/t-owned-authed/history",
        json={"limit": 5},
    )
    assert r.status_code == 404


async def test_history_anon_owner_gets_200(client):
    """An anon caller (no token) who owns an anon thread (same session_id)
    can read their own history — 200, not 404/401."""
    get_thread_owner_store().claim_thread("t-anon-owned", None)
    app.state.checkpointer = None

    r = await client.post(
        "/api/v1/threads/t-anon-owned/history",
        json={"limit": 5},
    )
    assert r.status_code == 200
    assert r.json() == []


async def test_history_non_owner_gets_404(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    r = await client.post(
        "/api/v1/threads/t-owned/history",
        json={"limit": 5},
        headers=bearer_for("userB@x"),
    )
    assert r.status_code == 404


async def test_history_unknown_thread_gets_404_identical_body(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")

    non_owner = await client.post(
        "/api/v1/threads/t-owned/history",
        json={"limit": 5},
        headers=bearer_for("userB@x"),
    )
    unknown = await client.post(
        "/api/v1/threads/does-not-exist/history",
        json={"limit": 5},
        headers=bearer_for("userB@x"),
    )
    assert non_owner.status_code == unknown.status_code == 404
    assert non_owner.json() == unknown.json()


async def test_history_owner_gets_state_history(client, bearer_for, monkeypatch):
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
            headers=bearer_for("userA@x"),
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


async def test_history_no_checkpointer_returns_empty_list(client, bearer_for):
    get_thread_owner_store().claim_thread("t-owned", "userA@x")
    app.state.checkpointer = None

    r = await client.post(
        "/api/v1/threads/t-owned/history",
        json={"limit": 10},
        headers=bearer_for("userA@x"),
    )
    assert r.status_code == 200
    assert r.json() == []
