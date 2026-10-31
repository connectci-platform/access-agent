"""Tests for POST /threads/{id}/runs/{run_id}/cancel (explicit protocol cancel).

Agent tokens come from the shared ``agent_tokens`` fixture; the sqlite
thread-owner store and run-registry reset mirror tests/test_thread_routes_run.py.
"""

import asyncio

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.thread_owners import get_thread_owner_store

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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
def bearer_for(agent_tokens):
    return agent_tokens.headers


@pytest.fixture
def bearer(bearer_for):
    return bearer_for("runner@x")


@pytest.fixture
def seed_owner():
    def _seed(thread_id: str, user: str) -> None:
        get_thread_owner_store().claim_thread(thread_id, user)

    return _seed


# ---------------------------------------------------------------------------
# POST /threads/{thread_id}/runs/{run_id}/cancel
# ---------------------------------------------------------------------------


async def test_cancel_anon_caller_unknown_thread_404(client):
    """Anon (no token) caller on a thread with no owner row: 404, not 401."""
    r = await client.post("/api/v1/threads/t/runs/r/cancel")
    assert r.status_code == 404


async def test_cancel_anon_caller_on_authed_owned_thread_404(client, seed_owner):
    """Anon (no token) caller hitting an existing AUTHED-owned thread's
    cancel is a non-owner: 404, not 401/403."""
    seed_owner("t-owned-authed", "alice@x")
    r = await client.post("/api/v1/threads/t-owned-authed/runs/whatever/cancel")
    assert r.status_code == 404


async def test_cancel_non_owner_404(client, bearer_for, seed_owner):
    seed_owner("t-owned", "alice@x")
    r = await client.post(
        "/api/v1/threads/t-owned/runs/whatever/cancel",
        headers=bearer_for("bob@x"),
    )
    assert r.status_code == 404  # not bob's thread → indistinguishable from unknown


async def test_cancel_unknown_run_404(client, bearer_for, seed_owner):
    seed_owner("t-mine", "me@x")
    r = await client.post(
        "/api/v1/threads/t-mine/runs/no-such-run/cancel",
        headers=bearer_for("me@x"),
    )
    assert r.status_code == 404  # owner, but no in-flight run with that id


async def test_cancel_owner_cancels_in_flight_run(client, bearer_for, seed_owner):
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
        headers=bearer_for("me@x"),
    )
    assert r.status_code == 200
    assert r.json() == {"status": "cancelled"}

    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled.is_set()
    assert "t-live" not in tr._active_by_thread  # held_run's finally released the lock


async def test_cancel_cannot_cross_threads(client, bearer_for, seed_owner):
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
        headers=bearer_for("attacker@x"),
    )
    assert r.status_code == 404  # run-b is not active on thread-a
    assert not task.done()  # B's run is untouched

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
