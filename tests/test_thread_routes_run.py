"""Tests for POST /threads/{id}/runs/stream (the gated, bounded streaming run)
and the widget-path ownership/lock sharing.

Agent tokens come from the shared ``agent_tokens`` fixture. stream_agent is
mocked so no LLM/MCP is needed. The thread-owner store is pointed at a temp
sqlite file.
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessageChunk

from src.api import thread_routes
from src.api.identity import CLAIM_KEY_HEADER
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
def _sqlite_turn_reporter(monkeypatch, tmp_path):
    """Point get_turn_reporter()'s module-level singleton at the SAME temp
    sqlite file _sqlite_owner_store uses (load-bearing: get_turn_reporter()
    caches its engine on first touch, and this is the first run-test suite to
    touch it — without a per-test reset, a stale engine from another test, or
    "" from CI env, leaks in and the "exactly one row" assertions go
    order-dependent/flaky)."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.turn_reporter as reporter_mod

    reporter_mod._turn_reporter = None
    yield
    reporter_mod._turn_reporter = None


@pytest.fixture(autouse=True)
def _reset_run_registry():
    """Clear the in-process per-thread run registry between tests."""
    import src.api.thread_runs as tr

    tr._active_by_thread.clear()
    tr._task_by_run.clear()
    yield
    tr._active_by_thread.clear()
    tr._task_by_run.clear()


@pytest.fixture(autouse=True)
def _disable_turnstile(monkeypatch):
    """Isolate from local .env TURNSTILE_SECRET_KEY leakage (mirrors
    test_auth_e2e.py's fixture of the same name). Tests that need Turnstile
    enabled override this explicitly via monkeypatch inside the test body."""
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)


@pytest.fixture(autouse=True)
def _reset_turnstile_guard():
    """The module-level TurnstileGuard singleton leaks session state across
    tests (same hazard as the turn_reporter singleton elsewhere in this
    suite) — reset it so one test's free-query count or verified flag can't
    bleed into another's assertions."""
    import src.turnstile as ts

    ts._guard = None
    yield
    ts._guard = None


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


CLAIM_KEY = "k" * 43


def _mock_stream_agent(monkeypatch, *, content: str = "hi", msg_id: str = "m1"):
    """Patch thread_routes.stream_agent to yield a canned messages chunk + updates."""

    async def _fake(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content=content, id=msg_id),
                {"langgraph_node": "model"},
            ),
        )
        yield "updates", {"tool_calling_loop": {"final_answer": content}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)


# ---------------------------------------------------------------------------
# POST /threads/{id}/runs/stream
# ---------------------------------------------------------------------------


async def test_run_stream_emits_protocol_events(client, bearer, monkeypatch):
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-new/runs/stream",
        headers=bearer,
        json={
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 200
    body = r.text
    assert "event: metadata" in body
    assert "event: messages\n" in body
    assert "event: messages/partial" not in body
    assert "event: messages/complete" not in body
    assert "event: values" in body
    assert "event: end" not in body  # protocol has no end terminator


async def test_run_anon_caller_on_new_thread_200(client, monkeypatch):
    """Anon caller (no token), new thread → 200, not 401. claim_thread lazily
    creates an anon-owned thread for a None caller."""
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-anon-new/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200


async def test_run_anon_caller_on_authed_owned_thread_404(client, bearer_for, seed_owner):
    """Anon (no token) caller hitting an existing AUTHED-owned thread is a
    non-owner: 404, not 401/403."""
    seed_owner("t-owned-authed", "alice@x")
    r = await client.post(
        "/api/v1/threads/t-owned-authed/runs/stream",
        json={
            "input": {"messages": [{"role": "user", "content": "x"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 404


async def test_run_authed_caller_still_requires_nothing_new(client, bearer, monkeypatch):
    """Regression: an authed caller on a brand-new thread still gets 200."""
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-authed-new/runs/stream",
        headers=bearer,
        json={
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 200


async def test_run_stream_releases_lock_if_generator_never_starts(client, bearer, monkeypatch):
    """If the streaming generator is never started, the per-thread lock must not
    leak (Finding I4).

    The lock is acquired synchronously in the handler and released in the
    generator's finally, but an async generator that is never iterated never runs
    its finally. Force the handler's synchronous post-acquire path to raise
    (before StreamingResponse is handed back) and assert the thread is not left
    409-wedged: a subsequent run on the same thread must succeed, not 409.
    """
    import src.api.thread_runs as tr

    with (
        patch("src.api.thread_routes.StreamingResponse", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await client.post(
            "/api/v1/threads/t-wedge/runs/stream",
            headers=bearer,
            json={"input": {"messages": [{"role": "user", "content": "x"}]}},
        )

    assert "t-wedge" not in tr._active_by_thread  # lock released on the failure path

    # And a real run on the same thread now succeeds (not permanently wedged).
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-wedge/runs/stream",
        headers=bearer,
        json={"input": {"messages": [{"role": "user", "content": "x"}]}},
    )
    assert r.status_code == 200


async def test_run_on_another_users_thread_404(client, bearer_for, seed_owner, monkeypatch):
    _mock_stream_agent(monkeypatch)
    seed_owner("t-owned", "alice@x")
    r = await client.post(
        "/api/v1/threads/t-owned/runs/stream",
        headers=bearer_for("bob@x"),
        json={
            "input": {"messages": [{"role": "user", "content": "x"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 404


async def test_resource_context_from_config_forwarded(client, bearer, monkeypatch):
    captured = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        if False:
            yield  # make it an async generator

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)
    await client.post(
        "/api/v1/threads/t/runs/stream",
        headers=bearer,
        json={
            "input": {"messages": [{"role": "user", "content": "x"}]},
            "if_not_exists": "create",
            "config": {"configurable": {"resource_context": "delta"}},
        },
    )
    assert captured.get("resource_context") == "delta"


async def test_concurrent_run_rejected(client, bearer, monkeypatch):
    release = asyncio.Event()

    async def _blocking_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "model"},
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
        return await client.post("/api/v1/threads/t-busy/runs/stream", headers=bearer, json=body)

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


async def test_widget_and_fullscreen_share_lock(client, bearer, monkeypatch):
    """A widget /query and a /runs/stream on the same session_id can't run at once."""
    release = asyncio.Event()

    async def _blocking_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "model"},
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
            headers=bearer,
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
        headers=bearer,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert fullscreen.status_code == 409

    release.set()
    await widget_task


async def test_widget_thread_owned_by_creator(client, bearer_for, monkeypatch):
    """A widget turn claims ownership; a stranger's runs/stream on it → 404."""

    async def _fake_stream(**_kwargs):
        yield (
            "messages",
            (
                AIMessageChunk(content="hi", id="m1"),
                {"langgraph_node": "model"},
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
        headers=bearer_for("alice@x"),
        json={"query": "hi", "session_id": "widget-sess"},
    )
    assert widget.status_code == 200
    _ = widget.text  # drain the stream so the generator's finally releases the lock

    assert get_thread_owner_store().owned_by("widget-sess", "alice@x") is True

    # Bob can't run Alice's widget-born thread.
    bob = await client.post(
        "/api/v1/threads/widget-sess/runs/stream",
        headers=bearer_for("bob@x"),
        json={"input": {"messages": [{"role": "user", "content": "x"}]}},
    )
    assert bob.status_code == 404


async def test_widget_registry_failure_does_not_leak_lock(client, bearer, monkeypatch):
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
        headers=bearer,
        json={"query": "hi", "session_id": "leak-sess"},
    )
    assert r.status_code == 200
    _ = r.text  # drain the stream; the error path runs and held_run's finally fires

    import src.api.thread_runs as tr

    assert "leak-sess" not in tr._active_by_thread  # lock released, not leaked
    # A subsequent acquire on the same session must NOT 409.
    tr.acquire_thread_run("leak-sess", "next-run")  # raises HTTPException(409) if leaked


# ---------------------------------------------------------------------------
# thread_run_stream error branches: cancel / timeout / generic exception,
# and final_messages built from a real "updates" chunk.
# ---------------------------------------------------------------------------


async def test_run_stream_builds_final_messages_from_updates_chunk(client, bearer, monkeypatch):
    """The 'updates' branch pulls node_output['messages'] into final_messages,
    which the 'values' event carries even when no messages event ever fired."""

    async def _fake(**_kwargs):
        yield "updates", {"some_node": {"messages": [{"type": "human", "content": "hi"}]}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)

    r = await client.post(
        "/api/v1/threads/t-updates/runs/stream",
        headers=bearer,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200
    assert 'event: values\ndata: {"messages": [{"type": "human", "content": "hi"}]}' in r.text


async def test_run_stream_cancelled_mid_turn_emits_error_and_propagates(bearer_for, seed_owner):
    """A real cancel (client disconnect / cancel_run) while streaming emits
    event: error error:cancelled, then propagates so the task truly cancels.

    Drives thread_run_stream directly (not via httpx) and cancels the task
    consuming its body_iterator — ASGITransport doesn't cleanly support
    cancelling the serving task out from under a client.post() call, but the
    endpoint itself is an ordinary async function returning a
    StreamingResponse, so this exercises the real code path without that
    transport limitation.
    """
    from types import SimpleNamespace

    from src.agent_token import AgentIdentity
    from src.api.thread_routes import RunRequest, thread_run_stream

    seed_owner("t-cancel-mid", "canceller@x")

    import src.api.thread_runs as tr

    tr._active_by_thread.clear()
    tr._task_by_run.clear()

    started = asyncio.Event()
    saw_cancelled = asyncio.Event()

    async def _long_stream(**_kwargs):
        started.set()
        try:
            await asyncio.sleep(10)
            yield "updates", {}
        except asyncio.CancelledError:
            saw_cancelled.set()
            raise

    identity = AgentIdentity(
        sub="canceller@x", scopes=frozenset({"agent:read"}), exp=int(time.time()) + 900
    )
    raw_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(checkpointer=None)))
    body = RunRequest.model_validate({"input": {"messages": [{"role": "user", "content": "hi"}]}})

    collected: list[str] = []

    async def _consume() -> None:
        with patch("src.api.thread_routes.stream_agent", _long_stream):
            response = await thread_run_stream("t-cancel-mid", body, raw_request, identity)
            async for chunk in response.body_iterator:
                collected.append(chunk)

    task = asyncio.ensure_future(_consume())
    await started.wait()
    for _ in range(50):
        await asyncio.sleep(0.01)
        if "t-cancel-mid" in tr._active_by_thread:
            break

    run_id = tr._active_by_thread["t-cancel-mid"]
    assert tr.cancel_run("t-cancel-mid", run_id) is True

    with pytest.raises(asyncio.CancelledError):
        await task

    assert saw_cancelled.is_set()
    assert any('event: error\ndata: {"error": "cancelled"}' in c for c in collected)
    assert "t-cancel-mid" not in tr._active_by_thread


async def test_run_stream_timeout_emits_error_and_returns(client, bearer, monkeypatch):
    """asyncio.timeout expiry (TimeoutError, not CancelledError in 3.11) emits
    event: error error:timeout and returns without re-raising."""
    from src.config import settings

    monkeypatch.setattr(settings, "AGENT_TURN_TIMEOUT_S", 0, raising=False)

    async def _slow_stream(**_kwargs):
        await asyncio.sleep(0.05)
        yield "updates", {}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _slow_stream)

    r = await client.post(
        "/api/v1/threads/t-timeout/runs/stream",
        headers=bearer,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200
    assert 'event: error\ndata: {"error": "timeout"}' in r.text


async def test_run_stream_generic_exception_emits_agent_error(client, bearer, monkeypatch):
    """An unexpected exception from stream_agent is caught, logged, and turned
    into event: error error:agent_error rather than propagating as a 500."""

    async def _boom(**_kwargs):
        raise RuntimeError("unexpected agent failure")
        yield  # pragma: no cover - unreachable, makes this an async generator

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _boom)

    r = await client.post(
        "/api/v1/threads/t-boom/runs/stream",
        headers=bearer,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200
    assert 'event: error\ndata: {"error": "agent_error"}' in r.text


# ---------------------------------------------------------------------------
# Resolve-first ownership gate: anon->authed upgrade (replaces claim-then-check)
# ---------------------------------------------------------------------------


async def test_run_anon_thread_resumed_by_authed_caller_upgrades_ownership(
    client, bearer_for, seed_owner, monkeypatch
):
    """An anon thread resumed by a signed-in caller with its key flips ownership:
    the anon owner later signs in and resolve_owner reports was_authenticated=True.
    Mirrors /query's upgrade test in tests/test_query_ownership.py, minus the
    sidebar assertion (turn_reports writes are Task 3a's job)."""
    _mock_stream_agent(monkeypatch)
    get_thread_owner_store().claim_thread("t-upgrade", None, CLAIM_KEY)

    r = await client.post(
        "/api/v1/threads/t-upgrade/runs/stream",
        headers={**bearer_for("me@access-ci.org"), "X-Thread-Claim-Key": CLAIM_KEY},
        json={"input": {"messages": [{"role": "user", "content": "hi again"}]}},
    )
    assert r.status_code == 200

    owner = get_thread_owner_store().resolve_owner("t-upgrade")
    assert owner is not None
    assert owner.was_authenticated is True

    import hashlib

    expected_hash = hashlib.sha256(b"me@access-ci.org").hexdigest()[:16]
    assert owner.user_hash == expected_hash


async def test_run_anon_owned_thread_resumed_by_anon_caller_stays_anon(
    client, seed_owner, monkeypatch
):
    """Existing anon thread + its key holder resuming (no token) -> allowed,
    ownership stays anon-owned, and the final unconditional claim_thread is a
    harmless no-op."""
    _mock_stream_agent(monkeypatch)
    get_thread_owner_store().claim_thread("t-anon-resume", None, CLAIM_KEY)

    r = await client.post(
        "/api/v1/threads/t-anon-resume/runs/stream",
        headers={"X-Thread-Claim-Key": CLAIM_KEY},
        json={"input": {"messages": [{"role": "user", "content": "hi again"}]}},
    )
    assert r.status_code == 200

    owner = get_thread_owner_store().resolve_owner("t-anon-resume")
    assert owner is not None
    assert owner.was_authenticated is False
    assert owner.user_hash is None


async def test_run_authed_owned_thread_wrong_caller_404(
    client, bearer_for, seed_owner, monkeypatch
):
    """Authed-owned thread + wrong authed caller -> 404 (never 403), via the
    single access check."""
    _mock_stream_agent(monkeypatch)
    seed_owner("t-owned-wrong-caller", "alice@x")

    r = await client.post(
        "/api/v1/threads/t-owned-wrong-caller/runs/stream",
        headers=bearer_for("bob@x"),
        json={"input": {"messages": [{"role": "user", "content": "x"}]}},
    )
    assert r.status_code == 404


async def test_run_authed_owned_path_checks_access_exactly_once(
    client, bearer_for, seed_owner, monkeypatch
):
    """Guard against the double-gate regression: the run path decides access
    exactly once, in require_thread_access."""
    _mock_stream_agent(monkeypatch)
    seed_owner("t-single-check", "alice@x")

    with patch(
        "src.thread_owners.ThreadOwnerStore.access",
        wraps=get_thread_owner_store().access,
    ) as spy:
        r = await client.post(
            "/api/v1/threads/t-single-check/runs/stream",
            headers=bearer_for("alice@x"),
            json={"input": {"messages": [{"role": "user", "content": "x"}]}},
        )
        assert r.status_code == 200
        assert spy.call_count == 1


async def test_run_losing_concurrent_upgrade_caller_does_not_backfill(
    client, bearer_for, seed_owner, monkeypatch
):
    """Two authed callers holding the key race their first post-login turn on
    the same anon thread. The loser's resolve_owner read is stale (pre-dates the
    winner's flip); the loser's own claim_with_key call must lose the race
    (returns False) and therefore must NOT call backfill_user_hash — mirrors
    tests/test_query_ownership.py::test_losing_race_caller_does_not_backfill_winners_thread,
    adapted: this path doesn't write turn_reports rows (Task 3a), so the
    property is verified by spying on backfill_user_hash rather than
    inspecting a historical row.
    """
    from src.thread_owners import ThreadOwner

    _mock_stream_agent(monkeypatch)
    owner_store = get_thread_owner_store()
    owner_store.claim_thread("t-race", None, CLAIM_KEY)
    # Alice's flip has already won the race and committed.
    assert owner_store.claim_with_key("t-race", "alice@access-ci.org", CLAIM_KEY) is True

    # Bob's request resolved the owner as anon-owned (a stale read taken
    # before alice's flip committed) — fake only the read; the real
    # claim_with_key call below still hits the live (already-flipped) row.
    stale_anon_owner = ThreadOwner(False, None)
    with (
        patch.object(owner_store, "resolve_owner", return_value=stale_anon_owner),
        patch("src.turn_reporter.get_turn_reporter") as mock_get_reporter,
    ):
        mock_reporter = mock_get_reporter.return_value
        r = await client.post(
            "/api/v1/threads/t-race/runs/stream",
            headers={**bearer_for("bob@access-ci.org"), "X-Thread-Claim-Key": CLAIM_KEY},
            json={"input": {"messages": [{"role": "user", "content": "hi again"}]}},
        )
        assert r.status_code == 404  # the thread is alice's now
        mock_reporter.backfill_user_hash.assert_not_called()

    # Ownership must remain alice's — bob's flip lost the race.
    owner = owner_store.resolve_owner("t-race")
    assert owner is not None
    assert owner.was_authenticated is True
    import hashlib

    assert owner.user_hash == hashlib.sha256(b"alice@access-ci.org").hexdigest()[:16]


# ---------------------------------------------------------------------------
# Task 3a: turn_reports row written from the thread/run path
# ---------------------------------------------------------------------------


async def test_run_stream_writes_one_turn_report_row(client, monkeypatch):
    """An anon thread/run turn writes exactly one turn_reports row: session_id
    is the thread_id, user_hash is NULL (anon caller), query_text is the
    user's text. This is the real proof the reporter block ran (not just "no
    exception") — a forgotten get_turn_capture import would NameError, get
    swallowed by the block's `except Exception`, and silently write zero
    rows."""
    from src.turn_reporter import get_turn_reporter

    _mock_stream_agent(monkeypatch, content="the answer")

    r = await client.post(
        "/api/v1/threads/t-report-row/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "what is ACCESS?"}]}},
    )
    assert r.status_code == 200
    _ = r.text  # drain the stream so the post-stream reporter block runs

    reporter = get_turn_reporter()
    assert reporter.count_turns_for_session("t-report-row") == 1

    session = reporter._session_factory()
    try:
        from src.turn_reporter import TurnReport

        row = session.query(TurnReport).filter_by(session_id="t-report-row").one()
    finally:
        session.close()

    assert row.session_id == "t-report-row"
    assert row.user_hash is None
    assert row.query_text == "what is ACCESS?"
    assert row.success is True


async def test_run_stream_failed_turn_writes_success_false_row(client, monkeypatch):
    """A thread/run turn whose stream_agent raises mid-stream still writes
    exactly one turn_reports row, with success=False — failed-turn parity
    with /query's except-block write (routes.py:500-527)."""
    from src.turn_reporter import get_turn_reporter

    async def _boom(**_kwargs):
        yield (
            "messages",
            (AIMessageChunk(content="partial", id="m1"), {"langgraph_node": "model"}),
        )
        raise RuntimeError("mid-stream failure")

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _boom)

    r = await client.post(
        "/api/v1/threads/t-report-fail/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "will this fail?"}]}},
    )
    assert r.status_code == 200
    _ = r.text  # drain the stream so the error-path reporter block runs

    reporter = get_turn_reporter()
    assert reporter.count_turns_for_session("t-report-fail") == 1

    session = reporter._session_factory()
    try:
        from src.turn_reporter import TurnReport

        row = session.query(TurnReport).filter_by(session_id="t-report-fail").one()
    finally:
        session.close()

    assert row.success is False
    assert row.query_text == "will this fail?"


async def test_run_stream_reporter_failure_does_not_break_the_stream(client, monkeypatch):
    """A reporter write that raises must be swallowed by _write_turn_report's
    except block — the turn-report pipeline is off the response path, so a DB
    error there must never surface to the client or break the SSE stream.
    Mirrors tests/test_eval_battery_reports.py's
    test_reporter_failure_does_not_fail_the_run (log_turn_report.side_effect)."""
    _mock_stream_agent(monkeypatch, content="still works")

    broken_reporter = MagicMock()
    broken_reporter.count_turns_for_session.side_effect = RuntimeError("db down")
    monkeypatch.setattr(
        "src.turn_reporter.get_turn_reporter", lambda: broken_reporter, raising=False
    )

    r = await client.post(
        "/api/v1/threads/t-report-broken/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )

    assert r.status_code == 200
    body = r.text
    assert "event: messages\n" in body
    assert "event: values" in body
    assert "event: error" not in body  # reporter failure must not surface to the client
    broken_reporter.log_turn_report.assert_not_called()  # raised before this point


async def test_run_anon_caller_exceeding_free_queries_gets_turnstile_challenge(client, monkeypatch):
    """Turnstile enabled (immediate mode — challenge on the very first query,
    same as test_auth_e2e.py's test_body_acting_user_does_not_bypass_turnstile):
    an anon caller on thread_run_stream gets the requires_turnstile challenge
    response instead of a stream, and never acquires the run lock."""
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "test-secret", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate", raising=False)

    import src.api.thread_runs as tr

    r = await client.post(
        "/api/v1/threads/t-turnstile-anon/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )

    assert r.status_code == 200
    body = r.json()
    assert body.get("requires_turnstile") is True
    assert "site_key" in body
    # Never acquired the run lock — a challenged request must not wedge the thread.
    assert "t-turnstile-anon" not in tr._active_by_thread


async def test_run_authed_caller_not_challenged_with_turnstile_enabled(client, bearer, monkeypatch):
    """Authed callers skip Turnstile entirely, even with it enabled in
    immediate mode — mirrors /query's acting_user-truthy bypass."""
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "test-secret", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_MODE", "immediate", raising=False)

    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-turnstile-authed/runs/stream",
        headers=bearer,
        json={
            "input": {"messages": [{"role": "user", "content": "hi"}]},
            "if_not_exists": "create",
        },
    )
    assert r.status_code == 200
    assert "event: metadata" in r.text  # streamed normally, not challenged


async def test_run_anon_caller_not_challenged_when_turnstile_disabled(client, monkeypatch):
    """Regression guard: with Turnstile disabled (default, empty secret key —
    the _disable_turnstile autouse fixture's state), an anon caller streams
    normally. Proves this change is a no-op in the default configuration."""
    _mock_stream_agent(monkeypatch)
    r = await client.post(
        "/api/v1/threads/t-turnstile-off/runs/stream",
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200
    assert "event: metadata" in r.text


def thread_routes_runs():
    import src.api.thread_runs as tr

    return tr


def _deferred_turnstile(monkeypatch, free_queries: int = 3) -> None:
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "test-secret", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_SITE_KEY", "test-site-key", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_MODE", "deferred", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_FREE_QUERIES", free_queries, raising=False)


async def _ask(client, thread_id: str, *, headers=None, **body):
    # The client sends its claim key on every run, and an anonymous run on a
    # thread without one is refused.
    headers = {CLAIM_KEY_HEADER: CLAIM_KEY, **(headers or {})}
    return await client.post(
        f"/api/v1/threads/{thread_id}/runs/stream",
        headers=headers,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}, **body},
    )


def _challenged(response) -> bool:
    return response.headers["content-type"].startswith("application/json") and (
        response.json().get("requires_turnstile") is True
    )


async def test_run_anon_caller_is_challenged_after_the_free_questions(client, monkeypatch):
    """D8-2978: the count has to grow on this route, or deferred mode never fires."""
    _deferred_turnstile(monkeypatch)
    _mock_stream_agent(monkeypatch)

    for _ in range(3):
        r = await _ask(client, "t-free")
        assert "event: metadata" in r.text

    r = await _ask(client, "t-free")
    assert _challenged(r)
    assert r.json()["site_key"] == "test-site-key"


async def test_run_free_question_count_follows_the_setting(client, monkeypatch):
    _deferred_turnstile(monkeypatch, free_queries=1)
    _mock_stream_agent(monkeypatch)

    assert not _challenged(await _ask(client, "t-one-free"))
    assert _challenged(await _ask(client, "t-one-free"))


async def test_run_failed_turn_does_not_use_a_free_question(client, monkeypatch):
    _deferred_turnstile(monkeypatch, free_queries=1)

    async def _boom(**_kwargs):
        raise RuntimeError("model down")
        yield  # pragma: no cover

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _boom)
    r = await _ask(client, "t-failed")
    assert "event: error" in r.text

    _mock_stream_agent(monkeypatch)
    assert not _challenged(await _ask(client, "t-failed"))


async def test_run_count_is_shared_with_the_widget_session(client, monkeypatch):
    """A handed-off conversation keeps its count: session_id == thread_id."""
    from src.turnstile import get_turnstile_guard

    _deferred_turnstile(monkeypatch)
    _mock_stream_agent(monkeypatch)
    for _ in range(3):
        get_turnstile_guard().record_query("t-handed-off")

    assert _challenged(await _ask(client, "t-handed-off"))


async def test_run_signing_in_continues_a_thread_at_the_limit(client, bearer, monkeypatch):
    _deferred_turnstile(monkeypatch, free_queries=1)
    _mock_stream_agent(monkeypatch)
    await _ask(client, "t-sign-in")
    assert _challenged(await _ask(client, "t-sign-in"))

    r = await _ask(client, "t-sign-in", headers=bearer)

    assert "event: metadata" in r.text
    assert get_thread_owner_store().resolve_owner("t-sign-in").was_authenticated


async def test_run_signed_in_turns_are_not_counted(client, bearer, monkeypatch):
    from src.turnstile import get_turnstile_guard

    _deferred_turnstile(monkeypatch, free_queries=1)
    _mock_stream_agent(monkeypatch)

    for _ in range(3):
        assert not _challenged(await _ask(client, "t-authed-count", headers=bearer))

    assert "t-authed-count" not in get_turnstile_guard()._sessions


@pytest.mark.parametrize(
    "body",
    [{"turnstile_token": "tok"}, {"context": {"turnstile_token": "tok"}}],
    ids=["top-level", "context"],
)
async def test_run_passing_the_challenge_answers_the_question(client, monkeypatch, body):
    """The SDK's submit can only send the token in ``context``."""
    _deferred_turnstile(monkeypatch, free_queries=1)
    _mock_stream_agent(monkeypatch)
    verify = MagicMock(return_value=True)

    async def _verify(token):
        return verify(token)

    monkeypatch.setattr("src.turnstile.verify_turnstile_token", _verify)
    await _ask(client, "t-verify")
    assert _challenged(await _ask(client, "t-verify"))

    r = await _ask(client, "t-verify", **body)

    assert "event: metadata" in r.text
    verify.assert_called_once_with("tok")
    assert not _challenged(await _ask(client, "t-verify"))  # verified for the TTL


async def test_run_rejected_token_is_challenged_again(client, monkeypatch):
    _deferred_turnstile(monkeypatch, free_queries=1)
    _mock_stream_agent(monkeypatch)

    async def _reject(_token):
        return False

    monkeypatch.setattr("src.turnstile.verify_turnstile_token", _reject)
    await _ask(client, "t-bad-token")

    assert _challenged(await _ask(client, "t-bad-token", context={"turnstile_token": "bad"}))


async def test_run_count_failure_does_not_fail_the_turn(client, monkeypatch):
    _deferred_turnstile(monkeypatch)
    _mock_stream_agent(monkeypatch)
    monkeypatch.setattr(
        "src.turnstile.TurnstileGuard.record_query",
        MagicMock(side_effect=RuntimeError("count failed")),
    )

    r = await _ask(client, "t-count-fails")

    assert "event: values" in r.text
    assert "event: error" not in r.text


def _anon_count(thread_id: str) -> int:
    from src.turnstile import get_turnstile_guard

    return get_turnstile_guard()._sessions[thread_id].query_count


async def test_run_is_counted_before_the_answer_starts(client, monkeypatch):
    """Stop can't be a way to read an answer for free."""
    _deferred_turnstile(monkeypatch)
    seen: list[int] = []

    async def _fake(**_kwargs):
        seen.append(_anon_count("t-early"))
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)
    await _ask(client, "t-early")

    assert seen == [1]
    assert _anon_count("t-early") == 1  # and only once


async def test_run_cancelled_turn_uses_a_question(monkeypatch):
    from types import SimpleNamespace

    import src.api.thread_runs as tr
    from src.api.thread_routes import RunRequest, thread_run_stream

    _deferred_turnstile(monkeypatch)
    started = asyncio.Event()

    async def _long_stream(**_kwargs):
        started.set()
        await asyncio.sleep(10)
        yield "updates", {}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _long_stream)
    raw_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(checkpointer=None)))
    body = RunRequest.model_validate({"input": {"messages": [{"role": "user", "content": "hi"}]}})

    async def _consume() -> None:
        response = await thread_run_stream("t-stop", body, raw_request, None)
        async for _ in response.body_iterator:
            pass

    task = asyncio.ensure_future(_consume())
    await started.wait()
    assert tr.cancel_run("t-stop", tr._active_by_thread["t-stop"]) is True
    with pytest.raises(asyncio.CancelledError):
        await task

    assert _anon_count("t-stop") == 1


async def test_run_timed_out_turn_refunds_the_question(client, monkeypatch):
    _deferred_turnstile(monkeypatch)

    async def _times_out(**_kwargs):
        raise TimeoutError
        yield  # pragma: no cover

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _times_out)

    r = await _ask(client, "t-timeout")

    assert '"error": "timeout"' in r.text
    assert _anon_count("t-timeout") == 0


async def test_run_refund_is_in_place_before_the_error_event_is_sent(client, monkeypatch):
    """The client may leave as soon as it reads the error."""
    _deferred_turnstile(monkeypatch)

    async def _boom(**_kwargs):
        raise RuntimeError("model down")
        yield  # pragma: no cover

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _boom)
    real_format = thread_routes.format_sse_event
    seen: list[int] = []

    def _format(event, data):
        if event == "error":
            seen.append(_anon_count("t-refund-first"))
        return real_format(event, data)

    monkeypatch.setattr(thread_routes, "format_sse_event", _format)

    await _ask(client, "t-refund-first")

    assert seen == [0]


async def test_run_refused_as_busy_does_not_use_a_question(client, monkeypatch):
    import src.api.thread_runs as tr

    _deferred_turnstile(monkeypatch)
    _mock_stream_agent(monkeypatch)
    await _ask(client, "t-busy")
    tr.acquire_thread_run("t-busy", "other-run")

    r = await _ask(client, "t-busy")

    assert r.status_code == 409
    assert _anon_count("t-busy") == 1


async def test_run_concurrent_questions_cannot_exceed_the_limit(client, monkeypatch):
    """Only the run that holds the thread is counted, so a burst can't share
    one free question or spend extra ones."""
    _deferred_turnstile(monkeypatch, free_queries=1)
    release = asyncio.Event()

    async def _waits(**_kwargs):
        await release.wait()
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _waits)

    first = asyncio.ensure_future(_ask(client, "t-burst"))
    for _ in range(50):
        await asyncio.sleep(0.01)
        if "t-burst" in thread_routes_runs()._active_by_thread:
            break
    second = await _ask(client, "t-burst")
    release.set()
    await first

    assert _challenged(second)
    assert _anon_count("t-burst") == 1


async def test_run_failure_before_hand_off_refunds_the_question(client, monkeypatch):
    _deferred_turnstile(monkeypatch)

    with (
        patch("src.api.thread_routes.StreamingResponse", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError, match="boom"),
    ):
        await _ask(client, "t-no-hand-off")

    assert _anon_count("t-no-hand-off") == 0


def _sse(body: str) -> list[tuple[str, Any]]:
    out = []
    for block in body.strip().split("\n\n"):
        lines = block.split("\n")
        out.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return out


def _stream_of(monkeypatch, items):
    async def _fake(**_kwargs):
        for item in items:
            yield item

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)


async def _run_events(client, bearer, thread_id):
    r = await client.post(
        f"/api/v1/threads/{thread_id}/runs/stream",
        headers=bearer,
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}},
    )
    assert r.status_code == 200
    return _sse(r.text)


async def test_messages_event_is_a_message_and_metadata_pair(client, bearer, monkeypatch):
    _mock_stream_agent(monkeypatch, content="hello", msg_id="m9")
    events = await _run_events(client, bearer, "t-pair")
    (data,) = [d for n, d in events if n == "messages"]
    assert len(data) == 2
    assert data[0]["id"] == "m9"
    assert data[0]["content"] == "hello"
    assert data[1] == {"langgraph_node": "model"}


async def test_tool_call_chunks_are_not_sent(client, bearer, monkeypatch):
    chunk = AIMessageChunk(
        content="",
        id="m1",
        tool_call_chunks=[{"name": "search", "args": "{}", "id": "c1", "index": 0}],
    )
    _stream_of(monkeypatch, [("messages", (chunk, {"langgraph_node": "model"}))])
    events = await _run_events(client, bearer, "t-toolchunk")
    assert "messages" not in [n for n, _ in events]


async def test_sentinel_is_not_sent(client, bearer, monkeypatch):
    from src.llm import EMPTY_ANSWER_SENTINEL

    chunk = AIMessageChunk(content=EMPTY_ANSWER_SENTINEL, id="m1")
    _stream_of(monkeypatch, [("messages", (chunk, {"langgraph_node": "model"}))])
    events = await _run_events(client, bearer, "t-sentinel")
    assert "messages" not in [n for n, _ in events]


async def test_values_fallback_has_one_message_per_id_in_order(client, bearer, monkeypatch):
    meta = {"langgraph_node": "model"}
    _stream_of(
        monkeypatch,
        [
            ("messages", (AIMessageChunk(content="Let me ", id="a"), meta)),
            ("messages", (AIMessageChunk(content="look.", id="a"), meta)),
            ("messages", (AIMessageChunk(content="The answer.", id="b"), meta)),
        ],
    )
    events = await _run_events(client, bearer, "t-fallback")
    (values,) = [d for n, d in events if n == "values"]
    assert [(m["id"], m["content"]) for m in values["messages"]] == [
        ("a", "Let me look."),
        ("b", "The answer."),
    ]


async def test_whitespace_then_tool_call_sends_no_messages_event(client, bearer, monkeypatch):
    meta = {"langgraph_node": "model"}
    tool = AIMessageChunk(
        content="",
        id="a",
        tool_call_chunks=[{"name": "search", "args": "{}", "id": "c1", "index": 0}],
    )
    _stream_of(
        monkeypatch,
        [
            ("messages", (AIMessageChunk(content="\n\n", id="a"), meta)),
            ("messages", (tool, meta)),
        ],
    )
    events = await _run_events(client, bearer, "t-ws-tool")
    assert "messages" not in [n for n, _ in events]


async def test_leading_whitespace_is_held_back_not_resent(client, bearer, monkeypatch):
    meta = {"langgraph_node": "model"}
    _stream_of(
        monkeypatch,
        [
            ("messages", (AIMessageChunk(content="\n\n", id="a"), meta)),
            ("messages", (AIMessageChunk(content="The ", id="a"), meta)),
        ],
    )
    events = await _run_events(client, bearer, "t-ws-lead")
    sent = [d[0]["content"] for n, d in events if n == "messages"]
    assert sent == ["The "]


async def test_mid_answer_whitespace_is_still_sent(client, bearer, monkeypatch):
    meta = {"langgraph_node": "model"}
    _stream_of(
        monkeypatch,
        [
            ("messages", (AIMessageChunk(content="Hi", id="a"), meta)),
            ("messages", (AIMessageChunk(content=" ", id="a"), meta)),
            ("messages", (AIMessageChunk(content="there", id="a"), meta)),
        ],
    )
    events = await _run_events(client, bearer, "t-ws-mid")
    sent = [d[0]["content"] for n, d in events if n == "messages"]
    assert sent == ["Hi", " ", "there"]


async def test_status_events_reach_the_client_before_the_answer(client, bearer, monkeypatch):
    _stream_of(
        monkeypatch,
        [
            ("custom", {"type": "status", "message": "Searching software..."}),
            ("custom", {"type": "other", "message": "ignored"}),
            ("custom", "not a dict"),
            ("custom", {"type": "status"}),
            ("custom", {"type": "status", "message": 42}),
            (
                "messages",
                (AIMessageChunk(content="Yes", id="m1"), {"langgraph_node": "model"}),
            ),
        ],
    )
    events = await _run_events(client, bearer, "t-status")
    names = [name for name, _ in events]
    custom = [data for name, data in events if name == "custom"]
    assert custom == [{"type": "status", "message": "Searching software..."}]
    assert names.index("custom") < names.index("messages")
    assert "error" not in names


# ---------------------------------------------------------------------------
# Server-side allocated-resources profile (D8-3028)
# ---------------------------------------------------------------------------

DELTA_PROFILE_DICT = {
    "allocated_resources": [
        {
            "name": "NCSA Delta GPU",
            "rp_slug": "delta",
            "resource_id": "delta-gpu.ncsa.access-ci.org",
        }
    ]
}


def _capture_stream_agent(monkeypatch) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        yield "updates", {"tool_calling_loop": {"final_answer": "ok"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)
    return captured


def _stub_profile_lookup(monkeypatch, profile):
    from unittest.mock import AsyncMock

    lookup = AsyncMock(return_value=profile)
    monkeypatch.setattr("src.api.thread_routes.allocated_profile_for", lookup)
    return lookup


def _use_real_lookup(monkeypatch, result):
    """The real allocated_profile_for, with the MCP call and groups faked."""
    from src.agent import profile_source

    calls: list[Any] = []

    class _MCP:
        async def call_tool(self, server, tool_name, arguments, acting_user=None):
            calls.append((server, tool_name, acting_user))
            return result

    class _Groups:
        def refresh_in_background(self):
            return None

        def list_slugs(self):
            return ["delta"]

        def slugs_for_resource_id(self, resource_id):
            return ["delta"] if resource_id.startswith("delta-") else []

    monkeypatch.setattr(profile_source, "MCPClient", _MCP)
    monkeypatch.setattr(profile_source, "get_rp_cache", _Groups)
    monkeypatch.setattr(
        "src.api.thread_routes.allocated_profile_for", profile_source.allocated_profile_for
    )
    return calls


def _accounts_result(accounts, state="rows_fresh"):
    from src.tools.mcp_client import MCPToolResult

    return MCPToolResult(success=True, data={"accounts": accounts, "state": state})


DELTA_ACCOUNT = {"resource_id": "delta-gpu.ncsa.access-ci.org", "rp_display_name": "NCSA Delta GPU"}


async def _run(client, headers=None, **body):
    return await client.post(
        "/api/v1/threads/t-profile/runs/stream",
        headers=headers or {},
        json={"input": {"messages": [{"role": "user", "content": "hi"}]}, **body},
    )


async def test_signed_in_run_gets_the_server_built_profile(client, bearer, monkeypatch):
    from src.agent.profile import UserProfile

    profile = UserProfile.model_validate(DELTA_PROFILE_DICT)
    lookup = _stub_profile_lookup(monkeypatch, profile)
    captured = _capture_stream_agent(monkeypatch)

    r = await _run(client, bearer)

    assert r.status_code == 200
    lookup.assert_awaited_once_with("runner@x")
    assert captured["profile"] == profile


async def test_signed_in_run_looks_up_allocations_as_the_caller(client, bearer, monkeypatch):
    from src.agent.profile import UserProfile

    calls = _use_real_lookup(monkeypatch, _accounts_result([DELTA_ACCOUNT]))
    captured = _capture_stream_agent(monkeypatch)

    await _run(client, bearer)

    assert calls == [("allocations", "get_my_rp_accounts", "runner@x")]
    assert captured["profile"] == UserProfile.model_validate(DELTA_PROFILE_DICT)


async def test_later_messages_in_a_thread_reuse_the_lookup(client, bearer, monkeypatch):
    calls = _use_real_lookup(monkeypatch, _accounts_result([DELTA_ACCOUNT]))
    _capture_stream_agent(monkeypatch)

    await _run(client, bearer)
    await _run(client, bearer)

    assert len(calls) == 1


async def test_anonymous_run_makes_no_lookup_and_has_no_profile(client, monkeypatch):
    calls = _use_real_lookup(monkeypatch, _accounts_result([DELTA_ACCOUNT]))
    captured = _capture_stream_agent(monkeypatch)

    r = await _run(client)

    assert r.status_code == 200
    assert calls == []
    assert captured["profile"] is None


async def test_a_failed_lookup_runs_the_turn_without_a_profile(client, bearer, monkeypatch):
    from src.tools.mcp_client import MCPToolResult

    _use_real_lookup(monkeypatch, MCPToolResult(success=False, error="HTTP 503"))
    captured = _capture_stream_agent(monkeypatch)

    r = await _run(client, bearer)

    assert r.status_code == 200
    assert "event: values" in r.text
    assert "event: error" not in r.text
    assert captured["profile"] is None


async def test_a_syncing_lookup_runs_the_turn_without_a_profile(client, bearer, monkeypatch):
    _use_real_lookup(monkeypatch, _accounts_result([], state="syncing"))
    captured = _capture_stream_agent(monkeypatch)

    await _run(client, bearer)

    assert captured["profile"] is None


async def test_the_browser_cannot_supply_a_profile(client, bearer, monkeypatch):
    _stub_profile_lookup(monkeypatch, None)
    captured = _capture_stream_agent(monkeypatch)

    r = await _run(
        client,
        bearer,
        profile=DELTA_PROFILE_DICT,
        context={"profile": DELTA_PROFILE_DICT},
        config={"configurable": {"profile": DELTA_PROFILE_DICT}},
    )

    assert r.status_code == 200
    assert captured["profile"] is None


async def test_the_lookup_runs_before_the_turn_timer_starts(client, bearer, monkeypatch):
    """The lookup must not eat the turn budget: the turn's deadline is set
    only after it returns."""
    from src.agent.turn_capture import get_turn_capture
    from src.config import settings

    finished: list[float] = []

    async def _slow_lookup(acting_user):
        await asyncio.sleep(0.05)
        finished.append(time.monotonic())

    monkeypatch.setattr("src.api.thread_routes.allocated_profile_for", _slow_lookup)
    seen: dict[str, Any] = {}

    async def _fake(**kwargs):
        seen["deadline"] = get_turn_capture()["turn_deadline"]
        yield "updates", {"tool_calling_loop": {"final_answer": "ok"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)

    await _run(client, bearer)

    assert seen["deadline"] - settings.AGENT_TURN_TIMEOUT_S >= finished[0]


async def _system_prompt_for_run(client, monkeypatch, headers=None) -> str:
    """Run one turn through the real stream_agent and loop node with a fake
    model, and return the system prompt the model saw."""
    from types import SimpleNamespace

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    import src.agent.nodes.tool_calling_loop as loop_mod

    prompts: list[str] = []

    class _RecordingModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "recording"

        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            prompts.extend(m.text for m in messages if isinstance(m, SystemMessage))
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Done."))])

    async def _fake_registry():
        return SimpleNamespace(catalog={"servers": []})

    monkeypatch.setattr("src.api.routes.get_registry", _fake_registry)
    monkeypatch.setattr(loop_mod, "get_llm", lambda *a, **k: _RecordingModel())

    r = await _run(client, headers)

    assert r.status_code == 200
    assert "event: error" not in r.text
    assert prompts
    return prompts[0]


async def test_the_profile_reaches_the_system_prompt_on_the_thread_path(
    client, bearer, monkeypatch
):
    _use_real_lookup(monkeypatch, _accounts_result([DELTA_ACCOUNT]))

    prompt = await _system_prompt_for_run(client, monkeypatch, bearer)

    assert "## User profile" in prompt
    assert "Allocated resources: NCSA Delta GPU (rp_name='delta')" in prompt


async def test_an_anonymous_system_prompt_has_no_profile(client, monkeypatch):
    _use_real_lookup(monkeypatch, _accounts_result([DELTA_ACCOUNT]))

    prompt = await _system_prompt_for_run(client, monkeypatch)

    assert "## User profile" not in prompt


async def test_metadata_goes_out_before_the_profile_lookup_finishes(seed_owner, monkeypatch):
    """The run_id reaches the client at once; the lookup runs alongside."""
    from types import SimpleNamespace

    from src.agent_token import AgentIdentity
    from src.api.thread_routes import RunRequest, thread_run_stream

    release = asyncio.Event()
    started = asyncio.Event()

    async def _blocked_lookup(acting_user):
        started.set()
        await release.wait()

    monkeypatch.setattr("src.api.thread_routes.allocated_profile_for", _blocked_lookup)
    captured = _capture_stream_agent(monkeypatch)
    identity = AgentIdentity(
        sub="meta@x", scopes=frozenset({"agent:read"}), exp=int(time.time()) + 900
    )
    raw_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(checkpointer=None)))
    body = RunRequest.model_validate({"input": {"messages": [{"role": "user", "content": "hi"}]}})

    response = await thread_run_stream("t-meta-first", body, raw_request, identity)
    chunks = response.body_iterator.__aiter__()
    first = await asyncio.wait_for(chunks.__anext__(), timeout=1)

    assert first.startswith("event: metadata")
    assert started.is_set()
    assert not release.is_set()

    release.set()
    rest = [c async for c in chunks]
    assert any(c.startswith("event: values") for c in rest)
    assert "metadata" not in "".join(rest)
    assert captured["profile"] is None


async def test_a_disconnect_during_the_lookup_cancels_it(seed_owner, monkeypatch):
    from types import SimpleNamespace

    from src.agent_token import AgentIdentity
    from src.api.thread_routes import RunRequest, thread_run_stream

    cancelled = asyncio.Event()
    started = asyncio.Event()

    async def _hanging_lookup(acting_user):
        started.set()
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr("src.api.thread_routes.allocated_profile_for", _hanging_lookup)
    identity = AgentIdentity(
        sub="gone@x", scopes=frozenset({"agent:read"}), exp=int(time.time()) + 900
    )
    raw_request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(checkpointer=None)))
    body = RunRequest.model_validate({"input": {"messages": [{"role": "user", "content": "hi"}]}})

    response = await thread_run_stream("t-meta-gone", body, raw_request, identity)
    chunks = response.body_iterator
    await chunks.__anext__()  # metadata
    await asyncio.wait_for(started.wait(), timeout=1)
    await chunks.aclose()  # client went away mid-lookup

    await asyncio.sleep(0)
    assert cancelled.is_set()
