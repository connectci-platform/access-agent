"""A deleted conversation doesn't come back (D8-3126 review).

Each test is one way a write could land after a delete: a stopped turn's
save, the turn report written after the run lock is released, a run that
passed its access check before the delete, a rating, and a new run claiming
the deleted id.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src import conversation_delete, thread_lifecycle
from src.agent.graph import create_checkpointed_graph
from src.api import rating_routes, stopped_turns, thread_access, thread_routes, thread_runs
from src.config import settings
from src.thread_lifecycle import mark_deleted, writing
from src.thread_owners import ThreadAccess

TID = "11111111-1111-1111-1111-111111111111"


async def test_a_stopped_turn_saved_after_the_delete_isnt_written_back():
    # The reviewer's repro: a first question stopped, then deleted, while the
    # save task was still to run.
    checkpointer = InMemorySaver()
    mark_deleted([TID])
    await checkpointer.adelete_thread(TID)
    stopped_turns.keep_stopped_turn(
        checkpointer,
        TID,
        HumanMessage(content="my secret question", id="q1"),
        {"a1": AIMessageChunk(content="partial answer", id="a1")},
        "user",
        start=None,
    )
    await stopped_turns.stopped_turn_saved(TID)

    state = await create_checkpointed_graph(checkpointer).aget_state(
        {"configurable": {"thread_id": TID}}
    )
    assert (state.values or {}).get("messages", []) == []


async def test_the_turn_report_written_after_the_lock_is_skipped(monkeypatch):
    reporter = SimpleNamespace(count_turns_for_session=lambda tid: 0, log_turn_report=None)
    calls: list[dict] = []
    reporter.log_turn_report = lambda **kw: calls.append(kw)
    monkeypatch.setattr("src.turn_reporter.get_turn_reporter", lambda: reporter)
    mark_deleted([TID])

    await thread_routes._write_turn_report(
        final_state={},
        thread_id=TID,
        run_id="r1",
        query="q",
        start_time=0.0,
        caller="alice@access-ci.org",
        success=True,
    )

    assert calls == []


async def test_the_delete_waits_for_a_report_already_being_written(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'w.db'}", raising=False)
    conversation_delete._reset_engine_for_test()
    order: list[str] = []
    import asyncio

    started, release = asyncio.Event(), asyncio.Event()

    async def report():
        async with writing(TID):
            started.set()
            await release.wait()
            order.append("report written")

    task = asyncio.create_task(report())
    await started.wait()
    deleting = asyncio.create_task(
        conversation_delete.delete_conversations(
            [TID], user="alice@access-ci.org", scope="one", checkpointer=None
        )
    )
    await asyncio.sleep(0.01)
    assert not deleting.done()
    release.set()
    await task
    assert await deleting == 1
    order.append("deleted")
    assert order == ["report written", "deleted"]
    conversation_delete._reset_engine_for_test()


async def test_a_write_that_wont_finish_makes_the_delete_a_conflict(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 'b.db'}", raising=False)
    monkeypatch.setattr(conversation_delete, "SETTLE_TIMEOUT_S", 0.01)
    conversation_delete._reset_engine_for_test()
    import asyncio

    started, release = asyncio.Event(), asyncio.Event()

    async def slow():
        async with writing(TID):
            started.set()
            await release.wait()

    task = asyncio.create_task(slow())
    await started.wait()
    checkpointer = AsyncMock()
    with pytest.raises(conversation_delete.ConversationBusyError):
        await conversation_delete.delete_conversations(
            [TID], user="alice@access-ci.org", scope="one", checkpointer=checkpointer
        )
    # Nothing deleted, and the conversation still works.
    checkpointer.adelete_thread.assert_not_awaited()
    assert not thread_lifecycle.is_deleted(TID)
    release.set()
    await task
    conversation_delete._reset_engine_for_test()


def test_a_run_that_passed_its_check_before_the_delete_is_refused():
    thread_runs.acquire_thread_run(TID, "run-1")
    mark_deleted([TID])
    with pytest.raises(HTTPException) as refused:
        thread_routes._refuse_if_deleted(TID, "run-1")
    assert refused.value.status_code == 404
    # The lock was let go.
    thread_runs.acquire_thread_run(TID, "run-2")
    thread_runs.release_thread_run(TID, "run-2")


def test_a_run_on_a_live_thread_goes_ahead():
    thread_runs.acquire_thread_run(TID, "run-1")
    try:
        thread_routes._refuse_if_deleted(TID, "run-1")
    finally:
        thread_runs.release_thread_run(TID, "run-1")


async def test_a_rating_after_the_delete_is_refused(monkeypatch):
    monkeypatch.setattr(rating_routes, "require_thread_access", lambda *a, **k: None)
    monkeypatch.setattr(rating_routes, "_is_turn_answer", AsyncMock(return_value=True))
    rate = AsyncMock()
    monkeypatch.setattr(
        rating_routes, "get_turn_reporter", lambda: SimpleNamespace(rate_answer=rate)
    )
    mark_deleted([TID])

    with pytest.raises(HTTPException) as refused:
        await rating_routes.rate_answer(
            TID,
            "m1",
            rating_routes.AnswerRatingRequest(rating="helpful"),
            SimpleNamespace(),
            None,
        )
    assert refused.value.status_code == 404
    rate.assert_not_called()


def test_a_deleted_id_cant_be_claimed_as_a_new_conversation(monkeypatch):
    store = SimpleNamespace(
        access=lambda *a: ThreadAccess.NEW,
        claim_thread=lambda *a: pytest.fail("claimed a deleted id"),
    )
    monkeypatch.setattr(thread_access, "get_thread_owner_store", lambda: store)
    mark_deleted([TID])

    with pytest.raises(HTTPException) as refused:
        thread_access.require_thread_access(TID, "mallory@access-ci.org", None, create=True)
    assert refused.value.status_code == 404
