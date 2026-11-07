"""Every /query turn writes exactly one turn report: answered, failed or
timed out.

Harness copied from tests/test_stream_events_branches.py: drive _stream_events
directly with stream_agent/get_registry mocked. The reporter is a recording
fake that reports two earlier turns on the session.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.context import AgentContext
from src.api import routes
from src.turnstile import NOT_CHARGED, charge_query, get_turnstile_guard


async def _fake_registry():
    return SimpleNamespace(catalog={"servers": []})


def _raw_request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(checkpointer=None)))


@pytest.fixture
def reporter():
    fake = MagicMock()
    fake.count_turns_for_session.return_value = 2
    with (
        patch("src.turn_reporter.get_turn_reporter", return_value=fake),
        patch("src.services.resource_matcher.resources_for_turn", AsyncMock(return_value=[])),
        patch.object(routes, "get_usage_logger", return_value=MagicMock()),
    ):
        yield fake


async def _drain(fake_stream, run_id: str | None = None, charge=NOT_CHARGED) -> list[str]:
    events: list[str] = []
    with (
        patch.object(routes, "stream_agent", fake_stream),
        patch.object(routes, "get_registry", _fake_registry),
    ):
        gen = routes._stream_events(
            routes.QueryRequest(query="q", session_id="s1"),
            context=AgentContext.anonymous(),
            session_id="s1",
            question_id="q1",
            include_trace=False,
            raw_request=_raw_request(),
            run_id=run_id,
            charge=charge,
        )
        try:
            async for line in gen:
                events += [p[len("event: ") :] for p in line.split("\n") if p.startswith("event: ")]
        finally:
            await gen.aclose()
    return events


def _reports(reporter) -> list[dict]:
    return [call.kwargs for call in reporter.log_turn_report.call_args_list]


async def _answers(**_kwargs):
    yield "updates", {"tool_calling_loop": {"final_answer": "an answer"}}


def test_answered_turn_writes_one_success_report(reporter):
    events = asyncio.run(_drain(_answers))

    assert events[-1] == "done"
    (report,) = _reports(reporter)
    assert report["success"] is True
    assert report["turn_index"] == 3


def test_turn_failing_before_its_report_writes_one_failure_report(reporter):
    async def fails(**_kwargs):
        raise RuntimeError("agent blew up")
        yield  # pragma: no cover - unreachable, makes this an async generator

    events = asyncio.run(_drain(fails))

    assert events == ["error", "done"]
    (report,) = _reports(reporter)
    assert report["success"] is False
    assert report["turn_index"] == 3


def test_timed_out_turn_writes_one_failure_report(reporter):
    async def times_out(**_kwargs):
        raise TimeoutError
        yield  # pragma: no cover - unreachable, makes this an async generator

    events = asyncio.run(_drain(times_out))

    assert events == ["error", "done"]
    (report,) = _reports(reporter)
    assert report["success"] is False
    assert report["turn_index"] == 3


def test_failure_after_the_answer_does_not_fail_the_turn(reporter):
    """Bookkeeping after done can't turn an answered turn into a failed one."""
    boom = MagicMock(side_effect=RuntimeError("store down"))

    with patch.object(routes, "get_usage_logger", boom):
        events = asyncio.run(_drain(_answers))

    assert events == ["done"]
    (report,) = _reports(reporter)
    assert report["success"] is True


@pytest.fixture
def charged(monkeypatch):
    """A question counted against session s1, as the handler does on acquire."""
    monkeypatch.setattr("src.turnstile._guard", None)
    return charge_query(None, "s1")


def _s1_count() -> int:
    return get_turnstile_guard()._sessions["s1"].query_count


async def _fails(**_kwargs):
    raise RuntimeError("model down")
    yield  # pragma: no cover


async def _times_out(**_kwargs):
    raise TimeoutError
    yield  # pragma: no cover


async def _cancelled(**_kwargs):
    raise asyncio.CancelledError
    yield  # pragma: no cover


def test_answered_turn_keeps_its_question_used(reporter, charged):
    asyncio.run(_drain(_answers, charge=charged))

    assert _s1_count() == 1


@pytest.mark.parametrize("stream", [_fails, _times_out], ids=["agent_error", "timeout"])
def test_failed_turn_refunds_its_question(reporter, charged, stream):
    asyncio.run(_drain(stream, charge=charged))

    assert _s1_count() == 0


def test_cancelled_turn_keeps_its_question_used(reporter, charged):
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_drain(_cancelled, charge=charged))

    assert _s1_count() == 1


def test_report_write_failure_never_reaches_the_stream(reporter):
    reporter.log_turn_report.side_effect = RuntimeError("db down")

    events = asyncio.run(_drain(_answers))

    assert events == ["done"]
    reporter.log_turn_report.assert_called_once()


def test_timed_out_turn_is_reported_after_the_thread_lock_is_released(reporter):
    from src.api.thread_runs import _active_by_thread, acquire_thread_run, release_thread_run

    # The /query handler acquires the lock before streaming; do the same.
    acquire_thread_run("s1", "run-1")
    held_during_report: list[bool] = []
    reporter.log_turn_report.side_effect = lambda **_kw: held_during_report.append(
        "s1" in _active_by_thread
    )

    async def times_out(**_kwargs):
        raise TimeoutError
        yield  # pragma: no cover - unreachable, makes this an async generator

    try:
        asyncio.run(_drain(times_out, run_id="run-1"))
    finally:
        release_thread_run("s1", "run-1")

    assert held_during_report == [False]
