"""Battery runs write turn_reports rows (origin='battery') via the eval runner."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

STATE = {
    "final_answer": "An answer. https://access-ci.org/x",
    "tool_results": [],
    "tools_used": [],
    "node_trace": [],
}


def _patches(reporter):
    return (
        patch("src.eval.runner.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.runner.resources_for_turn", new=AsyncMock(return_value=[])),
        patch("src.eval.runner.get_turn_reporter", return_value=reporter),
    )


@pytest.mark.asyncio
async def test_battery_run_writes_turn_report():
    from src.eval.runner import run_question

    reporter = MagicMock()
    p1, p2, p3 = _patches(reporter)
    with p1, p2, p3:
        result = await run_question(
            "q1",
            "How do I check my allocation?",
            tool_catalog=None,
            battery_id="phase3_smoke_battery",
            battery_run_id="run-A",
        )

    assert result.success
    kwargs = reporter.log_turn_report.call_args.kwargs
    assert kwargs["origin"] == "battery"
    assert kwargs["battery_id"] == "phase3_smoke_battery"
    assert kwargs["battery_run_id"] == "run-A"
    assert kwargs["session_id"] == "eval_run-A_q1"
    assert kwargs["turn_index"] == 1
    assert kwargs["question_id"] == "q1"
    assert kwargs["success"] is True


@pytest.mark.asyncio
async def test_session_ids_distinct_across_runs():
    from src.eval.runner import run_question

    reporter = MagicMock()
    session_ids = []
    for run_id in ("run-A", "run-B"):
        p1, p2, p3 = _patches(reporter)
        with p1, p2, p3:
            await run_question(
                "q1",
                "Same question, two runs",
                tool_catalog=None,
                battery_id="phase3_smoke_battery",
                battery_run_id=run_id,
            )
        session_ids.append(reporter.log_turn_report.call_args.kwargs["session_id"])

    assert session_ids == ["eval_run-A_q1", "eval_run-B_q1"]


@pytest.mark.asyncio
async def test_no_battery_context_writes_nothing():
    from src.eval.runner import run_question

    reporter = MagicMock()
    p1, p2, p3 = _patches(reporter)
    with p1, p2, p3:
        await run_question("q1", "Plain eval question", tool_catalog=None)

    reporter.log_turn_report.assert_not_called()


@pytest.mark.asyncio
async def test_reporter_failure_does_not_fail_the_run():
    from src.eval.runner import run_question

    reporter = MagicMock()
    reporter.log_turn_report.side_effect = RuntimeError("db down")
    p1, p2, p3 = _patches(reporter)
    with p1, p2, p3:
        result = await run_question(
            "q1",
            "How do I check my allocation?",
            tool_catalog=None,
            battery_id="phase3_smoke_battery",
            battery_run_id="run-A",
        )

    assert result.success  # reporting is best-effort; the eval answer still counts


@pytest.mark.asyncio
async def test_raw_rag_battery_run_writes_turn_report():
    # raw_rag answers must be reviewable too (humans score BOTH systems for the
    # judge-vs-human comparison), so the raw_rag path writes battery turns —
    # with an answer-only state (no tools, no trace).
    from src.eval.runner import run_question

    uky_response = MagicMock()
    uky_response.response = "A retrieval-only answer."
    client = MagicMock()
    client.ask = AsyncMock(return_value=uky_response)

    reporter = MagicMock()
    _, p2, p3 = _patches(reporter)
    with p2, p3, patch("src.eval.runner.get_uky_client", return_value=client):
        result = await run_question(
            "q1",
            "How do I check my allocation?",
            tool_catalog=None,
            system="raw_rag",
            battery_id="gapfill",
            battery_run_id="run-R",
        )

    assert result.success
    kwargs = reporter.log_turn_report.call_args.kwargs
    assert kwargs["origin"] == "battery"
    assert kwargs["battery_run_id"] == "run-R"
    assert kwargs["session_id"] == "eval_run-R_q1"
    assert kwargs["final_state"]["final_answer"] == "A retrieval-only answer."


@pytest.mark.asyncio
async def test_raw_rag_without_battery_context_writes_nothing():
    from src.eval.runner import run_question

    uky_response = MagicMock()
    uky_response.response = "An answer."
    client = MagicMock()
    client.ask = AsyncMock(return_value=uky_response)

    reporter = MagicMock()
    _, p2, p3 = _patches(reporter)
    with p2, p3, patch("src.eval.runner.get_uky_client", return_value=client):
        await run_question("q1", "Plain eval question", tool_catalog=None, system="raw_rag")

    reporter.log_turn_report.assert_not_called()


@pytest.mark.asyncio
async def test_raw_rag_failure_still_writes_failed_report():
    # Mirrors the agent path: an /ask outage (e.g. the 2026-07-09 500s) must
    # leave a failed battery turn, not silently nothing.
    from src.eval.runner import run_question

    client = MagicMock()
    client.ask = AsyncMock(side_effect=RuntimeError("UKY /ask 500"))

    reporter = MagicMock()
    _, p2, p3 = _patches(reporter)
    with p2, p3, patch("src.eval.runner.get_uky_client", return_value=client):
        result = await run_question(
            "q1",
            "How do I check my allocation?",
            tool_catalog=None,
            system="raw_rag",
            battery_id="gapfill",
            battery_run_id="run-R",
        )

    assert not result.success
    kwargs = reporter.log_turn_report.call_args.kwargs
    assert kwargs["success"] is False
    assert kwargs["origin"] == "battery"
    assert kwargs["session_id"] == "eval_run-R_q1"


@pytest.mark.asyncio
async def test_agent_failure_still_writes_failed_report():
    from src.eval.runner import run_question

    reporter = MagicMock()
    p1, p2, p3 = _patches(reporter)
    with (
        p1,
        p2,
        p3,
        patch("src.eval.runner.run_agent", new=AsyncMock(side_effect=RuntimeError("agent died"))),
    ):
        result = await run_question(
            "q1",
            "How do I check my allocation?",
            tool_catalog=None,
            battery_id="phase3_smoke_battery",
            battery_run_id="run-A",
        )

    assert not result.success
    kwargs = reporter.log_turn_report.call_args.kwargs
    assert kwargs["success"] is False
    assert kwargs["origin"] == "battery"
    assert kwargs["battery_run_id"] == "run-A"
    assert kwargs["session_id"] == "eval_run-A_q1"
