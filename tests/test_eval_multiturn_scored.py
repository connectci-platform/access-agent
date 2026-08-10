"""Scored multiturn path: per-turn judging, persistence, history propagation."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval.db import EvalDB
from src.eval.judge import JudgeResult

GOOD = JudgeResult(
    scores={
        "correctness": 2,
        "specificity": 2,
        "relevance": 2,
        "citation_quality": 2,
        "hedging": 1,
    },
    justifications={},
    composite=0.9,
    answerable=True,
)

THREAD = {
    "thread_id": "mt-f-01",
    "description": "d",
    "questions": [
        {"turn_id": "t1", "question": "What GPU resources exist?"},
        {"turn_id": "t2", "question": "Which have A100s?"},
        {"turn_id": "t3", "question": "Walltime on that one?"},
    ],
}

STATE = {"final_answer": "An answer.", "tools_used": ["search_resources"], "messages": [1, 2]}


def _scoring(tmp_path):
    from src.eval.multiturn import ScoringContext

    db = EvalDB(f"sqlite:///{tmp_path}/eval.db")
    run = db.create_run(run_type="pre_production")
    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    return (
        ScoringContext(db=db, judge=judge, run_id=str(run.id), battery_id="test_battery"),
        db,
        judge,
    )


@pytest.mark.asyncio
async def test_scored_thread_persists_one_row_per_turn(tmp_path):
    from src.eval.multiturn import run_thread

    scoring, db, judge = _scoring(tmp_path)
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        result = await run_thread(
            THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring
        )

    rows = db.get_scores_for_run(scoring.run_id)
    # Sort in the assertion — get_scores_for_run orders by created_at, which can tie
    # across three fast mocked inserts on SQLite (review finding: don't rely on it).
    assert sorted(r.question_id for r in rows) == ["mt-f-01_t1", "mt-f-01_t2", "mt-f-01_t3"]
    assert all(r.source == "judge" for r in rows)
    assert all(t.composite == 0.9 for t in result.turns)
    # Turn 3's judge call saw turns 1-2 as history
    hist = judge.score.call_args_list[2].kwargs["conversation_history"]
    assert [q for q, _ in hist] == ["What GPU resources exist?", "Which have A100s?"]


@pytest.mark.asyncio
async def test_failed_turn_marks_history_and_writes_skipped(tmp_path):
    from src.eval.multiturn import run_thread

    scoring, db, judge = _scoring(tmp_path)
    calls = {"n": 0}

    async def flaky_agent(**kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("boom")
        return STATE

    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=flaky_agent)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    rows = {r.question_id: r for r in db.get_scores_for_run(scoring.run_id)}
    assert rows["mt-f-01_t2"].source == "skipped"
    hist = judge.score.call_args_list[-1].kwargs["conversation_history"]
    assert hist[1][1] == "(no answer — turn failed)"


@pytest.mark.asyncio
async def test_turn_reports_carry_real_turn_index(tmp_path):
    from src.eval.multiturn import run_thread

    scoring, _, _ = _scoring(tmp_path)
    reporter = AsyncMock()
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=reporter),
    ):
        await run_thread(THREAD, tool_catalog=None, session_namespace="ns", scoring=scoring)

    indexes = [c.kwargs["turn_index"] for c in reporter.call_args_list]
    assert indexes == [1, 2, 3]
