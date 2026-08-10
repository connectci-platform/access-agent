"""Scored multiturn path: per-turn judging, persistence, history propagation."""

import json
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


@pytest.mark.asyncio
async def test_scored_battery_creates_run_and_composites(tmp_path):
    from src.eval import multiturn

    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([THREAD, {**THREAD, "thread_id": "mt-f-02"}]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    assert summary is not None
    assert summary["run_id"].startswith("mtloop-")
    assert summary["composite_score"] == pytest.approx(0.9)
    assert summary["thread_composites"] == {
        "mt-f-01": pytest.approx(0.9),
        "mt-f-02": pytest.approx(0.9),
    }
    db = EvalDB(db_url)
    run = db.get_run(summary["run_id"])
    assert run.metadata_["mode"] == "multiturn"
    assert run.question_count == 6


@pytest.mark.asyncio
async def test_all_failed_thread_excluded_from_run_composite(tmp_path):
    from src.eval import multiturn

    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([THREAD, {**THREAD, "thread_id": "mt-dead"}]))
    db_url = f"sqlite:///{tmp_path}/eval.db"

    async def agent_by_thread(**kwargs):
        if "mt-dead" in kwargs["session_id"]:
            raise RuntimeError("dead thread")
        return STATE

    judge = MagicMock()
    judge.score = AsyncMock(return_value=GOOD)
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(side_effect=agent_by_thread)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    assert summary["unscored_threads"] == ["mt-dead"]
    assert summary["composite_score"] == pytest.approx(0.9)  # dead thread excluded, not 0.45


@pytest.mark.asyncio
async def test_run_composite_is_mean_of_thread_means_not_turns(tmp_path):
    """Long-thread non-dominance: a 3-turn thread and a 1-turn thread weigh equally."""
    from src.eval import multiturn
    from src.eval.judge import JudgeResult

    short_thread = {
        "thread_id": "mt-short",
        "questions": [{"turn_id": "t1", "question": "One?"}],
    }
    battery = tmp_path / "b.json"
    battery.write_text(json.dumps([THREAD, short_thread]))  # 3 turns + 1 turn
    db_url = f"sqlite:///{tmp_path}/eval.db"

    def _result(composite):
        return JudgeResult(
            scores={
                "correctness": 2,
                "specificity": 2,
                "relevance": 2,
                "citation_quality": 2,
                "hedging": 1,
            },
            justifications={},
            composite=composite,
            answerable=True,
        )

    # 3-turn thread scores 0.3 per turn; 1-turn thread scores 0.9.
    judge = MagicMock()
    judge.score = AsyncMock(side_effect=[_result(0.3)] * 3 + [_result(0.9)])
    with (
        patch("src.eval.multiturn.run_agent", new=AsyncMock(return_value=STATE)),
        patch("src.eval.multiturn.report_battery_turn", new=AsyncMock()),
        patch("src.eval.multiturn.get_catalog_aggregator") as agg,
        patch("src.eval.multiturn._build_judge", return_value=judge),
    ):
        agg.return_value.fetch_catalog = AsyncMock(return_value={"tools": []})
        _, summary = await multiturn.run_battery(
            battery_path=str(battery), score=True, database_url=db_url
        )

    # Mean of thread means = (0.3 + 0.9) / 2 = 0.6; mean of turns would be 0.45.
    assert summary["composite_score"] == pytest.approx(0.6)
