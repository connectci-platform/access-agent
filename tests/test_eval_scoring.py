"""Shared judge-then-persist write path (single source of the eval_scores contract)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.eval.db import EvalDB
from src.eval.judge import JudgeResult


def _db(tmp_path):
    return EvalDB(f"sqlite:///{tmp_path}/eval.db")


def _judge_returning(result):
    judge = MagicMock()
    judge.score = AsyncMock(return_value=result)
    return judge


GOOD_RESULT = JudgeResult(
    scores={
        "correctness": 2,
        "specificity": 2,
        "relevance": 2,
        "citation_quality": 2,
        "hedging": 1,
    },
    justifications={"correctness": "j"},
    composite=1.0,
    answerable=True,
    specificity_na=False,
    fact_verdicts=[{"id": "f1", "verdict": "yes", "justification": "j"}],
)


@pytest.mark.asyncio
async def test_judge_success_writes_full_row(tmp_path):
    from src.eval.scoring import RUBRIC_VERSION, score_and_persist_turn

    db = _db(tmp_path)
    run = db.create_run(run_type="pre_production")
    result = await score_and_persist_turn(
        db,
        _judge_returning(GOOD_RESULT),
        run_id=str(run.id),
        question_id="t-01_t2",
        question_text="Which of those have A100s?",
        answer="Delta does.",
        rag_context="rag",
        tool_results="tools",
        conversation_history=[("q1", "a1")],
        extra_context={"thread_id": "t-01", "turn_index": 2},
        duration_ms=42.0,
    )

    assert result is GOOD_RESULT
    row = db.get_scores_for_run(str(run.id))[0]
    assert row.source == "judge"
    assert row.rubric_version == RUBRIC_VERSION
    assert row.context_completeness == "full"
    assert row.composite_score == 1.0
    assert row.context["thread_id"] == "t-01"
    assert row.context["fact_verdicts"][0]["id"] == "f1"


@pytest.mark.asyncio
async def test_judge_failure_writes_judge_error_row(tmp_path):
    from src.eval.scoring import score_and_persist_turn

    db = _db(tmp_path)
    run = db.create_run(run_type="pre_production")
    result = await score_and_persist_turn(
        db,
        _judge_returning(None),
        run_id=str(run.id),
        question_id="t-01_t1",
        question_text="q",
        answer="a",
        extra_context={"thread_id": "t-01", "turn_index": 1},
    )

    assert result is None
    row = db.get_scores_for_run(str(run.id))[0]
    assert row.source == "judge_error"
    assert row.context["thread_id"] == "t-01"


def test_skipped_row(tmp_path):
    from src.eval.scoring import persist_skipped_turn

    db = _db(tmp_path)
    run = db.create_run(run_type="pre_production")
    persist_skipped_turn(
        db,
        run_id=str(run.id),
        question_id="t-01_t3",
        question_text="q",
        error="agent died",
        duration_ms=5.0,
    )
    row = db.get_scores_for_run(str(run.id))[0]
    assert row.source == "skipped"
    assert row.answer_text == "agent died"
