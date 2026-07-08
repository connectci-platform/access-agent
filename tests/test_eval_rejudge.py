"""Test for rejudge_run (v2 rubric replay).

Seeds an original run with one frozen judge score, mocks the judge LLM so no
network call is made, and replays it through rejudge_run. Verifies the new run
is linked back via metadata.rejudged_from and that the rescored row carries the
v2 fields (specificity, specificity_na, answerable, rubric_version=2, composite).
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval.db import EvalDB
from src.eval.rejudge import rejudge_run

# v2 judge payload: fair answer, actionable specificity, all-best -> composite 1.0.
MOCK_JUDGE_BEST = json.dumps(
    {
        "answerable": "Fair",
        "correctness": {"value": "Correct", "justification": "Accurate"},
        "specificity": {"value": "Actionable", "justification": "Named resources"},
        "relevance": {"value": "On-target", "justification": "On topic"},
        "citation_quality": {"value": "Good", "justification": "Valid URLs"},
        "hedging": {"value": "Calibrated", "justification": "Well calibrated"},
    }
)


def _completion(content: str) -> MagicMock:
    mock_message = MagicMock()
    mock_message.content = content
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    return mock_completion


@pytest.fixture
def mock_db(tmp_path):
    return f"sqlite:///{tmp_path}/test_rejudge.db"


async def test_rejudge_run_replays_and_writes_v2(mock_db):
    """Rejudge an original run: new run links back and rescored row is v2."""
    db = EvalDB(mock_db)

    # Seed an original run with one frozen judge score (question/answer/context).
    original = db.create_run(
        run_type="pre_production",
        agent_commit="abc123",
        agent_branch="main",
        llm_model="qwen",
        judge_model="gpt-4o-mini",
        question_set="tiny",
        question_count=1,
        composite_score=0.5,
        metadata_={"system": "access-agent"},
    )
    db.add_score(
        run_id=original.id,
        question_id="q1",
        source="judge",
        question_text="What is ACCESS?",
        answer_text="ACCESS is a program for HPC resources.",
        context={"rag_context": "some docs", "tool_results": None, "node_trace": None},
        context_completeness=1.0,
        correctness=1,
        specificity=1,
        specificity_na=False,
        answerable=True,
        rubric_version=2,
        relevance=1,
        citation_quality=1,
        hedging=1,
        composite_score=0.5,
        duration_ms=123,
    )

    # Mock the judge LLM so no network call is made (same pattern as the
    # integration test: patch AsyncOpenAI where Judge constructs it).
    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [_completion(MOCK_JUDGE_BEST)]
        mock_openai_cls.return_value = mock_client

        summary = await rejudge_run(
            original_run_id=str(original.id),
            database_url=mock_db,
        )

    # Summary reflects one rescored item.
    assert summary["rescored"] == 1
    assert summary["skipped"] == 0
    assert summary["errors"] == 0
    assert summary["original_run_id"] == str(original.id)
    new_run_id = str(summary["new_run_id"])
    assert new_run_id != str(original.id)

    # New run is a rejudge run linked back to the original via metadata.
    new_run = db.get_run(new_run_id)
    assert new_run is not None
    assert new_run.run_type == "rejudge"
    assert new_run.metadata_["rejudged_from"] == str(original.id)
    assert new_run.metadata_["system"] == "access-agent"
    assert new_run.composite_score is not None

    # The rescored row carries the v2 fields.
    new_scores = db.get_scores_for_run(new_run_id)
    assert len(new_scores) == 1
    rescored = new_scores[0]
    assert rescored.source == "judge"
    assert rescored.question_id == "q1"
    assert rescored.rubric_version == 2
    assert rescored.answerable is True
    assert rescored.specificity == 2  # "Actionable"
    assert rescored.specificity_na is False
    assert rescored.composite_score is not None
    assert abs(rescored.composite_score - 1.0) < 1e-9
