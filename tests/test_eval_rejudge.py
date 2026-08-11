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


async def test_rejudge_replays_required_facts(mock_db):
    """The scorer freezes required_facts in context; rejudge must pass them back
    to the judge so rejudged runs keep fact-grounded verdicts (and the right
    token budget)."""
    db = EvalDB(mock_db)
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
        answer_text="ACCESS is a program.",
        context={
            "rag_context": None,
            "tool_results": None,
            "node_trace": None,
            "required_facts": [{"fact_id": 42, "fact_text": "ACCESS allocates HPC resources"}],
        },
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

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        await rejudge_run(original_run_id=str(original.id), database_url=mock_db)
        # The prompt sent to the judge must contain the stored fact (and its stable id).
        sent = mock_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
        assert "ACCESS allocates HPC resources" in sent
        assert "Required Facts" in sent


async def test_rejudge_propagates_mode_metadata(mock_db):
    """A rejudged multiturn run must stay excluded from the dashboard aggregate, so
    metadata.mode rides along to the new run."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=1,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-f-01_t2",
        source="judge",
        question_text="Which have A100s?",
        answer_text="Delta and DeltaAI.",
        context={"rag_context": None, "tool_results": None, "node_trace": None},
        composite_score=0.5,
    )

    with patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create = AsyncMock(return_value=_completion(MOCK_JUDGE_BEST))
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    new_run = db.get_run(str(summary["new_run_id"]))
    assert new_run.metadata_["mode"] == "multiturn"
    assert new_run.metadata_["rejudged_from"] == str(original.id)


async def test_rejudge_replays_conversation_history(mock_db):
    """Multiturn rows persist the prior transcript; replaying it keeps
    reference-resolution turns judgeable instead of recording phantom drops."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="mt.yaml",
        question_count=1,
        metadata_={"system": "agent_full", "mode": "multiturn"},
    )
    db.add_score(
        run_id=original.id,
        question_id="mt-f-01_t2",
        source="judge",
        question_text="Which of those have A100s?",
        answer_text="Delta and DeltaAI.",
        # Stored as list-of-lists — JSON has no tuples.
        context={
            "rag_context": None,
            "tool_results": None,
            "node_trace": None,
            "conversation_history": [["What GPU resources exist?", "Delta, DeltaAI, Anvil."]],
        },
        composite_score=0.5,
    )

    judge = MagicMock()
    judge.score = AsyncMock(return_value=None)  # judge_error path; we only assert the call
    with patch("src.eval.rejudge.Judge", return_value=judge):
        await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    history = judge.score.call_args.kwargs["conversation_history"]
    assert history == [("What GPU resources exist?", "Delta, DeltaAI, Anvil.")]


async def test_rejudge_single_turn_passes_no_history(mock_db):
    """Single-turn rows have no stored history — the judge call must stay unchanged."""
    db = EvalDB(mock_db)
    original = db.create_run(
        run_type="pre_production",
        llm_model="qwen",
        question_set="tiny",
        question_count=1,
        metadata_={"system": "access-agent"},
    )
    db.add_score(
        run_id=original.id,
        question_id="q1",
        source="judge",
        question_text="What is ACCESS?",
        answer_text="A program.",
        context={"rag_context": None, "tool_results": None, "node_trace": None},
        composite_score=0.5,
    )

    judge = MagicMock()
    judge.score = AsyncMock(return_value=None)
    with patch("src.eval.rejudge.Judge", return_value=judge):
        summary = await rejudge_run(original_run_id=str(original.id), database_url=mock_db)

    assert judge.score.call_args.kwargs["conversation_history"] is None
    # No mode on the source run means no mode key invented on the rejudge run.
    assert "mode" not in (db.get_run(str(summary["new_run_id"])).metadata_ or {})
