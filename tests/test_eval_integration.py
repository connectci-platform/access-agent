"""Integration test for the eval pipeline.

Mocks run_agent and the judge LLM to test the full pipeline
without requiring live services.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval.db import EvalDB
from src.eval.scorer import run_eval


@pytest.fixture
def mock_db(tmp_path):
    return f"sqlite:///{tmp_path}/test_eval.db"


@pytest.fixture
def tiny_question_set(tmp_path):
    questions = [
        {"id": "t1", "question": "What is ACCESS?", "capability_area": "general"},
        {"id": "t2", "question": "How do I get an allocation?", "capability_area": "allocations"},
        {"id": "t3", "question": "Is Delta down?", "capability_area": "delta"},
    ]
    path = tmp_path / "test_questions.json"
    path.write_text(json.dumps(questions))
    return str(path)


MOCK_JUDGE_RESPONSE = json.dumps(
    {
        "correctness": {"score": 4, "justification": "Accurate"},
        "completeness": {"score": 4, "justification": "Thorough"},
        "relevance": {"score": 5, "justification": "On topic"},
        "citation_quality": {"score": 3, "justification": "Some URLs"},
        "hedging": {"score": 4, "justification": "Well calibrated"},
    }
)


@pytest.mark.asyncio
async def test_full_eval_run(mock_db, tiny_question_set):
    mock_state = {
        "final_answer": "ACCESS is a program for HPC resources.",
        "rag_matches": [],
        "tool_results": [],
        "node_trace": [{"node": "classify", "query_type": "static"}],
        "tools_used": [],
    }

    # Create a mock completion response object
    mock_message = MagicMock()
    mock_message.content = MOCK_JUDGE_RESPONSE
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    with (
        patch("src.eval.runner.run_agent", new_callable=AsyncMock, return_value=mock_state),
        patch("src.eval.scorer.ToolRegistry") as mock_registry_cls,
        patch("src.eval.judge.AsyncOpenAI") as mock_openai_cls,
    ):
        # Mock tool registry
        mock_registry = AsyncMock()
        mock_registry.tool_count = 10
        mock_registry.catalog = {"tools": [{"name": "test_tool"}]}
        mock_registry_cls.return_value = mock_registry

        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai_cls.return_value = mock_client

        summary = await run_eval(
            question_set_path=tiny_question_set,
            database_url=mock_db,
        )

    assert summary["questions"] == 3
    assert summary["scored"] == 3
    assert summary["skipped"] == 0
    assert summary["composite_score"] > 0
    assert "correctness" in summary["per_dimension"]

    # Verify DB has the data
    db = EvalDB(mock_db)
    run = db.get_run(summary["run_id"])
    assert run is not None
    assert run.question_count == 3

    scores = db.get_scores_for_run(summary["run_id"])
    assert len(scores) == 3
    assert all(s.source == "judge" for s in scores)
