"""Judge.score passes conversation_history through to the prompt builder."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

VALID_JUDGE_JSON = json.dumps(
    {
        "answerable": "Fair",
        "correctness": {"value": "Correct", "justification": "j"},
        "specificity": {"value": "Actionable", "justification": "j"},
        "relevance": {"value": "On-target", "justification": "j"},
        "citation_quality": {"value": "Good", "justification": "j"},
        "hedging": {"value": "Calibrated", "justification": "j"},
    }
)


def _fake_response(text: str):
    choice = MagicMock()
    choice.message.content = text
    choice.finish_reason = "stop"
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@pytest.mark.asyncio
async def test_score_passes_history_into_prompt():
    from src.eval.judge import Judge

    judge = Judge(model="test-model")
    create = AsyncMock(return_value=_fake_response(VALID_JUDGE_JSON))
    judge.client = MagicMock()
    judge.client.chat.completions.create = create

    result = await judge.score(
        query="Which of those have A100s?",
        answer="Delta does.",
        conversation_history=[("What GPU resources exist?", "Delta and Anvil.")],
    )

    assert result is not None
    prompt = create.call_args.kwargs["messages"][0]["content"]
    assert "## Conversation so far" in prompt
    assert "What GPU resources exist?" in prompt
