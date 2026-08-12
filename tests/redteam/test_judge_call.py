from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.eval.judge import Judge


def _choice(content, finish="stop"):
    c = MagicMock()
    c.message.content = content
    c.finish_reason = finish
    r = MagicMock()
    r.choices = [c]
    return r


@pytest.mark.asyncio
async def test_call_once_strips_think_and_reports_not_truncated():
    j = Judge(base_url="http://onprem/v1", model="qwen", thinking=False)
    with patch.object(
        j.client.chat.completions,
        "create",
        new=AsyncMock(return_value=_choice("reasoning...</think>FINAL")),
    ):
        raw, truncated = await j._call_once("prompt", 500)
    assert raw == "FINAL" and truncated is False


@pytest.mark.asyncio
async def test_call_once_reports_truncation():
    j = Judge(base_url="http://onprem/v1", model="qwen")
    with patch.object(
        j.client.chat.completions,
        "create",
        new=AsyncMock(return_value=_choice("partial", finish="length")),
    ):
        _raw, truncated = await j._call_once("prompt", 500)
    assert truncated is True


@pytest.mark.asyncio
async def test_call_once_none_on_exception():
    j = Judge(base_url="http://onprem/v1", model="qwen")
    with patch.object(
        j.client.chat.completions,
        "create",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        raw, _truncated = await j._call_once("prompt", 500)
    assert raw is None


@pytest.mark.asyncio
async def test_score_caps_at_two_api_calls():
    # BEHAVIOR PRESERVATION: a parse-failure on both attempts = exactly 2 API calls, not more.
    j = Judge(base_url="http://onprem/v1", model="qwen")
    create = AsyncMock(return_value=_choice("not json"))
    with patch.object(j.client.chat.completions, "create", new=create):
        result = await j.score(query="q", answer="a")
    assert result is None
    assert create.await_count == 2


def _good_judge_payload():
    return """```json
{
  "answerable": "Fair",
  "correctness": {"value": "Correct", "justification": "a"},
  "specificity": {"value": "Actionable", "justification": "b"},
  "relevance": {"value": "On-target", "justification": "c"},
  "citation_quality": {"value": "Good", "justification": "d"},
  "hedging": {"value": "Calibrated", "justification": "e"}
}
```"""


@pytest.mark.asyncio
async def test_score_retries_after_call_once_returns_none():
    # _call_once returning (None, False) on the first attempt (the exception path)
    # must be skipped via `continue`, not treated as a terminal failure.
    j = Judge(base_url="http://onprem/v1", model="qwen")
    call_once = AsyncMock(
        side_effect=[
            (None, False),
            (_good_judge_payload(), False),
        ]
    )
    with patch.object(j, "_call_once", new=call_once):
        result = await j.score(query="q", answer="a")
    assert result is not None
    assert result.scores["correctness"] == 2
    assert call_once.await_count == 2
