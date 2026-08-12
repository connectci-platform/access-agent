from unittest.mock import AsyncMock, patch

import pytest

from src.eval.judge import Judge, parse_compliance


@pytest.mark.asyncio
async def test_score_binary_true_on_complies():
    j = Judge(base_url="http://onprem/v1", model="qwen")
    with patch.object(j, "_call_once", new=AsyncMock(return_value=('{"complies": true}', False))):
        assert await j.score_binary("how to X", "sure, step one...") is True


@pytest.mark.asyncio
async def test_score_binary_false_on_refusal():
    j = Judge(base_url="http://onprem/v1", model="qwen")
    with patch.object(j, "_call_once", new=AsyncMock(return_value=('{"complies": false}', False))):
        assert await j.score_binary("how to X", "I can't help") is False


@pytest.mark.asyncio
async def test_score_binary_none_when_unparseable():
    j = Judge(base_url="http://onprem/v1", model="qwen")
    with patch.object(j, "_call_once", new=AsyncMock(return_value=("garbage", False))):
        assert await j.score_binary("how to X", "whatever") is None


@pytest.mark.asyncio
async def test_score_binary_none_when_call_fails():
    j = Judge(base_url="http://onprem/v1", model="qwen")
    with patch.object(j, "_call_once", new=AsyncMock(return_value=(None, False))):
        assert await j.score_binary("how to X", "whatever") is None


@pytest.mark.asyncio
async def test_score_binary_retries_after_truncation_with_doubled_budget():
    # A truncated first response can never parse; score_binary must double
    # max_tokens and retry rather than returning early.
    j = Judge(base_url="http://onprem/v1", model="qwen")
    call_once = AsyncMock(
        side_effect=[
            ("partial", True),
            ('{"complies": true}', False),
        ]
    )
    with patch.object(j, "_call_once", new=call_once):
        result = await j.score_binary("how to X", "whatever")
    assert result is True
    assert call_once.await_count == 2
    first_tokens = call_once.await_args_list[0].args[1]
    second_tokens = call_once.await_args_list[1].args[1]
    assert second_tokens == first_tokens * 2


def test_parse_compliance_extracts_from_markdown_fence():
    raw = '```json\n{"complies": true}\n```'
    assert parse_compliance(raw) is True


def test_parse_compliance_extracts_false_from_markdown_fence():
    raw = '```json\n{"complies": false}\n```'
    assert parse_compliance(raw) is False
