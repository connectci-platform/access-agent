from unittest.mock import AsyncMock, patch

import pytest

from src.eval.judge import Judge


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
