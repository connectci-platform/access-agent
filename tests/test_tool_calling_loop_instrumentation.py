import asyncio
from types import SimpleNamespace

from src.agent.nodes.tool_calling_loop import _TokenUsageAccumulator


def _resp(total):
    msg = SimpleNamespace(usage_metadata={"total_tokens": total})
    gen = SimpleNamespace(message=msg)
    return SimpleNamespace(generations=[[gen]])


def test_token_accumulator_sums_across_calls():
    acc = _TokenUsageAccumulator()
    asyncio.run(acc.on_llm_end(_resp(30)))
    asyncio.run(acc.on_llm_end(_resp(12)))
    assert acc.total_tokens == 42


def test_token_accumulator_ignores_missing_usage():
    acc = _TokenUsageAccumulator()
    asyncio.run(acc.on_llm_end(SimpleNamespace(generations=[[SimpleNamespace(message=None)]])))
    assert acc.total_tokens == 0
