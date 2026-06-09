import asyncio
from types import SimpleNamespace

from langchain.agents.middleware import SummarizationMiddleware

from src.agent.nodes.tool_calling_loop import (
    _FlaggingSummarizationMiddleware,
    _TokenUsageAccumulator,
)
from src.agent.turn_capture import get_turn_capture, reset_turn_capture


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


def _instance():
    # Bypass the real constructor (it wants a model); we only exercise the
    # override, whose super() call we patch below.
    return _FlaggingSummarizationMiddleware.__new__(_FlaggingSummarizationMiddleware)


def test_marks_summarized_when_base_summarizes(monkeypatch):
    async def fake_abefore(self, state, runtime):
        return {"messages": []}  # non-None == a summary was produced

    monkeypatch.setattr(SummarizationMiddleware, "abefore_model", fake_abefore, raising=True)
    reset_turn_capture()
    asyncio.run(_instance().abefore_model({}, None))
    assert get_turn_capture()["summarized"] is True


def test_does_not_mark_when_base_skips(monkeypatch):
    async def fake_abefore(self, state, runtime):
        return None  # no summarization this turn

    monkeypatch.setattr(SummarizationMiddleware, "abefore_model", fake_abefore, raising=True)
    reset_turn_capture()
    asyncio.run(_instance().abefore_model({}, None))
    assert get_turn_capture()["summarized"] is False
