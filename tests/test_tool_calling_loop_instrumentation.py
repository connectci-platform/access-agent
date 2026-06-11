import asyncio
from types import SimpleNamespace

from langchain.agents.middleware import SummarizationMiddleware

from src.agent.nodes.tool_calling_loop import (
    _FlaggingSummarizationMiddleware,
    _TokenUsageAccumulator,
)
from src.agent.state import AgentState, create_initial_state
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


def test_sync_before_model_marks_summarized(monkeypatch):
    def fake_before(self, state, runtime):
        return {"messages": []}

    monkeypatch.setattr(SummarizationMiddleware, "before_model", fake_before, raising=True)
    reset_turn_capture()
    _instance().before_model({}, None)
    assert get_turn_capture()["summarized"] is True


def test_total_tokens_is_a_declared_state_channel():
    # Must be a declared channel or LangGraph strips it from the updates stream
    # and final_state never carries it (the loop's return value is silently dropped).
    assert "total_tokens" in AgentState.__annotations__


def test_create_initial_state_sets_total_tokens():
    state = create_initial_state(
        query="q",
        session_id="s",
        question_id="qid",
        tool_catalog={},
    )
    assert state["total_tokens"] is None
