"""Every model request fits the window, counted the way Qwen counts (D8-3055).

One step's results share a budget, a turn too big for the window keeps what
fits, and the last line, applied to every model call including the out-of-time
one, cuts tool results oldest first until the request fits. The stored results
stay whole. The samples are real tool results whose token counts were measured
against the deployment (src/agent/token_estimate.py).
"""

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool

from src.agent.middleware.turn_budget import (
    ANSWER_NOW_NOTE,
    TurnBudgetMiddleware,
    _shares,
    fitted_view,
    step_capped_view,
)
from src.agent.nodes.tool_calling_loop import (
    _edit_aware_token_counter,
    _FlaggingSummarizationMiddleware,
)
from src.agent.prompts.system_prompt import build_system_prompt
from src.agent.token_estimate import estimate_messages, estimate_text, estimate_tools
from src.config import settings

FIXTURES = Path(__file__).parent / "fixtures" / "tool_results"
# Real Qwen counts, from vLLM's prompt_tokens (2026-11-09).
XDMOD_REAL_TOKENS = 3_911


def _sample(name: str) -> str:
    return json.loads((FIXTURES / name).read_text())["content"][0]["text"]


ANVIL = _sample("list_all_software_anvil.json")
XDMOD = _sample("get_chart_data_wait_time.json")


def _step(results: list[str], prefix: str) -> list[Any]:
    calls = [
        {"name": "get_chart_data", "args": {"n": i}, "id": f"{prefix}{i}"}
        for i in range(len(results))
    ]
    return [
        AIMessage("", id=f"{prefix}-ai", tool_calls=calls),
        *(
            ToolMessage(text, tool_call_id=c["id"], id=f"{c['id']}-t")
            for c, text in zip(calls, results, strict=True)
        ),
    ]


def _middleware(model: Any = None, **kwargs: Any) -> _FlaggingSummarizationMiddleware:
    return _FlaggingSummarizationMiddleware(
        model=model or GenericFakeChatModel(messages=iter([AIMessage("a summary")] * 50)),
        token_counter=_edit_aware_token_counter(),
        trigger=("tokens", settings.SUMMARIZATION_TRIGGER_TOKENS),
        keep=("tokens", settings.SUMMARIZATION_KEEP_TOKENS),
        **kwargs,
    )


def test_the_approximate_counter_undercounts_chart_text():
    # Why the window is counted with token_estimate (tests/test_token_estimate.py).
    assert count_tokens_approximately([HumanMessage(XDMOD)]) < XDMOD_REAL_TOKENS / 2
    assert estimate_text(XDMOD) >= XDMOD_REAL_TOKENS


def test_a_lone_result_can_use_the_whole_budget():
    assert _shares([500], 300) == [300]
    assert _shares([100, 500, 500], 700) == [100, 300, 300]


def test_three_parallel_software_lists_share_the_step_budget():
    thread = [HumanMessage("compare software", id="q"), *_step([ANVIL] * 3, "s")]

    view = step_capped_view(thread)

    results = [m for m in view if isinstance(m, ToolMessage)]
    assert sum(estimate_text(m.content) for m in results) <= settings.TOOL_STEP_MAX_TOKENS
    assert all("[result truncated:" in m.content for m in results)
    assert all(m.content == ANVIL for m in thread if isinstance(m, ToolMessage))


def test_two_software_lists_in_one_step_are_shown_whole():
    thread = [HumanMessage("compare software", id="q"), *_step([ANVIL] * 2, "s")]

    assert step_capped_view(thread) is thread


def test_a_result_without_text_content_is_left_alone():
    blocks = ToolMessage(content=[{"type": "text", "text": "x"}], tool_call_id="c")
    thread = [AIMessage("", tool_calls=[]), blocks, *_step([XDMOD * 60], "s")[1:]]

    view = step_capped_view(thread)

    assert view[1] is blocks
    assert "[result truncated:" in view[2].content


def test_the_window_guard_cuts_oldest_first_and_keeps_every_message():
    thread = [HumanMessage("q", id="q"), *_step([XDMOD * 10], "a"), *_step([XDMOD * 10], "b")]
    limit = estimate_messages(thread) - 20_000

    view = fitted_view(thread, limit)

    assert estimate_messages(view) <= limit
    assert [m.id for m in view] == [m.id for m in thread]
    assert [getattr(m, "tool_call_id", None) for m in view] == [
        getattr(m, "tool_call_id", None) for m in thread
    ]
    assert "[result truncated:" in view[2].content
    assert view[4] is thread[4]


def test_the_window_guard_leaves_a_request_that_fits():
    thread = [HumanMessage("q"), *_step([XDMOD], "a")]

    assert fitted_view(thread, 1_000_000) is thread


def test_the_window_guard_does_what_it_can_when_nothing_would_fit(caplog):
    blocks = ToolMessage(content=[{"type": "text", "text": "x"}], tool_call_id="b")
    thread = [HumanMessage("q" * 10_000), AIMessage("", tool_calls=[]), blocks]
    thread += _step([XDMOD], "a")[1:]

    view = fitted_view(thread, 10)

    assert view[2] is blocks
    assert view[3].content.startswith("\n\n[result truncated: 0 of")
    assert "over the window" in caplog.text


def test_a_turn_over_the_ceiling_keeps_as_many_steps_as_fit():
    thread = [HumanMessage("compare four resources", id="q")]
    for prefix in "abcd":
        thread += _step([ANVIL], prefix)

    result = _middleware().before_model({"messages": thread}, None)

    assert result is not None
    kept = {m.id for m in result["messages"][1:]}
    assert {"b0-t", "c0-t", "d0-t"} <= kept
    assert "a0-t" not in kept


def test_three_software_lists_in_a_turn_are_kept_whole():
    thread = [HumanMessage("compare three resources", id="q")]
    for prefix in "abc":
        thread += _step([ANVIL], prefix)

    assert _middleware().before_model({"messages": thread}, None) is None


def test_with_no_step_under_the_ceiling_the_last_one_is_kept_for_the_guard():
    last = _step([XDMOD], "b")
    last[0].content = "Y" * 400_000
    thread = [HumanMessage("q", id="q"), *_step([ANVIL], "a"), *last]

    result = _middleware().before_model({"messages": thread}, None)

    assert result is not None
    assert [m.id for m in result["messages"][-2:]] == ["b-ai", "b0-t"]
    assert "a0-t" not in {m.id for m in result["messages"]}


def test_a_question_over_the_ceiling_falls_back_to_upstream_s_cutoff():
    thread = [HumanMessage("Anvil?", id="q-0"), *_step([ANVIL], "a")]
    thread.append(HumanMessage("Y" * 400_000, id="q-1"))

    result = _middleware().before_model({"messages": thread}, None)

    assert result is not None
    assert result["messages"][-1].id == "q-1"


def test_a_reported_total_over_the_trigger_does_not_compact():
    """The reported total includes ~29,000 of schemas and system prompt, so it
    fired far below the conversation size the trigger is sized for."""
    model = GenericFakeChatModel(messages=iter([AIMessage("a summary")] * 5))
    usage = {"input_tokens": 99_000, "output_tokens": 1_000, "total_tokens": 100_000}
    answered = AIMessage("Y" * 120_000, id="earlier-answer")
    current = [HumanMessage("and Delta?", id="q-1"), *_step([XDMOD], "now")]
    current[1].usage_metadata = usage  # type: ignore[attr-defined]
    current[1].response_metadata = {"model_provider": "genericfakechatmodel"}
    thread = [HumanMessage("Anvil?", id="q-0"), answered, *current]
    assert estimate_messages(thread) < settings.SUMMARIZATION_TRIGGER_TOKENS

    assert _middleware(model).before_model({"messages": thread}, None) is None


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


class _Requests(AgentMiddleware):
    """Listed innermost, so it records each request exactly as the model gets it."""

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[Any] = []

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self.seen.append(request)
        return await handler(request)


class _OutOfTimeAfter(TurnBudgetMiddleware):
    """Out of time once ``calls`` model requests have gone out."""

    def __init__(self, requests: _Requests, calls: int) -> None:
        super().__init__(tool_deadline=float("inf"))
        self._requests = requests
        self._calls = calls

    def out_of_time(self) -> bool:
        return len(self._requests.seen) >= self._calls


def _estimate(request: Any) -> int:
    system = request.system_message.text if request.system_message else ""
    return (
        estimate_text(system)
        + estimate_tools(request.tools)
        + estimate_messages(request.messages)
        + settings.MAX_TOKENS_LOOP
    )


async def test_three_chart_steps_then_the_deadline_never_overflow_the_window():
    """The out-of-time call skips compaction, so only the guard bounds it."""
    long_chart = XDMOD * 45  # a long daily series, ~170k real tokens

    @tool
    def get_chart_data(n: int) -> str:
        """Get chart data."""
        return long_chart

    def asks(i: int) -> AIMessage:
        call = {"name": "get_chart_data", "args": {"n": i}, "id": f"c{i}"}
        return AIMessage("", tool_calls=[call])

    requests = _Requests()
    budget = _OutOfTimeAfter(requests, calls=3)
    agent = create_agent(
        model=_ScriptedModel(messages=iter([asks(0), asks(1), asks(2), AIMessage("Done.")])),
        tools=[get_chart_data],
        system_prompt=build_system_prompt(acting_user=None),
        middleware=[_middleware(out_of_time=budget.out_of_time), budget, requests],
    )

    result = await agent.ainvoke({"messages": [HumanMessage("Daily wait times, everywhere?")]})

    assert len(requests.seen) == 4
    assert requests.seen[-1].tools == []
    assert ANSWER_NOW_NOTE in requests.seen[-1].messages[-1].content
    assert all(_estimate(r) <= settings.MODEL_CONTEXT_TOKENS for r in requests.seen)
    # Not vacuous: sent whole, the last request would be far over the window.
    assert estimate_messages(result["messages"]) > settings.MODEL_CONTEXT_TOKENS
    assert all(m.content == long_chart for m in result["messages"] if isinstance(m, ToolMessage))
