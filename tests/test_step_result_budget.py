"""One step's tool results share a budget in the model's view, so parallel
large results can't overflow the window (D8-3055). The stored results stay whole."""

from typing import Any

from langchain.agents.middleware.types import ModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from src.agent.middleware.turn_budget import TurnBudgetMiddleware, _shares, step_capped_view
from src.agent.nodes.tool_calling_loop import (
    _edit_aware_token_counter,
    _FlaggingSummarizationMiddleware,
)
from src.config import settings

# Measured against the deployment (src/config.py): the window, and the tool
# schemas, system prompt and reply budget every request carries.
WINDOW = 131_072
FIXED_OVERHEAD = 29_028
# Qwen tokenizes tool JSON at about 3 characters a token, the counter assumes 4.
JSON_DENSITY = 4 / 3


def _step(n: int, chars: int, prefix: str) -> list[Any]:
    calls = [
        {"name": "list_all_software", "args": {"resource": f"{prefix}{i}"}, "id": f"{prefix}{i}"}
        for i in range(n)
    ]
    return [
        AIMessage("", id=f"{prefix}-ai", tool_calls=calls),
        *(
            ToolMessage('{"items": "' + "x" * chars + '"}', tool_call_id=c["id"], id=f"{c['id']}-t")
            for c in calls
        ),
    ]


def _real_request_tokens(messages: list[Any]) -> float:
    return count_tokens_approximately(step_capped_view(messages)) * JSON_DENSITY + FIXED_OVERHEAD


def _middleware() -> _FlaggingSummarizationMiddleware:
    return _FlaggingSummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([AIMessage("a summary")] * 50)),
        token_counter=_edit_aware_token_counter(),
        trigger=("tokens", settings.SUMMARIZATION_TRIGGER_TOKENS),
        keep=("tokens", settings.SUMMARIZATION_KEEP_TOKENS),
    )


def _after(messages: list[Any]) -> list[Any]:
    result = _middleware().before_model({"messages": messages}, None)
    return messages if result is None else result["messages"][1:]


def test_a_lone_result_can_use_the_whole_budget():
    assert _shares([500], 300) == [300]
    assert _shares([100, 500, 500], 700) == [100, 300, 300]


def test_three_parallel_large_results_stay_under_the_window():
    thread = [HumanMessage("compare software on Anvil, Delta and Expanse", id="q")]
    thread += _step(3, 200_000, "s")

    after = _after(thread)

    assert _real_request_tokens(after) < WINDOW
    view = step_capped_view(after)
    assert sum(len(m.content) for m in view if isinstance(m, ToolMessage)) < (
        settings.TOOL_STEP_MAX_CHARS + 300
    )
    assert all("[result truncated:" in m.content for m in view if isinstance(m, ToolMessage))


def test_a_large_step_after_another_stays_under_the_window():
    thread = [HumanMessage("compare software everywhere", id="q")]
    thread += _step(1, 150_000, "a") + _step(3, 190_000, "b")

    after = _after(thread)

    assert _real_request_tokens(after) < WINDOW
    assert any(isinstance(m, HumanMessage) for m in after)


def test_the_stored_results_stay_whole():
    thread = [HumanMessage("q", id="q"), *_step(2, 200_000, "s")]

    view = step_capped_view(thread)

    assert view is not thread
    assert all(len(m.content) > 200_000 for m in thread if isinstance(m, ToolMessage))


def test_a_thread_that_fits_is_returned_as_is():
    thread = [HumanMessage("q"), *_step(2, 100, "s"), AIMessage("done")]

    assert step_capped_view(thread) is thread


def test_a_result_without_text_content_is_left_alone():
    blocks = ToolMessage(content=[{"type": "text", "text": "x"}], tool_call_id="c")
    thread = [AIMessage("", tool_calls=[]), blocks, *_step(1, 300_000, "s")[1:]]

    view = step_capped_view(thread)

    assert view[1] is blocks
    assert "[result truncated:" in view[2].content


class _Request:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.tools = ["a_tool"]

    def override(self, **changes: Any) -> "_Request":
        request = _Request(changes.get("messages", self.messages))
        request.tools = changes.get("tools", self.tools)
        return request


async def _model_sees(budget: TurnBudgetMiddleware, messages: list[Any]) -> Any:
    seen: list[Any] = []

    async def handler(request: Any) -> ModelResponse:
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    await budget.awrap_model_call(_Request(messages), handler)
    return seen[0]


async def test_the_model_is_shown_the_cut_results():
    import time

    thread = [HumanMessage("q"), *_step(2, 200_000, "s")]

    seen = await _model_sees(TurnBudgetMiddleware(time.monotonic() + 60), thread)

    assert "[result truncated:" in seen.messages[-1].content
    assert seen.tools == ["a_tool"]


async def test_the_cut_results_are_shown_when_out_of_time_too():
    import time

    thread = [HumanMessage("q"), *_step(2, 200_000, "s")]

    seen = await _model_sees(TurnBudgetMiddleware(time.monotonic() - 1), thread)

    assert "[result truncated:" in seen.messages[-1].content
    assert seen.tools == []
