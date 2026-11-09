"""A read repeated with the same arguments in one turn isn't run a third time (D8-3055).

QA saw five identical search_access_documents calls in one turn, which filled
the context until compaction fired mid-answer.
"""

import json
import time
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.agent.middleware.earlier_results import (
    EARLIER_RESULT_PREFIX,
    EARLIER_TURN_NOTE,
    earlier_results_view,
)
from src.agent.middleware.turn_budget import (
    MAX_IDENTICAL_RUNS,
    REPEAT_FAILED_TEXT,
    REPEATED_CALL_NOTE,
    TurnBudgetMiddleware,
)
from src.agent.nodes.tool_calling_loop import _build_tool_results


class _Request:
    def __init__(self, name: str, args: dict[str, Any], call_id: str = "chatcmpl-tool-0") -> None:
        self.tool_call = {"name": name, "args": args, "id": call_id, "type": "tool_call"}
        self.tool = None


class _Handler:
    def __init__(self, status: str = "success") -> None:
        self.calls = 0
        self.status = status

    async def __call__(self, request: Any) -> ToolMessage:
        self.calls += 1
        return ToolMessage(
            content=json.dumps({"hits": self.calls}),
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"],
            status=self.status,
        )


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


def _budget() -> TurnBudgetMiddleware:
    return TurnBudgetMiddleware(time.monotonic() + 60)


async def test_one_repeat_runs_and_the_next_gets_the_earlier_result():
    budget, handler = _budget(), _Handler()
    request = _Request("search_access_documents", {"query": "pending username"})

    results = [await budget.awrap_tool_call(request, handler) for _ in range(5)]

    assert handler.calls == MAX_IDENTICAL_RUNS == 2
    served = results[2]
    assert served.content.startswith(REPEATED_CALL_NOTE)
    assert served.content.endswith('{"hits": 2}')
    assert served.tool_call_id == "chatcmpl-tool-0"
    assert served.name == "search_access_documents"
    assert served.artifact == {"refusal": {"reason": "repeat_call"}}
    assert all(r.content == served.content for r in results[3:])
    assert budget.repeats_served == 3


async def test_a_call_that_only_failed_is_refused_after_two_runs():
    budget, handler = _budget(), _Handler(status="error")
    request = _Request("search_access_documents", {"query": "x"})

    results = [await budget.awrap_tool_call(request, handler) for _ in range(4)]

    assert handler.calls == 2
    assert budget.repeats_served == 0
    assert results[2].content == REPEAT_FAILED_TEXT
    assert results[2].artifact == {"refusal": {"reason": "repeat_failure"}}


async def test_the_last_successful_result_is_served_not_a_later_failure():
    budget = _budget()
    request = _Request("get_infrastructure_news", {"resource": "bridges2"})
    replies = iter(
        [
            ToolMessage('{"outage": "Full"}', tool_call_id="c", name="get_infrastructure_news"),
            ToolMessage('{"error": "HTTP 502"}', tool_call_id="c", name="get_infrastructure_news"),
        ]
    )

    async def handler(request: Any) -> ToolMessage:
        return next(replies)

    await budget.awrap_tool_call(request, handler)
    await budget.awrap_tool_call(request, handler)
    served = await budget.awrap_tool_call(request, handler)

    assert served.content.endswith('{"outage": "Full"}')
    assert served.status == "success"
    assert budget.repeats_served == 1


async def test_different_arguments_or_tools_still_run():
    budget, handler = _budget(), _Handler()

    for _ in range(3):
        await budget.awrap_tool_call(_Request("search_access_documents", {"query": "a"}), handler)
    await budget.awrap_tool_call(_Request("search_access_documents", {"query": "b"}), handler)
    await budget.awrap_tool_call(_Request("search_software", {"query": "a"}), handler)

    assert handler.calls == 4


async def test_a_change_is_never_served_from_an_earlier_result():
    budget, handler = _budget(), _Handler()
    request = _Request("register_for_event", {"event_id": 7})

    for _ in range(3):
        await budget.awrap_tool_call(request, handler)

    assert handler.calls == 3


async def test_a_new_turn_runs_the_call_again():
    handler = _Handler()
    request = _Request("search_access_documents", {"query": "a"})
    first = _budget()
    for _ in range(3):
        await first.awrap_tool_call(request, handler)

    await _budget().awrap_tool_call(request, handler)

    assert handler.calls == 3


def test_a_served_result_is_recorded_as_not_run():
    call = {"name": "search_access_documents", "args": {"query": "a"}, "id": "c1"}
    served = TurnBudgetMiddleware._served_again(
        call, ToolMessage(content="{}", tool_call_id="c0", name="search_access_documents")
    )
    results, _ = _build_tool_results([AIMessage(content="", tool_calls=[call]), served], [], [])

    assert results[0].refused
    assert results[0].error == "refused:repeat_call"


async def test_a_real_agent_stops_running_a_search_it_keeps_repeating():
    ran: list[str] = []

    @tool
    def search_access_documents(query: str) -> str:
        """Search ACCESS documentation."""
        ran.append(query)
        return json.dumps({"chunks": ["The account is still being set up."]})

    def asks(i: int) -> AIMessage:
        call = {"name": "search_access_documents", "args": {"query": "pending"}, "id": f"c{i}"}
        return AIMessage(content="", tool_calls=[call])

    model = _ScriptedModel(
        messages=iter([*(asks(i) for i in range(5)), AIMessage(content="It's being set up.")])
    )
    agent = create_agent(model=model, tools=[search_access_documents], middleware=[_budget()])

    result = await agent.ainvoke({"messages": [HumanMessage(content="My username is pending")]})

    assert len(ran) == 2
    tool_results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_results) == 5
    assert all(REPEATED_CALL_NOTE in m.content for m in tool_results[2:])
    assert "still being set up" in tool_results[-1].content


def _served(name: str, body: str) -> ToolMessage:
    call = {"name": name, "args": {}, "id": "c1"}
    return TurnBudgetMiddleware._served_again(
        call, ToolMessage(content=body, tool_call_id="c0", name=name)
    )


def test_a_served_live_result_from_an_earlier_turn_is_marked_earlier():
    thread = [
        HumanMessage("is bridges-2 up?"),
        _served("get_infrastructure_news", '{"outage": "Full"}'),
        AIMessage("Bridges-2 is down."),
        HumanMessage("and now?"),
    ]

    seen = earlier_results_view(thread)[1].content

    assert seen.startswith(EARLIER_RESULT_PREFIX)
    assert REPEATED_CALL_NOTE not in seen
    assert seen.endswith('{"outage": "Full"}')


def test_a_served_reference_result_from_an_earlier_turn_loses_only_its_note():
    thread = [
        HumanMessage("pending?"),
        _served("search_access_documents", "New accounts can take a few days."),
        AIMessage("It's being set up."),
        HumanMessage("thanks"),
    ]

    assert earlier_results_view(thread)[1].content == "New accounts can take a few days."


def test_a_served_result_in_the_current_turn_keeps_its_note():
    served = _served("search_access_documents", "x")
    thread = [HumanMessage("pending?"), served]

    assert earlier_results_view(thread)[1].content == served.content


def test_a_served_reference_result_alone_adds_no_earlier_turn_note():
    thread = [
        HumanMessage("pending?"),
        _served("search_access_documents", "New accounts can take a few days."),
        AIMessage("It's being set up."),
        HumanMessage("thanks"),
    ]

    view = earlier_results_view(thread)

    assert view[-1].content == "thanks"
    assert EARLIER_TURN_NOTE not in view[-1].content
