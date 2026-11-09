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

from src.agent.middleware.turn_budget import (
    MAX_IDENTICAL_RUNS,
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


async def test_a_served_failure_keeps_its_status():
    budget, handler = _budget(), _Handler(status="error")
    request = _Request("search_access_documents", {"query": "x"})

    results = [await budget.awrap_tool_call(request, handler) for _ in range(3)]

    assert handler.calls == 2
    assert results[2].status == "error"


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
