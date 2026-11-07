"""A turn answers from what it has before the time limit, instead of losing it all."""

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from src.agent.context import AgentContext
from src.agent.domains.tools import MCPToolWrapper, _build_args_schema
from src.agent.graph import create_agent_graph
from src.agent.middleware.turn_budget import (
    ANSWER_NOW_NOTE,
    INCOMPLETE_NOTICE,
    OUT_OF_TIME_ANSWER,
    TIMED_OUT_ARTIFACT,
    TurnBudgetMiddleware,
)
from src.agent.nodes.tool_calling_loop import (
    _build_tool_results,
    _FlaggingSummarizationMiddleware,
)
from src.agent.tools.access_documents import search_access_documents
from src.agent.turn_capture import (
    get_turn_capture,
    record_turn_deadline,
    reset_turn_capture,
)
from src.config import settings
from src.tools.mcp_client import MCPClient, MCPToolResult
from tests.agent_runtime import run_loop

# vLLM reuses ids like this one across a thread.
CALL_ID = "chatcmpl-tool-0"


def _later(seconds: float = 60.0) -> float:
    return time.monotonic() + seconds


def _call(args: dict[str, Any], name: str = "get_chart_data", call_id: str = CALL_ID) -> Any:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _asks(*calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=list(calls))


class _ToolRequest:
    def __init__(self, tool_call: dict[str, Any], registered: Any = None) -> None:
        self.tool_call = tool_call
        self.tool = registered


class _ModelRequest:
    def __init__(self, messages: list[Any], tools: list[Any]) -> None:
        self.messages = messages
        self.tools = tools

    def override(self, **changes: Any) -> "_ModelRequest":
        return _ModelRequest(
            changes.get("messages", self.messages), changes.get("tools", self.tools)
        )


def _timed_out(request: Any) -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"error": "Timeout calling xdmod/get_chart_data"}),
        tool_call_id=request.tool_call["id"],
        artifact=TIMED_OUT_ARTIFACT,
    )


class _Tool:
    """A tool handler that counts its calls."""

    def __init__(self, result: Any = None) -> None:
        self.calls = 0
        self._result = result

    async def __call__(self, request: Any) -> ToolMessage:
        self.calls += 1
        if self._result:
            return self._result(request)
        return ToolMessage(content="{}", tool_call_id=request.tool_call["id"])


async def _model_sees(middleware: TurnBudgetMiddleware, messages: list[Any]) -> Any:
    seen: list[Any] = []

    async def handler(request: Any) -> ModelResponse:
        seen.append(request)
        return ModelResponse(result=[AIMessage(content="ok")])

    await middleware.awrap_model_call(_ModelRequest(messages, ["a_tool"]), handler)
    return seen[0]


async def test_a_call_before_the_deadline_runs():
    handler = _Tool()

    result = await TurnBudgetMiddleware(_later()).awrap_tool_call(
        _ToolRequest(_call({"realm": "Jobs"})), handler
    )

    assert handler.calls == 1
    assert result.status == "success"


async def test_a_call_after_the_deadline_never_runs_and_says_why():
    handler = _Tool()

    result = await TurnBudgetMiddleware(_later(-1)).awrap_tool_call(
        _ToolRequest(_call({"realm": "Jobs"})), handler
    )

    assert handler.calls == 0
    assert result.status == "error"
    assert result.tool_call_id == CALL_ID
    assert result.name == "get_chart_data"
    assert "Not run" in result.content
    assert result.artifact == {"refusal": {"reason": "turn_deadline"}}


async def test_a_running_tool_is_cut_off_at_the_deadline():
    middleware = TurnBudgetMiddleware(_later(0.02))
    cancelled = asyncio.Event()

    async def hangs(request: Any) -> ToolMessage:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("never finishes")  # pragma: no cover

    result = await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "SUPREMM"})), hangs)

    assert cancelled.is_set()
    assert result.status == "error"
    assert "no result" in result.content
    # It ran, so it isn't a refusal, and the turn's tool count includes it.
    assert result.artifact == TIMED_OUT_ARTIFACT


async def test_a_change_already_running_is_left_to_finish():
    middleware = TurnBudgetMiddleware(_later(0.02))
    finished = ToolMessage(content="{}", tool_call_id=CALL_ID, name="register_for_event")

    async def slow_write(request: Any) -> ToolMessage:
        await asyncio.sleep(0.06)
        return finished

    call = {"name": "register_for_event", "args": {}, "id": CALL_ID, "type": "tool_call"}
    result = await middleware.awrap_tool_call(_ToolRequest(call), slow_write)

    assert result is finished


async def test_a_change_that_timed_out_is_not_repeated():
    middleware = TurnBudgetMiddleware(_later())
    timed_out = ToolMessage(
        content="{}",
        tool_call_id=CALL_ID,
        name="register_for_event",
        artifact=TIMED_OUT_ARTIFACT,
    )
    calls: list[Any] = []

    async def write(request: Any) -> ToolMessage:
        calls.append(request)
        return timed_out

    call = {"name": "register_for_event", "args": {}, "id": CALL_ID, "type": "tool_call"}
    await middleware.awrap_tool_call(_ToolRequest(call), write)
    result = await middleware.awrap_tool_call(_ToolRequest(call), write)

    assert len(calls) == 1
    assert "Not run" in result.content


def test_the_answer_note_warns_against_saying_there_are_none():
    assert "never say something doesn't exist" in ANSWER_NOW_NOTE


def test_the_answer_note_says_nothing_more_can_happen():
    assert "This is your final message" in ANSWER_NOW_NOTE


async def test_the_same_call_after_a_timeout_is_blocked():
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_timed_out)

    await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "SUPREMM"})), handler)
    repeat = await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "SUPREMM"})), handler)

    assert handler.calls == 1
    assert repeat.status == "error"
    assert "timed out" in repeat.content
    assert repeat.artifact == {"refusal": {"reason": "repeat_timeout"}}


async def test_the_block_does_not_depend_on_call_ids():
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_timed_out)
    args = {"realm": "SUPREMM"}

    await middleware.awrap_tool_call(_ToolRequest(_call(args, call_id="a")), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call(args, call_id="b")), handler)

    assert handler.calls == 1


async def test_a_reused_call_id_with_new_arguments_still_runs():
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_timed_out)

    await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "SUPREMM"})), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "Jobs"})), handler)

    assert handler.calls == 2


async def test_another_tool_with_the_same_arguments_still_runs():
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_timed_out)
    args = {"realm": "SUPREMM"}

    await middleware.awrap_tool_call(_ToolRequest(_call(args)), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call(args, "get_raw_data")), handler)

    assert handler.calls == 2


async def test_a_call_that_succeeded_can_repeat():
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool()

    await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "Jobs"})), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "Jobs"})), handler)

    assert handler.calls == 2


async def test_a_call_that_failed_for_another_reason_can_repeat():
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(
        lambda request: ToolMessage(
            content=json.dumps({"error": "HTTP 500"}),
            tool_call_id=request.tool_call["id"],
            artifact=None,
        )
    )

    await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "Jobs"})), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call({"realm": "Jobs"})), handler)

    assert handler.calls == 2


async def test_a_command_result_passes_through():
    command = object()

    async def handler(request: Any) -> Any:
        return command

    result = await TurnBudgetMiddleware(_later()).awrap_tool_call(
        _ToolRequest(_call({"realm": "Jobs"})), handler
    )

    assert result is command


PARAMETERS = [
    {"name": "realm", "type": "string", "required": True},
    {"name": "show_title", "type": "boolean"},
    {"name": "limit", "type": "integer"},
    {"name": "filters", "type": "object"},
]
SCHEMA_TOOL = type("_Registered", (), {"args_schema": _build_args_schema("chart", PARAMETERS)})()


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"realm": "SUPREMM", "show_title": True}, {"realm": "SUPREMM", "show_title": "true"}),
        ({"realm": "SUPREMM", "limit": 5}, {"realm": "SUPREMM", "limit": "5"}),
        (
            {"realm": "SUPREMM", "filters": {"a": 1, "b": 2}},
            {"realm": "SUPREMM", "filters": '{"b": 2, "a": 1}'},
        ),
        ({"realm": "SUPREMM"}, {"realm": "SUPREMM", "limit": None}),
        ({"realm": "SUPREMM", "limit": 5}, {"limit": 5, "realm": "SUPREMM"}),
    ],
)
async def test_the_same_arguments_in_another_form_are_the_same_call(first, second):
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_timed_out)

    await middleware.awrap_tool_call(_ToolRequest(_call(first), SCHEMA_TOOL), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call(second), SCHEMA_TOOL), handler)

    assert handler.calls == 1


async def test_arguments_the_tool_would_reject_are_compared_as_sent():
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_timed_out)
    bad = {"limit": "many"}

    await middleware.awrap_tool_call(_ToolRequest(_call(bad), SCHEMA_TOOL), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call(bad), SCHEMA_TOOL), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call({"limit": "few"}), SCHEMA_TOOL), handler)

    assert handler.calls == 2


async def test_a_tool_with_a_plain_dict_schema_is_compared_as_sent():
    registered = type("_Registered", (), {"args_schema": {"type": "object"}})()
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_timed_out)

    await middleware.awrap_tool_call(_ToolRequest(_call({"a": 1}), registered), handler)
    await middleware.awrap_tool_call(_ToolRequest(_call({"a": 1}), registered), handler)

    assert handler.calls == 1


async def test_before_the_deadline_the_model_call_is_unchanged():
    middleware = TurnBudgetMiddleware(_later())
    messages = [HumanMessage(content="usage?")]

    seen = await _model_sees(middleware, messages)

    assert seen.tools == ["a_tool"]
    assert seen.messages == messages
    assert not middleware.deadline_hit


async def test_after_the_deadline_the_model_gets_no_tools_and_is_told_to_answer():
    middleware = TurnBudgetMiddleware(_later(-1))
    result = ToolMessage(content='{"rows": 3}', tool_call_id=CALL_ID)
    messages = [HumanMessage(content="usage?"), _asks(_call({})), result]

    seen = await _model_sees(middleware, messages)

    assert seen.tools == []
    # A tool result outweighs prompt text for this model, so the note rides on one.
    assert seen.messages[-1].content == f'{{"rows": 3}}\n\n{ANSWER_NOW_NOTE}'
    assert middleware.deadline_hit
    assert result.content == '{"rows": 3}'


async def test_with_no_tool_result_to_carry_it_the_note_is_its_own_message():
    messages = [HumanMessage(content="usage?")]

    seen = await _model_sees(TurnBudgetMiddleware(_later(-1)), messages)

    assert seen.messages[0].content == "usage?"
    assert isinstance(seen.messages[-1], HumanMessage)
    assert seen.messages[-1].content == ANSWER_NOW_NOTE


def test_a_call_that_never_ran_counts_as_refused_in_the_turn_record():
    ai = _asks(_call({"realm": "Jobs"}))
    blocked = ToolMessage(
        content="Not run.",
        name="get_chart_data",
        tool_call_id=CALL_ID,
        status="error",
        artifact={"refusal": {"reason": "turn_deadline"}},
    )

    results, _ = _build_tool_results([ai, blocked], [], [])

    assert results[0].refused
    assert results[0].error == "refused:turn_deadline"


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


class _SeesModelCalls(AgentMiddleware):
    """Listed last, so it records exactly what reaches the model."""

    def __init__(self) -> None:
        super().__init__()
        self.tools: list[list[Any]] = []
        self.messages: list[list[Any]] = []

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        self.tools.append(list(request.tools))
        self.messages.append(list(request.messages))
        return await handler(request)


class _SlowModel(AgentMiddleware):
    """Every model call takes ``seconds``, as the budget sees it."""

    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = seconds

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        await asyncio.sleep(self.seconds)
        return await handler(request)


async def test_a_real_agent_answers_when_a_tool_hangs_past_the_soft_deadline():
    ran: list[str] = []

    @tool
    async def get_chart_data(realm: str) -> str:
        """Get chart data."""
        ran.append(realm)
        if realm == "SUPREMM":
            await asyncio.Event().wait()
        return json.dumps({"realm": realm, "rows": 3})

    model = _ScriptedModel(
        messages=iter(
            [
                _asks(_call({"realm": "Jobs"})),
                _asks(_call({"realm": "SUPREMM"})),
                AIMessage(content="Jobs shows 3 rows. SUPREMM data did not come back."),
            ]
        )
    )
    sees = _SeesModelCalls()
    agent = create_agent(
        model=model,
        tools=[get_chart_data],
        middleware=[TurnBudgetMiddleware(_later(0.3)), sees],
    )

    result = await asyncio.wait_for(
        agent.ainvoke({"messages": [HumanMessage(content="Usage for AI projects?")]}), 5
    )

    assert ran == ["Jobs", "SUPREMM"]
    assert [len(tools) for tools in sees.tools] == [1, 1, 0]
    assert ANSWER_NOW_NOTE in sees.messages[-1][-1].content
    assert any('"rows": 3' in str(m.content) for m in sees.messages[-1])
    assert result["messages"][-1].content.startswith("Jobs shows 3 rows")


async def test_a_real_agent_refuses_a_call_that_arrives_after_the_deadline():
    ran: list[str] = []

    @tool
    def get_chart_data(realm: str) -> str:
        """Get chart data."""
        ran.append(realm)
        return "{}"

    model = _ScriptedModel(
        messages=iter([_asks(_call({"realm": "Jobs"})), AIMessage(content="Out of time.")])
    )
    agent = create_agent(
        model=model,
        tools=[get_chart_data],
        middleware=[TurnBudgetMiddleware(_later(0.05)), _SlowModel(0.1)],
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Usage?")]})

    assert ran == []
    assert result["messages"][-1].content == f"Out of time.\n\n{INCOMPLETE_NOTICE}"


def _mcp_tool(client: MCPClient) -> MCPToolWrapper:
    return MCPToolWrapper(
        name="get_chart_data",
        description="Get chart data.",
        tool_server="xdmod",
        mcp_client=client,
        args_schema=_build_args_schema("get_chart_data", PARAMETERS),
    )


async def test_a_real_agent_stops_a_timed_out_mcp_call_repeating_in_another_form(monkeypatch):
    client = MCPClient()
    call_tool = AsyncMock(
        return_value=MCPToolResult(
            success=False, error="Timeout calling xdmod/get_chart_data", timed_out=True
        )
    )
    monkeypatch.setattr(client, "call_tool", call_tool)
    model = _ScriptedModel(
        messages=iter(
            [
                _asks(_call({"realm": "SUPREMM", "show_title": True, "filters": {"a": 1}})),
                _asks(_call({"realm": "SUPREMM", "show_title": "true", "filters": '{"a": 1}'})),
                AIMessage(content="SUPREMM data timed out."),
            ]
        )
    )
    agent = create_agent(
        model=model, tools=[_mcp_tool(client)], middleware=[TurnBudgetMiddleware(_later())]
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Usage?")]})

    assert call_tool.await_count == 1
    last_tool_result = [m for m in result["messages"] if isinstance(m, ToolMessage)][-1]
    assert "timed out" in last_tool_result.content


async def test_a_new_turn_can_try_a_call_the_last_turn_timed_out():
    handler = _Tool(_timed_out)
    args = {"realm": "SUPREMM"}

    await TurnBudgetMiddleware(_later()).awrap_tool_call(_ToolRequest(_call(args)), handler)
    await TurnBudgetMiddleware(_later()).awrap_tool_call(_ToolRequest(_call(args)), handler)

    assert handler.calls == 2


def _answering_graph() -> AsyncMock:
    graph = AsyncMock()
    graph.ainvoke.return_value = {"messages": [AIMessage(content="ok")]}
    return graph


async def _budget_for_a_turn() -> TurnBudgetMiddleware:
    with patch(
        "src.agent.nodes.tool_calling_loop.create_agent", return_value=_answering_graph()
    ) as build:
        await run_loop({"messages": [HumanMessage(content="hi")], "tool_catalog": {}})
    (budget,) = [
        m for m in build.call_args.kwargs["middleware"] if isinstance(m, TurnBudgetMiddleware)
    ]
    return budget


async def test_the_loop_stops_tools_the_answer_reserve_before_the_routes_turn_limit():
    reset_turn_capture()
    limit = _later(100)
    record_turn_deadline(limit)

    budget = await _budget_for_a_turn()

    assert budget.tool_deadline == limit - settings.AGENT_ANSWER_RESERVE_S


async def test_with_no_route_limit_the_turn_is_timed_from_the_loop():
    reset_turn_capture()
    before = time.monotonic()

    budget = await _budget_for_a_turn()

    expected = settings.AGENT_TURN_TIMEOUT_S - settings.AGENT_ANSWER_RESERVE_S
    assert before + expected <= budget.tool_deadline <= time.monotonic() + expected


def test_arguments_the_tool_would_drop_do_not_make_a_new_call():
    class _Args(BaseModel):
        realm: str

    registered = type("_Registered", (), {"args_schema": _Args})()

    assert TurnBudgetMiddleware._call_key(
        _call({"realm": "Jobs", "ignored": 1}), registered
    ) == TurnBudgetMiddleware._call_key(_call({"realm": "Jobs"}), registered)


async def test_the_loop_answers_with_what_it_has_when_an_xdmod_call_hangs(monkeypatch):
    async def call_tool(self: Any, server: str, tool_name: str, arguments: Any, **_: Any) -> Any:
        if arguments["realm"] == "SUPREMM":
            await asyncio.Event().wait()
        return MCPToolResult(success=True, data={"rows": 3})

    monkeypatch.setattr(MCPClient, "call_tool", call_tool)
    reset_turn_capture()
    record_turn_deadline(_later(settings.AGENT_ANSWER_RESERVE_S + 0.3))
    model = _ScriptedModel(
        messages=iter(
            [
                _asks(_call({"realm": "Jobs"})),
                _asks(_call({"realm": "SUPREMM"})),
                AIMessage(content="Jobs shows 3 rows. SUPREMM data did not come back."),
            ]
        )
    )
    catalog = {
        "servers": [
            {
                "server": "xdmod",
                "tools": [
                    {
                        "name": "get_chart_data",
                        "description": "Get chart data.",
                        "parameters": PARAMETERS,
                    }
                ],
            }
        ]
    }

    with patch("src.agent.nodes.tool_calling_loop.get_llm", return_value=model):
        result = await asyncio.wait_for(
            create_agent_graph().ainvoke(
                {"messages": [HumanMessage(content="Usage?")], "tool_catalog": catalog},
                context=AgentContext.anonymous(),
            ),
            5,
        )

    assert result["final_answer"].startswith("Jobs shows 3 rows")
    assert [r.success for r in result["tool_results"]] == [True, False]
    assert result["tool_call_count"] == 2


async def test_a_slow_model_loses_its_tools_while_it_can_still_answer_in_time():
    ran: list[str] = []

    @tool
    def get_chart_data(realm: str) -> str:
        """Get chart data."""
        ran.append(realm)
        return "{}"

    model = _ScriptedModel(
        messages=iter(
            [
                _asks(_call({"realm": "Jobs"})),
                AIMessage(content="Jobs shows 3 rows."),
            ]
        )
    )
    sees = _SeesModelCalls()
    budget = TurnBudgetMiddleware(_later(0.3))
    agent = create_agent(
        model=model, tools=[get_chart_data], middleware=[budget, sees, _SlowModel(0.2)]
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Usage?")]})

    # The second call starts before the deadline but would end after it.
    assert time.monotonic() < budget.tool_deadline + 0.2
    assert [len(tools) for tools in sees.tools] == [1, 0]
    assert ran == ["Jobs"]
    assert result["messages"][-1].content == f"Jobs shows 3 rows.\n\n{INCOMPLETE_NOTICE}"


async def test_a_fast_model_keeps_its_tools_until_the_deadline():
    middleware = TurnBudgetMiddleware(_later(0.5))
    messages = [HumanMessage(content="usage?")]

    async def handler(request: Any) -> ModelResponse:
        await asyncio.sleep(0.01)
        return ModelResponse(result=[AIMessage(content="ok")])

    await middleware.awrap_model_call(_ModelRequest(messages, ["a_tool"]), handler)

    assert (await _model_sees(middleware, messages)).tools == ["a_tool"]


def _summarizing(model: Any, budget: TurnBudgetMiddleware) -> _FlaggingSummarizationMiddleware:
    return _FlaggingSummarizationMiddleware(
        model=model, trigger=("messages", 2), keep=("messages", 1), out_of_time=budget.out_of_time
    )


async def test_a_real_agent_skips_summarizing_once_out_of_time():
    @tool
    async def get_chart_data(realm: str) -> str:
        """Get chart data."""
        await asyncio.sleep(0.15)
        return "{}"

    # Asked to summarize, this model has nothing to give and the turn would raise.
    no_summaries = _ScriptedModel(messages=iter([]))
    model = _ScriptedModel(
        messages=iter([_asks(_call({"realm": "Jobs"})), AIMessage(content="Answered.")])
    )
    budget = TurnBudgetMiddleware(_later(0.1))
    agent = create_agent(
        model=model,
        tools=[get_chart_data],
        middleware=[_summarizing(no_summaries, budget), budget],
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Usage?")]})

    assert result["messages"][-1].content == f"Answered.\n\n{INCOMPLETE_NOTICE}"


def test_summarizing_is_skipped_on_the_sync_path_too():
    no_summaries = _ScriptedModel(messages=iter([]))
    summarizing = _summarizing(no_summaries, TurnBudgetMiddleware(_later(-1)))
    messages = [HumanMessage(content="a"), AIMessage(content="b"), HumanMessage(content="c")]

    assert summarizing.before_model({"messages": messages}, None) is None


def _write_tool(client: MCPClient, name: str = "register_for_event", **fields: Any) -> Any:
    return MCPToolWrapper(
        name=name, description="A change.", tool_server="events", mcp_client=client, **fields
    )


def _agent_with_write(client: MCPClient, deadline: float, **fields: Any) -> Any:
    model = _ScriptedModel(
        messages=iter(
            [
                _asks(_call({}, "register_for_event")),
                AIMessage(content="Done with that."),
            ]
        )
    )
    return create_agent(
        model=model,
        tools=[_write_tool(client, **fields)],
        middleware=[TurnBudgetMiddleware(deadline)],
    )


async def test_a_real_agent_starts_no_write_that_could_outlast_the_tool_deadline(monkeypatch):
    client = MCPClient(timeout=1.0)
    call_tool = AsyncMock(return_value=MCPToolResult(success=True, data={}))
    monkeypatch.setattr(client, "call_tool", call_tool)

    result = await _agent_with_write(client, _later(0.5)).ainvoke(
        {"messages": [HumanMessage(content="Sign me up.")]}
    )

    assert call_tool.await_count == 0
    refused = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert refused.artifact == {"refusal": {"reason": "turn_deadline"}}
    assert "Not run" in refused.content


async def test_a_real_agent_starts_a_write_whose_timeout_fits(monkeypatch):
    client = MCPClient(timeout=0.1)
    call_tool = AsyncMock(return_value=MCPToolResult(success=True, data={}))
    monkeypatch.setattr(client, "call_tool", call_tool)

    await _agent_with_write(client, _later(0.5), can_write=True).ainvoke(
        {"messages": [HumanMessage(content="Sign me up.")]}
    )

    assert call_tool.await_count == 1


async def test_a_write_is_judged_by_the_timeout_its_server_would_really_get(monkeypatch):
    monkeypatch.setattr("src.tools.mcp_client.settings.XDMOD_TOOL_TIMEOUT_S", 5.0)
    client = MCPClient(timeout=0.1)
    registered = MCPToolWrapper(
        name="create_chart", description="d", tool_server="xdmod", mcp_client=client
    )
    handler = _Tool()
    call = _call({}, "create_chart")

    result = await TurnBudgetMiddleware(_later(1)).awrap_tool_call(
        _ToolRequest(call, registered), handler
    )

    assert handler.calls == 0
    assert result.status == "error"


async def test_a_read_may_still_start_inside_the_window_closed_to_writes(monkeypatch):
    client = MCPClient(timeout=5.0)
    registered = MCPToolWrapper(
        name="search_events", description="d", tool_server="events", mcp_client=client
    )
    handler = _Tool()

    await TurnBudgetMiddleware(_later(1)).awrap_tool_call(
        _ToolRequest(_call({}, "search_events"), registered), handler
    )

    assert handler.calls == 1


async def _final_message(response: AIMessage, budget: TurnBudgetMiddleware | None = None) -> Any:
    async def handler(request: Any) -> ModelResponse:
        return ModelResponse(result=[response])

    middleware = budget or TurnBudgetMiddleware(_later(-1))
    request = _ModelRequest([HumanMessage(content="usage?")], ["a_tool"])
    result = await middleware.awrap_model_call(request, handler)
    return result.result[0]


async def test_a_real_agent_ends_the_turn_when_the_model_asks_for_tools_it_lost():
    ran: list[str] = []

    @tool
    def get_chart_data(realm: str) -> str:
        """Get chart data."""
        ran.append(realm)
        return "{}"

    model = _ScriptedModel(
        messages=iter([AIMessage(content="Jobs shows 3 rows.", tool_calls=[_call({"realm": "x"})])])
    )
    agent = create_agent(
        model=model, tools=[get_chart_data], middleware=[TurnBudgetMiddleware(_later(-1))]
    )

    result = await agent.ainvoke({"messages": [HumanMessage(content="Usage?")]})

    assert ran == []
    assert result["messages"][-1].content == f"Jobs shows 3 rows.\n\n{INCOMPLETE_NOTICE}"
    assert result["messages"][-1].tool_calls == []


async def test_the_loop_says_it_ran_out_of_time_when_the_model_gives_only_tool_calls():
    reset_turn_capture()
    record_turn_deadline(_later(settings.AGENT_ANSWER_RESERVE_S - 1))
    model = _ScriptedModel(messages=iter([_asks(_call({"realm": "Jobs"}))]))

    with patch("src.agent.nodes.tool_calling_loop.get_llm", return_value=model):
        result = await asyncio.wait_for(
            create_agent_graph().ainvoke(
                {"messages": [HumanMessage(content="Usage?")], "tool_catalog": {}},
                context=AgentContext.anonymous(),
            ),
            5,
        )

    assert result["final_answer"] == OUT_OF_TIME_ANSWER
    assert result["messages"][-1].content == OUT_OF_TIME_ANSWER
    assert result["answer_unavailable"] is True


async def test_raw_tool_call_text_is_removed_from_the_last_answer():
    text = 'Jobs shows 3 rows.\n<tool_call>\n{"name": "x"}\n</tool_call>'

    message = await _final_message(AIMessage(content=text, id="m1"))

    assert message.content == f"Jobs shows 3 rows.\n\n{INCOMPLETE_NOTICE}"
    assert message.id == "m1"


async def test_an_unfinished_raw_tool_call_is_removed_too():
    message = await _final_message(AIMessage(content='Found 3.<tool_call>{"name"'))

    assert message.content == f"Found 3.\n\n{INCOMPLETE_NOTICE}"


async def test_a_response_of_only_raw_tool_call_text_becomes_the_fallback():
    budget = TurnBudgetMiddleware(_later(-1))

    message = await _final_message(AIMessage(content="<tool_call>{}</tool_call>"), budget)

    assert message.content == OUT_OF_TIME_ANSWER
    assert budget.gave_up


async def test_no_trace_of_a_tool_call_stays_on_the_last_answer():
    # The next turn's request would replay it as a call with no result.
    raw = {"tool_calls": [{"id": "x"}], "refusal": None}
    response = AIMessage(
        content="Found 3.",
        additional_kwargs=raw,
        tool_calls=[_call({})],
        invalid_tool_calls=[{"name": "x", "args": "{", "id": "y", "type": "invalid_tool_call"}],
    )

    message = await _final_message(response)

    assert message.tool_calls == []
    assert message.invalid_tool_calls == []
    assert message.additional_kwargs == {"refusal": None}


async def test_a_response_while_tools_are_available_is_left_alone():
    budget = TurnBudgetMiddleware(_later())
    response = AIMessage(content="<tool_call>{}</tool_call>", tool_calls=[_call({})])

    message = await _final_message(response, budget)

    assert message is response
    assert not budget.gave_up


async def test_a_tool_result_alongside_the_model_response_is_kept():
    other = ToolMessage(content="{}", tool_call_id="x")

    async def handler(request: Any) -> ModelResponse:
        return ModelResponse(result=[AIMessage(content="ok"), other])

    request = _ModelRequest([HumanMessage(content="usage?")], [])
    response = await TurnBudgetMiddleware(_later(-1)).awrap_model_call(request, handler)

    assert response.result[1] is other


def _write_timed_out(request: Any) -> ToolMessage:
    return ToolMessage(
        content=json.dumps({"error": "Timeout calling events/register_for_event"}),
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        artifact=TIMED_OUT_ARTIFACT,
    )


def _says_unknown(text: str) -> bool:
    return "may or may not have" in text and "check" in text and "failed" not in text


async def test_a_write_that_timed_out_reports_an_unknown_outcome():
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_write_timed_out)
    call = _call({}, "register_for_event")

    result = await middleware.awrap_tool_call(_ToolRequest(call), handler)
    repeat = await middleware.awrap_tool_call(_ToolRequest(call), handler)

    assert _says_unknown(result.content)
    assert result.status == "error"
    assert result.artifact == TIMED_OUT_ARTIFACT
    assert handler.calls == 1
    assert _says_unknown(repeat.content)
    assert "couldn't be retrieved" not in repeat.content


async def test_a_write_known_only_by_its_hint_reports_an_unknown_outcome():
    registered = _write_tool(MCPClient(timeout=0.1), "set_preference", read_only=False)
    middleware = TurnBudgetMiddleware(_later())
    handler = _Tool(_write_timed_out)
    call = _call({}, "set_preference")

    result = await middleware.awrap_tool_call(_ToolRequest(call, registered), handler)
    repeat = await middleware.awrap_tool_call(_ToolRequest(call, registered), handler)

    assert _says_unknown(result.content)
    assert _says_unknown(repeat.content)


async def test_a_repeat_of_a_timed_out_write_after_the_deadline_stays_unknown():
    middleware = TurnBudgetMiddleware(_later(0.02))
    handler = _Tool(_write_timed_out)
    call = _call({}, "register_for_event")

    await middleware.awrap_tool_call(_ToolRequest(call), handler)
    await asyncio.sleep(0.03)
    repeat = await middleware.awrap_tool_call(_ToolRequest(call), handler)

    assert _says_unknown(repeat.content)


async def test_an_unset_field_and_a_null_match_when_the_field_has_a_default():
    parameters = [{"name": "limit", "type": "integer", "default": 10}]
    registered = type("_R", (), {"args_schema": _build_args_schema("t", parameters)})()

    assert TurnBudgetMiddleware._call_key(
        _call({"limit": None}), registered
    ) == TurnBudgetMiddleware._call_key(_call({}), registered)


async def test_a_cut_off_document_search_records_one_timing(monkeypatch):
    async def hangs(*args: Any) -> str:
        await asyncio.Event().wait()
        raise AssertionError("never finishes")  # pragma: no cover

    monkeypatch.setattr("src.agent.tools.access_documents._search_access_documents_inner", hangs)
    reset_turn_capture()
    call = _call({"query": "globus"}, "search_access_documents")
    model = _ScriptedModel(messages=iter([_asks(call), AIMessage(content="Answered.")]))
    agent = create_agent(
        model=model,
        tools=[search_access_documents],
        middleware=[TurnBudgetMiddleware(_later(0.05))],
    )

    await agent.ainvoke({"messages": [HumanMessage(content="Globus?")]})

    assert len(get_turn_capture()["tool_timings"]) == 1


def test_a_refused_call_does_not_take_a_siblings_duration():
    ai = _asks(_call({"a": 1}, call_id="c1"), _call({"a": 2}, call_id="c2"))
    blocked = ToolMessage(
        content="Not run.",
        tool_call_id="c1",
        status="error",
        artifact={"refusal": {"reason": "repeat_timeout"}},
    )
    ran = ToolMessage(content="{}", tool_call_id="c2")
    timings = [{"tool_name": "get_chart_data", "duration_ms": 900}]

    results, _ = _build_tool_results([ai, blocked, ran], [], timings)

    assert [r.duration_ms for r in results] == [0, 900]


def test_an_answer_at_the_deadline_says_it_may_be_incomplete():
    middleware = TurnBudgetMiddleware(_later(-1))

    message = middleware._answer_only(AIMessage(content="Found one event. Let me check more."))

    assert message.content.startswith("Found one event.")
    assert message.content.endswith(INCOMPLETE_NOTICE)
