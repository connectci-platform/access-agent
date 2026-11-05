"""A tool server's readOnlyHint: read from the catalog, and it can only add WRITE.

access-mcp publishes each tool's write-ness as ``annotations.readOnlyHint``.
``readOnlyHint: false`` makes a tool a write whatever its name; ``true`` never
turns a write name into a read; no hint leaves today's name rules in charge.
``toggle_alerts`` is an invented write whose name the rules miss.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

from src.agent.auth_tools import ToolIdentityClass, tool_identity_class
from src.agent.domains.tools import MCPToolWrapper, create_mcp_tools_from_catalog
from src.agent.middleware.requires_auth import RequiresAuthMiddleware
from src.agent.nodes import tool_calling_loop as loop_mod
from src.tools.mcp_client import MCPClient
from src.tools.registry import CatalogAggregator
from src.turn_reporter import _assemble_turn_report
from tests.agent_runtime import context_for

READ_ONLY_TOKEN = context_for("u@x", can_write=False)
HIDDEN_WRITE = "toggle_alerts"


@pytest.mark.parametrize(
    ("name", "read_only", "expected"),
    [
        (HIDDEN_WRITE, False, ToolIdentityClass.WRITE),
        (HIDDEN_WRITE, None, ToolIdentityClass.NONE),
        (HIDDEN_WRITE, True, ToolIdentityClass.NONE),
        ("create_event", True, ToolIdentityClass.WRITE),
        ("create_event", None, ToolIdentityClass.WRITE),
        ("get_my_rp_accounts", True, ToolIdentityClass.AUTH_READ),
        ("search_events", True, ToolIdentityClass.NONE),
    ],
    ids=[
        "hint-false-makes-a-write",
        "no-hint-uses-names",
        "hint-true-on-a-read-name",
        "hint-true-never-removes-a-write",
        "no-hint-write-name",
        "hint-true-private-read",
        "hint-true-public-read",
    ],
)
def test_the_hint_can_only_add_write(name, read_only, expected):
    assert tool_identity_class(name, read_only) is expected


@pytest.mark.parametrize(
    ("annotations", "expected"),
    [
        ({"readOnlyHint": False}, False),
        ({"readOnlyHint": True}, True),
        (None, None),
        ({}, None),
        ({"readOnlyHint": "false"}, None),
        ([], None),
        ("readOnly", None),
    ],
    ids=["false", "true", "null", "empty", "not-a-bool", "a-list", "a-string"],
)
def test_the_catalog_keeps_a_boolean_hint_and_nothing_else(annotations, expected):
    tool = {"name": "t", "description": "", "inputSchema": {}, "annotations": annotations}

    assert CatalogAggregator(server_urls={})._process_tool(tool)["read_only"] is expected


def test_wrappers_carry_the_hint_from_the_catalog():
    catalog = {"servers": [{"server": "s", "tools": [{"name": HIDDEN_WRITE, "read_only": False}]}]}

    (wrapper,) = create_mcp_tools_from_catalog(catalog, "u@x")

    assert wrapper.read_only is False


def _call(tool: object) -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": HIDDEN_WRITE, "args": {}, "id": "c1", "type": "tool_call"},
        tool=tool,
        state={},
        runtime=Runtime(context=READ_ONLY_TOKEN),
    )


def _wrapper(read_only: bool | None) -> MCPToolWrapper:
    return MCPToolWrapper(
        name=HIDDEN_WRITE,
        description="t",
        tool_server="s",
        mcp_client=MCPClient(),
        acting_user="u@x",
        read_only=read_only,
    )


async def _runs(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ran", tool_call_id="c1")


@pytest.mark.asyncio
async def test_the_gate_refuses_a_hinted_write_for_a_read_only_token():
    gate = RequiresAuthMiddleware()

    result = await gate.awrap_tool_call(_call(_wrapper(read_only=False)), _runs)

    assert result.artifact == {"refusal": {"reason": "write_scope"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", [None, "unhinted"], ids=["unknown-tool", "no-hint"])
async def test_without_a_hint_the_gate_uses_names(tool):
    gate = RequiresAuthMiddleware()
    request = _call(None if tool is None else _wrapper(read_only=None))

    result = await gate.awrap_tool_call(request, _runs)

    assert result.content == "ran"


@pytest.mark.asyncio
async def test_a_read_only_hint_never_lets_a_write_name_through():
    gate = RequiresAuthMiddleware()
    request = _call(_wrapper(read_only=True))
    request.tool_call["name"] = "create_event"
    tool = _wrapper(read_only=True)
    tool.name = "create_event"

    gated = await gate.awrap_tool_call(request, _runs)
    _content, artifact = await tool._arun()

    assert gated.artifact == {"refusal": {"reason": "write_scope"}}
    assert artifact == {"refusal": {"reason": "write_scope"}}


@pytest.mark.asyncio
async def test_after_a_refusal_hinted_writes_are_hidden_too():
    gate = RequiresAuthMiddleware()
    await gate.awrap_tool_call(_call(_wrapper(read_only=False)), _runs)
    tools = [
        _wrapper(read_only=False),
        {"name": "dict_write", "read_only": False},
        {"name": "search_events", "read_only": True},
    ]
    request = ModelRequest(
        model=MagicMock(),
        messages=[HumanMessage("hi")],
        tools=tools,
        runtime=Runtime(context=READ_ONLY_TOKEN),
    )
    seen: list[str] = []

    async def handler(req):
        seen.extend(t["name"] if isinstance(t, dict) else t.name for t in req.tools)
        return AIMessage("ok")

    await gate.awrap_model_call(request, handler)

    assert seen == ["search_events"]


@pytest.mark.asyncio
async def test_the_wrapper_refuses_a_hinted_write_without_permission(monkeypatch):
    tool = _wrapper(read_only=False)
    call_tool = AsyncMock()
    monkeypatch.setattr(tool.mcp_client, "call_tool", call_tool)

    _content, artifact = await tool._arun()

    call_tool.assert_not_called()
    assert artifact == {"refusal": {"reason": "write_scope"}}


def test_read_only_strips_hinted_writes_but_not_prefix_matches(monkeypatch):
    monkeypatch.setattr(loop_mod.settings, "READ_ONLY", True)
    tools = [_wrapper(read_only=False), _wrapper(read_only=None), _wrapper(read_only=True)]
    tools[1].name = "create_widget"  # matches the prefix rule only
    tools[2].name = "search_events"

    kept = [t.name for t in loop_mod._apply_read_only_filter(tools)]

    assert kept == ["create_widget", "search_events"]


def test_invoked_write_counts_a_hinted_write():
    final_state = {
        "tool_results": [{"tool_name": HIDDEN_WRITE, "server": "s", "success": True}],
        "tool_catalog": {
            "servers": [{"server": "s", "tools": [{"name": HIDDEN_WRITE, "read_only": False}]}]
        },
    }

    report, _ = _assemble_turn_report(
        final_state=final_state,
        session_id="s",
        turn_index=1,
        question_id="q",
        query_text="q",
        duration_ms=1.0,
        acting_user="u@x",
        success=True,
        capabilities=[],
    )

    assert report["invoked_write"] is True


@pytest.mark.parametrize(
    "catalog",
    [
        None,
        "bad",
        {"servers": None},
        {"servers": ["bad", {"tools": None}, {"tools": [{"name": "x", "read_only": True}]}]},
    ],
)
def test_invoked_write_is_false_without_a_hinted_write(catalog):
    final_state = {
        "tool_results": [{"tool_name": "x", "server": "s", "success": True}],
        "tool_catalog": catalog,
    }

    report, _ = _assemble_turn_report(
        final_state=final_state,
        session_id="s",
        turn_index=1,
        question_id="q",
        query_text="q",
        duration_ms=1.0,
        acting_user="u@x",
        success=True,
        capabilities=[],
    )

    assert report["invoked_write"] is False
