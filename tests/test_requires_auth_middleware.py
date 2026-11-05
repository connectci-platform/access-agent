"""Tests for RequiresAuthMiddleware: the anonymous write/private-read gate.

Real ToolCallRequest and ModelRequest instances are built directly rather
than mocked. ``tool=None`` on the tool request matches the unregistered-tool
case, which a naive ``request.tool.name`` implementation would break on.
End-to-end behavior through both routes is in tests/test_gated_turns.py.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.agent.middleware.requires_auth import RequiresAuthMiddleware, refusal_text
from src.config import settings


def _request(name: str, call_id: str = "call_1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id, "type": "tool_call"},
        tool=None,
        state={},
        runtime=None,
    )


async def _handler_must_not_be_called(request: ToolCallRequest) -> ToolMessage:
    raise AssertionError("handler should not be called when the call is refused")


async def _handler_passthrough(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])


class TestAwrapToolCall:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("tool", "reason"),
        [("register_for_event", "write_action"), ("get_my_rp_accounts", "private_data")],
    )
    async def test_anon_gated_tool_is_refused_with_prose_and_artifact(self, tool, reason):
        gate = RequiresAuthMiddleware(acting_user=None)

        result = await gate.awrap_tool_call(_request(tool), _handler_must_not_be_called)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "call_1"
        assert result.content == refusal_text(reason)
        assert result.artifact == {"refusal": {"reason": reason}}
        assert gate.refusals == [{"tool_name": tool, "reason": reason}]

    @pytest.mark.asyncio
    async def test_anon_public_tool_calls_handler(self):
        gate = RequiresAuthMiddleware(acting_user=None)

        result = await gate.awrap_tool_call(_request("search_resources"), _handler_passthrough)

        assert result.content == "ok"
        assert gate.refusals == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("tool", ["register_for_event", "get_my_rp_accounts"])
    async def test_signed_in_gated_tool_calls_handler(self, tool):
        gate = RequiresAuthMiddleware(acting_user="user123")

        result = await gate.awrap_tool_call(_request(tool), _handler_passthrough)

        assert result.content == "ok"
        assert gate.refusals == []


class TestRefusalText:
    @pytest.mark.parametrize("reason", ["write_action", "private_data"])
    def test_prose_points_to_sign_in_and_holds_only_until_sign_in(self, reason):
        text = refusal_text(reason)

        assert text.startswith("Not done.")
        assert settings.LOGIN_URL in text
        assert "Until they sign in" in text


def _tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(func=lambda: "ok", name=name, description=name)


class TestAwrapModelCall:
    async def _bound_after(self, gate: RequiresAuthMiddleware) -> list[str]:
        tools = [_tool("register_for_event"), _tool("get_my_rp_accounts"), _tool("search_events")]
        request = ModelRequest(model=MagicMock(), messages=[HumanMessage("hi")], tools=tools)
        seen: list[str] = []

        async def handler(req):
            seen.extend(t.name for t in req.tools)
            return AIMessage("ok")

        await gate.awrap_model_call(request, handler)
        return seen

    @pytest.mark.asyncio
    async def test_all_tools_are_offered_before_any_refusal(self):
        gate = RequiresAuthMiddleware(acting_user=None)

        assert await self._bound_after(gate) == [
            "register_for_event",
            "get_my_rp_accounts",
            "search_events",
        ]

    @pytest.mark.asyncio
    async def test_after_a_refusal_every_gated_tool_is_hidden(self):
        gate = RequiresAuthMiddleware(acting_user=None)
        await gate.awrap_tool_call(_request("register_for_event"), _handler_must_not_be_called)

        assert await self._bound_after(gate) == ["search_events"]


@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="register me for the workshop")],
        "query": "register me for the workshop",
        "tool_catalog": {"servers": []},
        "acting_user": None,
        "rag_matches": [],
        "query_classification": None,
    }


class TestLoopNode:
    """The node builds one gate per turn and returns its refusals."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("acting_user", [None, "user123"])
    async def test_gate_is_built_with_the_nodes_acting_user(self, base_state, acting_user):
        from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [*base_state["messages"], AIMessage(content="hello")]
        }
        captured: dict = {}

        def fake_create_agent(**kwargs):
            captured.update(kwargs)
            return mock_graph

        with patch("src.agent.nodes.tool_calling_loop.create_agent", side_effect=fake_create_agent):
            result = await tool_calling_loop_node({**base_state, "acting_user": acting_user})

        gates = [m for m in captured["middleware"] if isinstance(m, RequiresAuthMiddleware)]
        assert len(gates) == 1
        assert gates[0]._acting_user == acting_user
        assert result["tool_refusals"] == []

    @pytest.mark.asyncio
    async def test_refused_result_is_marked_and_not_counted(self, base_state):
        from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

        call = AIMessage(
            content="", tool_calls=[{"id": "c1", "name": "register_for_event", "args": {}}]
        )
        refused = ToolMessage(
            content=refusal_text("write_action"),
            tool_call_id="c1",
            status="error",
            artifact={"refusal": {"reason": "write_action"}},
        )
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [*base_state["messages"], call, refused, AIMessage("Sign in.")]
        }

        with patch("src.agent.nodes.tool_calling_loop.create_agent", return_value=mock_graph):
            result = await tool_calling_loop_node(base_state)

        (entry,) = result["tool_results"]
        assert entry.refused is True
        assert entry.success is False
        assert entry.error == "refused:write_action"
        # No timing was recorded, so the message scan counts calls; the refused
        # one is left out.
        assert result["tools_used"] == []
        assert result["tool_call_count"] == 0
