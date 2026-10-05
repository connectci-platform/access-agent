"""Tests for RequiresAuthMiddleware: anon write/private-read tool-call gate.

Three mechanisms under test, each verified against the real installed API
(no mocks of langgraph/langchain internals):

1. awrap_tool_call suppression — anon + write/auth-read tool returns a
   sentinel ToolMessage WITHOUT calling handler; anon + unauth-read and
   authed + write both call handler normally.
2. find_requires_auth_reason — the post-invoke message-scan helper the loop
   node uses, detects the sentinel and extracts its coarse reason.
3. tool_calling_loop_node threading — with create_agent mocked (same pattern
   as test_tool_calling_loop.py), a result_messages list containing the
   sentinel surfaces `requires_auth` in the node's return dict; a clean
   result does not.

A real ToolCallRequest is constructed directly (not mocked) per the brief's
preference for real instances over mocks where feasible; `tool=None` matches
the unregistered-tool case the brief specifically flagged as a risk for a
naive `request.tool.name` implementation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest

from src.agent.middleware.requires_auth import (
    REQUIRES_AUTH_SENTINEL,
    RequiresAuthMiddleware,
    find_requires_auth_reason,
)


def _request(name: str, call_id: str = "call_1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id, "type": "tool_call"},
        tool=None,  # unregistered-tool case; must not touch request.tool.name
        state={},
        runtime=None,
    )


async def _handler_must_not_be_called(request: ToolCallRequest) -> ToolMessage:
    raise AssertionError("handler should not be called when the call is suppressed")


async def _handler_passthrough(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])


class TestAwrapToolCall:
    @pytest.mark.asyncio
    async def test_anon_write_tool_is_suppressed_with_sentinel(self):
        middleware = RequiresAuthMiddleware(acting_user=None)
        result = await middleware.awrap_tool_call(
            _request("register_for_event"), _handler_must_not_be_called
        )
        assert isinstance(result, ToolMessage)
        assert result.content.startswith(REQUIRES_AUTH_SENTINEL)
        assert result.status == "error"
        assert result.tool_call_id == "call_1"

    @pytest.mark.asyncio
    async def test_anon_auth_read_tool_is_suppressed_with_sentinel(self):
        middleware = RequiresAuthMiddleware(acting_user=None)
        result = await middleware.awrap_tool_call(
            _request("get_my_rp_accounts"), _handler_must_not_be_called
        )
        assert isinstance(result, ToolMessage)
        assert result.content.startswith(REQUIRES_AUTH_SENTINEL)

    @pytest.mark.asyncio
    async def test_anon_write_tool_reason_is_write_action(self):
        middleware = RequiresAuthMiddleware(acting_user=None)
        result = await middleware.awrap_tool_call(
            _request("register_for_event"), _handler_must_not_be_called
        )
        assert find_requires_auth_reason([result]) == "write_action"

    @pytest.mark.asyncio
    async def test_anon_auth_read_tool_reason_is_private_data(self):
        middleware = RequiresAuthMiddleware(acting_user=None)
        result = await middleware.awrap_tool_call(
            _request("get_rp_account"), _handler_must_not_be_called
        )
        assert find_requires_auth_reason([result]) == "private_data"

    @pytest.mark.asyncio
    async def test_anon_unauth_read_tool_calls_handler(self):
        """An anon turn using only read tools is unaffected."""
        middleware = RequiresAuthMiddleware(acting_user=None)
        result = await middleware.awrap_tool_call(
            _request("search_resources"), _handler_passthrough
        )
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_authed_write_tool_calls_handler(self):
        """With acting_user set, the same write call executes normally."""
        middleware = RequiresAuthMiddleware(acting_user="user123")
        result = await middleware.awrap_tool_call(
            _request("register_for_event"), _handler_passthrough
        )
        assert result.content == "ok"

    @pytest.mark.asyncio
    async def test_authed_auth_read_tool_calls_handler(self):
        middleware = RequiresAuthMiddleware(acting_user="user123")
        result = await middleware.awrap_tool_call(
            _request("get_my_rp_accounts"), _handler_passthrough
        )
        assert result.content == "ok"


class TestFindRequiresAuthReason:
    def test_no_sentinel_returns_none(self):
        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
            ToolMessage(content="42", tool_call_id="c1"),
        ]
        assert find_requires_auth_reason(messages) is None

    def test_sentinel_found_returns_reason(self):
        messages = [
            HumanMessage(content="register me"),
            ToolMessage(
                content=f"{REQUIRES_AUTH_SENTINEL}:write_action",
                tool_call_id="c1",
                status="error",
            ),
        ]
        assert find_requires_auth_reason(messages) == "write_action"

    def test_first_sentinel_wins_on_multiple(self):
        messages = [
            ToolMessage(
                content=f"{REQUIRES_AUTH_SENTINEL}:write_action",
                tool_call_id="c1",
                status="error",
            ),
            ToolMessage(
                content=f"{REQUIRES_AUTH_SENTINEL}:private_data",
                tool_call_id="c2",
                status="error",
            ),
        ]
        assert find_requires_auth_reason(messages) == "write_action"

    def test_non_tool_message_with_similar_content_ignored(self):
        """Only a real ToolMessage can carry the signal — an AIMessage that
        happens to mention the sentinel text must not trigger a false positive."""
        messages = [AIMessage(content=f"{REQUIRES_AUTH_SENTINEL}:write_action")]
        assert find_requires_auth_reason(messages) is None


@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="register me for the workshop")],
        "query": "register me for the workshop",
        "tool_catalog": {
            "servers": [
                {
                    "server": "events",
                    "tools": [
                        {
                            "name": "register_for_event",
                            "description": "Register for an event",
                            "inputSchema": {"properties": {}, "required": []},
                        }
                    ],
                }
            ]
        },
        "acting_user": None,
        "rag_matches": [],
        "query_classification": None,
    }


class TestLoopNodeThreading:
    """tool_calling_loop_node surfaces requires_auth in its return dict when
    the sentinel appears in the post-invoke message scan. create_agent is
    mocked (same boundary test_tool_calling_loop.py mocks) so this exercises
    the node's own scan/threading logic, not the real react loop.
    """

    @pytest.mark.asyncio
    async def test_sentinel_in_result_messages_surfaces_requires_auth(self, base_state):
        from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "register_for_event", "args": {}}],
        )
        sentinel_msg = ToolMessage(
            content=f"{REQUIRES_AUTH_SENTINEL}:write_action",
            tool_call_id="call_1",
            status="error",
        )
        final_msg = AIMessage(content="You'll need to sign in to register for this event.")

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [*base_state["messages"], tool_call_msg, sentinel_msg, final_msg]
        }

        with patch("src.agent.nodes.tool_calling_loop.create_agent", return_value=mock_graph):
            result = await tool_calling_loop_node(base_state)

        assert result["requires_auth"] == {
            "login_url": _expected_login_url(),
            "reason": "write_action",
        }

    @pytest.mark.asyncio
    async def test_authed_turn_has_no_requires_auth_key(self, base_state):
        """With acting_user set, the same turn executes normally — no sentinel,
        no requires_auth key in the return dict."""
        from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

        state = {**base_state, "acting_user": "user123"}
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "register_for_event", "args": {}}],
        )
        tool_result_msg = ToolMessage(content='{"status": "registered"}', tool_call_id="call_1")
        final_msg = AIMessage(content="You're registered for the workshop.")

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [*state["messages"], tool_call_msg, tool_result_msg, final_msg]
        }

        with patch("src.agent.nodes.tool_calling_loop.create_agent", return_value=mock_graph):
            result = await tool_calling_loop_node(state)

        assert "requires_auth" not in result

    @pytest.mark.asyncio
    async def test_anon_read_only_turn_has_no_requires_auth_key(self, base_state):
        """An anon turn using only read tools is unaffected."""
        from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

        state = {
            **base_state,
            "messages": [HumanMessage(content="what resources have GPUs?")],
            "tool_catalog": {
                "servers": [
                    {
                        "server": "compute-resources",
                        "tools": [
                            {
                                "name": "search_resources",
                                "description": "Search for compute resources",
                                "inputSchema": {"properties": {}, "required": []},
                            }
                        ],
                    }
                ]
            },
        }
        tool_call_msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "search_resources", "args": {}}],
        )
        tool_result_msg = ToolMessage(content='{"resources": []}', tool_call_id="call_1")
        final_msg = AIMessage(content="No GPU resources matched.")

        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [*state["messages"], tool_call_msg, tool_result_msg, final_msg]
        }

        with patch("src.agent.nodes.tool_calling_loop.create_agent", return_value=mock_graph):
            result = await tool_calling_loop_node(state)

        assert "requires_auth" not in result

    @pytest.mark.asyncio
    async def test_requires_auth_middleware_is_constructed_with_acting_user_closure(
        self, base_state
    ):
        """The middleware list passed to create_agent must include a
        RequiresAuthMiddleware built with the node's acting_user (closure
        capture), not read from inner agent state."""
        from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

        final_msg = AIMessage(content="hello")
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {"messages": [*base_state["messages"], final_msg]}

        captured_kwargs: dict = {}

        def fake_create_agent(**kwargs):
            captured_kwargs.update(kwargs)
            return mock_graph

        with patch("src.agent.nodes.tool_calling_loop.create_agent", side_effect=fake_create_agent):
            await tool_calling_loop_node({**base_state, "acting_user": "user123"})

        gate_middlewares = [
            m for m in captured_kwargs["middleware"] if isinstance(m, RequiresAuthMiddleware)
        ]
        assert len(gate_middlewares) == 1
        assert gate_middlewares[0]._acting_user == "user123"


def _expected_login_url() -> str:
    from src.config import settings

    return settings.LOGIN_URL
