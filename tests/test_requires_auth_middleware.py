"""Tests for RequiresAuthMiddleware: the gate on tools the caller may not use.

Real ToolCallRequest and ModelRequest instances are built directly rather
than mocked, each carrying the run's AgentContext on its runtime as
create_agent passes it. ``tool=None`` on the tool request matches the
unregistered-tool case, which a naive ``request.tool.name`` implementation
would break on. End-to-end behavior through both routes is in
tests/test_gated_turns.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime

from src.agent.middleware.requires_auth import (
    RequiresAuthMiddleware,
    refusal_artifact,
    refusal_text,
    signed_in_since,
)
from src.config import settings
from tests.agent_runtime import context_for, run_loop

if TYPE_CHECKING:
    from src.agent.context import AgentContext

ANON = context_for(None)
READ_ONLY_TOKEN = context_for("user123", can_write=False)
WRITE_TOKEN = context_for("user123", can_write=True)


def _request(name: str, context: object = ANON, call_id: str = "call_1") -> ToolCallRequest:
    return ToolCallRequest(
        tool_call={"name": name, "args": {}, "id": call_id, "type": "tool_call"},
        tool=None,
        state={},
        runtime=Runtime(context=context),
    )


async def _handler_must_not_be_called(request: ToolCallRequest) -> ToolMessage:
    raise AssertionError("handler should not be called when the call is refused")


async def _handler_passthrough(request: ToolCallRequest) -> ToolMessage:
    return ToolMessage(content="ok", tool_call_id=request.tool_call["id"])


class TestAwrapToolCall:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("context", "tool", "reason"),
        [
            (ANON, "register_for_event", "write_action"),
            (ANON, "get_my_rp_accounts", "private_data"),
            (READ_ONLY_TOKEN, "register_for_event", "write_scope"),
        ],
        ids=["anon-write", "anon-private-read", "read-only-token-write"],
    )
    async def test_disallowed_call_is_refused_with_prose_and_artifact(self, context, tool, reason):
        gate = RequiresAuthMiddleware()

        result = await gate.awrap_tool_call(_request(tool, context), _handler_must_not_be_called)

        assert isinstance(result, ToolMessage)
        assert result.status == "error"
        assert result.tool_call_id == "call_1"
        assert result.content == refusal_text(reason)
        assert result.artifact == {"refusal": {"reason": reason}}
        assert gate.refusals == [{"tool_name": tool, "reason": reason}]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("context", "tool"),
        [
            (ANON, "search_resources"),
            (READ_ONLY_TOKEN, "get_my_rp_accounts"),
            (READ_ONLY_TOKEN, "search_resources"),
            (WRITE_TOKEN, "register_for_event"),
            (WRITE_TOKEN, "get_my_rp_accounts"),
        ],
        ids=[
            "anon-public",
            "read-only-token-private-read",
            "read-only-token-public",
            "write-token-write",
            "write-token-private-read",
        ],
    )
    async def test_allowed_call_reaches_the_tool(self, context, tool):
        gate = RequiresAuthMiddleware()

        result = await gate.awrap_tool_call(_request(tool, context), _handler_passthrough)

        assert result.content == "ok"
        assert gate.refusals == []

    @pytest.mark.asyncio
    async def test_read_only_deployment_refuses_writes_for_a_write_token(self, monkeypatch):
        monkeypatch.setattr(settings, "READ_ONLY", True)
        gate = RequiresAuthMiddleware()

        result = await gate.awrap_tool_call(
            _request("register_for_event", WRITE_TOKEN), _handler_must_not_be_called
        )

        assert result.artifact == {"refusal": {"reason": "write_scope"}}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("context", [None, {"acting_user": "user123"}, "user123"])
    async def test_a_call_without_an_agent_context_fails_closed(self, context):
        gate = RequiresAuthMiddleware()

        with pytest.raises(TypeError):
            await gate.awrap_tool_call(
                _request("search_resources", context), _handler_must_not_be_called
            )


def test_a_tool_with_no_name_is_never_gated():
    from src.agent.middleware.requires_auth import refusal_reason

    assert refusal_reason(ANON, None) is None
    assert refusal_reason(ANON, "") is None


class TestRefusalText:
    @pytest.mark.parametrize("reason", ["write_action", "private_data"])
    def test_prose_points_to_sign_in_and_holds_only_until_sign_in(self, reason):
        text = refusal_text(reason)

        assert text.startswith("Not done.")
        assert settings.LOGIN_URL in text
        assert "Until they sign in" in text

    def test_write_scope_prose_offers_no_sign_in_and_points_to_a_ticket(self):
        text = refusal_text("write_scope")

        assert text.startswith("Not done.")
        assert settings.LOGIN_URL not in text
        assert "can't make changes on their behalf" in text
        assert "https://support.access-ci.org/open-a-ticket" in text


def _tool(name: str) -> StructuredTool:
    return StructuredTool.from_function(func=lambda: "ok", name=name, description=name)


class TestAwrapModelCall:
    async def _bound_after(self, gate: RequiresAuthMiddleware, context: AgentContext) -> list[str]:
        tools = [_tool("register_for_event"), _tool("get_my_rp_accounts"), _tool("search_events")]
        request = ModelRequest(
            model=MagicMock(),
            messages=[HumanMessage("hi")],
            tools=tools,
            runtime=Runtime(context=context),
        )
        seen: list[str] = []

        async def handler(req):
            seen.extend(t.name for t in req.tools)
            return AIMessage("ok")

        await gate.awrap_model_call(request, handler)
        return seen

    @pytest.mark.asyncio
    async def test_all_tools_are_offered_before_any_refusal(self):
        gate = RequiresAuthMiddleware()

        assert await self._bound_after(gate, ANON) == [
            "register_for_event",
            "get_my_rp_accounts",
            "search_events",
        ]

    @pytest.mark.asyncio
    async def test_after_an_anonymous_refusal_every_gated_tool_is_hidden(self):
        gate = RequiresAuthMiddleware()
        await gate.awrap_tool_call(_request("register_for_event"), _handler_must_not_be_called)

        assert await self._bound_after(gate, ANON) == ["search_events"]

    @pytest.mark.asyncio
    async def test_after_a_write_scope_refusal_only_write_tools_are_hidden(self):
        gate = RequiresAuthMiddleware()
        await gate.awrap_tool_call(
            _request("register_for_event", READ_ONLY_TOKEN), _handler_must_not_be_called
        )

        assert await self._bound_after(gate, READ_ONLY_TOKEN) == [
            "get_my_rp_accounts",
            "search_events",
        ]


@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="register me for the workshop")],
        "query": "register me for the workshop",
        "tool_catalog": {"servers": []},
        "rag_matches": [],
        "query_classification": None,
    }


class TestLoopNode:
    """The node reads the caller from its runtime and returns the gate's refusals."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("context", [None, {"acting_user": "user123"}, "user123"])
    async def test_a_run_without_an_agent_context_fails_closed(self, base_state, context):
        from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

        with pytest.raises(TypeError):
            await tool_calling_loop_node(base_state, Runtime(context=context))

    @pytest.mark.asyncio
    async def test_one_gate_is_built_per_turn(self, base_state):
        mock_graph = AsyncMock()
        mock_graph.ainvoke.return_value = {
            "messages": [*base_state["messages"], AIMessage(content="hello")]
        }
        captured: dict = {}

        def fake_create_agent(**kwargs):
            captured.update(kwargs)
            return mock_graph

        with patch("src.agent.nodes.tool_calling_loop.create_agent", side_effect=fake_create_agent):
            result = await run_loop(base_state, "user123")

        gates = [m for m in captured["middleware"] if isinstance(m, RequiresAuthMiddleware)]
        assert len(gates) == 1
        assert result["tool_refusals"] == []

    @pytest.mark.asyncio
    async def test_refused_result_is_marked_and_not_counted(self, base_state):
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
            result = await run_loop(base_state)

        (entry,) = result["tool_results"]
        assert entry.refused is True
        assert entry.success is False
        assert entry.error == "refused:write_action"
        # No timing was recorded, so the message scan counts calls; the refused
        # one is left out.
        assert result["tools_used"] == []
        assert result["tool_call_count"] == 0


def _refused(reason: str = "write_action") -> ToolMessage:
    return ToolMessage(
        content=refusal_text(reason),
        tool_call_id="c1",
        status="error",
        artifact=refusal_artifact(reason),
    )


def _after_sign_in(messages: list, context: object = WRITE_TOKEN) -> list:
    return signed_in_since(messages, context)


def test_a_refusal_from_before_sign_in_no_longer_tells_the_model_to_ask_for_it():
    messages = _after_sign_in(
        [HumanMessage(content="Sign me up."), _refused(), HumanMessage(content="Sign me up.")]
    )

    assert "has signed in since" in messages[1].content
    assert "sign in with their ACCESS account" not in messages[1].content


def test_the_latest_request_carries_a_note_that_the_user_is_signed_in():
    messages = _after_sign_in([_refused(), HumanMessage(content="Sign me up.")])

    assert messages[1].content.startswith("Sign me up.")
    assert "is signed in now" in messages[1].content


def test_an_earlier_request_is_not_changed():
    first = HumanMessage(content="Sign me up.")

    messages = _after_sign_in([first, _refused(), HumanMessage(content="Again.")])

    assert messages[0].content == "Sign me up."


def test_the_stored_messages_are_not_changed():
    refused = _refused()
    latest = HumanMessage(content="Again.")

    _after_sign_in([refused, latest])

    assert refused.content == refusal_text("write_action")
    assert latest.content == "Again."


def test_an_anonymous_caller_keeps_the_refusal():
    messages = [_refused(), HumanMessage(content="Again.")]

    assert _after_sign_in(messages, ANON) == messages


def test_a_write_scope_refusal_is_kept():
    messages = [_refused("write_scope"), HumanMessage(content="Again.")]

    assert _after_sign_in(messages, READ_ONLY_TOKEN) == messages


def test_a_write_refusal_stands_for_a_caller_who_still_cannot_write():
    messages = [_refused("write_action"), HumanMessage(content="Again.")]

    assert _after_sign_in(messages, READ_ONLY_TOKEN) == messages


def test_a_private_data_refusal_clears_for_a_read_only_caller():
    messages = _after_sign_in(
        [_refused("private_data"), HumanMessage(content="Again.")], READ_ONLY_TOKEN
    )

    assert "has signed in since" in messages[0].content


def test_a_conversation_with_no_refusal_is_unchanged():
    messages = [HumanMessage(content="Hello."), ToolMessage(content="{}", tool_call_id="c1")]

    assert _after_sign_in(messages) == messages


def test_a_refusal_after_the_latest_request_adds_no_note():
    latest = HumanMessage(content="Again.")

    messages = _after_sign_in([latest, _refused()])

    assert messages[0].content == "Again."


def test_a_request_with_block_content_is_left_alone():
    latest = HumanMessage(content=[{"type": "text", "text": "Again."}])

    messages = _after_sign_in([_refused(), latest])

    assert messages[1].content == latest.content


async def test_the_model_call_gets_the_rewritten_messages():
    seen: list = []

    async def handler(request):
        seen.extend(request.messages)
        return MagicMock()

    request = ModelRequest(
        model=MagicMock(),
        messages=[_refused(), HumanMessage(content="Again.")],
        tools=[],
        runtime=Runtime(context=WRITE_TOKEN),
    )

    await RequiresAuthMiddleware().awrap_model_call(request, handler)

    assert "is signed in now" in seen[1].content
