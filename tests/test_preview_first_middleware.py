"""A registration only goes through after the user has answered its preview."""

import json
from types import SimpleNamespace
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from src.agent.middleware.preview_first import PreviewFirstMiddleware

PREVIEW = json.dumps({"action": "register", "status": "preview", "executed": False})


def _call(args: dict[str, Any], call_id: str = "c9") -> dict[str, Any]:
    return {"name": "register_for_event", "args": args, "id": call_id, "type": "tool_call"}


def _previewed(event: str = "9372") -> list[Any]:
    return [
        HumanMessage(content="Sign me up."),
        AIMessage(content="", tool_calls=[_call({"eventinstance_id": event}, "c1")]),
        ToolMessage(content=PREVIEW, tool_call_id="c1", name="register_for_event"),
        AIMessage(content="Please confirm."),
    ]


class _Request:
    def __init__(self, tool_call: dict[str, Any], messages: list[Any]) -> None:
        self.tool_call = tool_call
        self.state = {"messages": messages}

    def override(self, **changes: Any) -> "_Request":
        return _Request(changes.get("tool_call", self.tool_call), self.state["messages"])


async def _sent_args(tool_call: dict[str, Any], messages: list[Any]) -> dict[str, Any]:
    seen: list[dict[str, Any]] = []

    async def handler(request: Any) -> ToolMessage:
        seen.append(request.tool_call["args"])
        return ToolMessage(content="{}", tool_call_id=request.tool_call["id"])

    await PreviewFirstMiddleware().awrap_tool_call(_Request(tool_call, messages), handler)
    return seen[0]


async def test_a_first_confirmed_call_becomes_a_preview():
    args = await _sent_args(_call({"eventinstance_id": "9372", "confirmed": True}), [])

    assert args == {"eventinstance_id": "9372", "confirmed": False}


async def test_a_preview_call_is_unchanged():
    args = await _sent_args(_call({"eventinstance_id": "9372"}), [])

    assert args == {"eventinstance_id": "9372"}


async def test_confirming_after_the_user_answered_the_preview_goes_through():
    messages = [*_previewed(), HumanMessage(content="Yes, confirm.")]

    args = await _sent_args(_call({"eventinstance_id": "9372", "confirmed": True}), messages)

    assert args == {"eventinstance_id": "9372", "confirmed": True}


async def test_confirming_with_no_user_answer_since_the_preview_stays_a_preview():
    args = await _sent_args(_call({"eventinstance_id": "9372", "confirmed": True}), _previewed())

    assert args["confirmed"] is False


async def test_a_preview_of_another_event_does_not_count():
    messages = [*_previewed("9373"), HumanMessage(content="Yes.")]

    args = await _sent_args(_call({"eventinstance_id": "9372", "confirmed": True}), messages)

    assert args["confirmed"] is False


async def test_a_failed_preview_does_not_count():
    messages = _previewed()
    messages[2] = ToolMessage(
        content=json.dumps({"status": "error"}), tool_call_id="c1", name="register_for_event"
    )
    messages.append(HumanMessage(content="Yes."))

    args = await _sent_args(_call({"eventinstance_id": "9372", "confirmed": True}), messages)

    assert args["confirmed"] is False


async def test_a_result_that_is_not_json_does_not_count():
    messages = _previewed()
    messages[2] = ToolMessage(content="Not done.", tool_call_id="c1", name="register_for_event")
    messages.append(HumanMessage(content="Yes."))

    args = await _sent_args(_call({"eventinstance_id": "9372", "confirmed": True}), messages)

    assert args["confirmed"] is False


async def test_other_tools_are_unchanged():
    call = {"name": "search_events", "args": {"confirmed": True}, "id": "c1", "type": "tool_call"}

    assert await _sent_args(call, []) == {"confirmed": True}


async def test_the_original_call_is_not_mutated():
    call = _call({"eventinstance_id": "9372", "confirmed": True})

    await _sent_args(call, [])

    assert call["args"]["confirmed"] is True


def _tool(content: str, call_id: str = "chatcmpl-tool-0") -> ToolMessage:
    return ToolMessage(content=content, tool_call_id=call_id, name="register_for_event")


def _ai(event: str, confirmed: Any = None, call_id: str = "chatcmpl-tool-0") -> AIMessage:
    args = {"eventinstance_id": event}
    if confirmed is not None:
        args["confirmed"] = confirmed
    return AIMessage(content="", tool_calls=[_call(args, call_id)])


REGISTERED = json.dumps({"status": "registered", "executed": True})


def test_a_reused_call_id_does_not_lend_one_events_preview_to_another():
    messages = [
        _ai("A"),
        _tool("Not done."),
        HumanMessage(content="Now B."),
        _ai("B"),
        _tool(PREVIEW),
        HumanMessage(content="Also A."),
    ]

    assert not PreviewFirstMiddleware._may_confirm(messages, "A")
    assert PreviewFirstMiddleware._may_confirm(messages, "B")


def test_a_downgraded_preview_counts_so_the_user_can_confirm():
    messages = [_ai("A", True), _tool(PREVIEW), HumanMessage(content="Yes.")]

    assert PreviewFirstMiddleware._may_confirm(messages, "A")


def test_a_newer_preview_needs_its_own_reply():
    messages = [
        _ai("A"),
        _tool(PREVIEW),
        HumanMessage(content="Still a seat?"),
        _ai("A"),
        _tool(PREVIEW),
    ]

    assert not PreviewFirstMiddleware._may_confirm(messages, "A")


def test_a_write_uses_the_preview_up():
    messages = [
        _ai("A"),
        _tool(PREVIEW),
        HumanMessage(content="Yes."),
        _ai("A", True),
        _tool(REGISTERED),
        HumanMessage(content="Thanks."),
    ]

    assert not PreviewFirstMiddleware._may_confirm(messages, "A")


def test_a_call_id_reused_by_another_tool_is_not_a_preview():
    other = AIMessage(
        content="",
        tool_calls=[
            {"name": "search_events", "args": {}, "id": "chatcmpl-tool-0", "type": "tool_call"}
        ],
    )
    messages = [other, _tool(PREVIEW), HumanMessage(content="Yes.")]

    assert not PreviewFirstMiddleware._may_confirm(messages, "A")


def test_numeric_ids_match_string_ids():
    middleware = PreviewFirstMiddleware()
    messages = [*_previewed(), HumanMessage(content="Yes.")]

    assert middleware._may_confirm(messages, 9372)


def test_missing_state_means_no_preview():
    request = SimpleNamespace(state={})

    assert PreviewFirstMiddleware._messages(request) == []


@pytest.mark.parametrize("confirmed", [True, "true", "True", "yes", 1])
async def test_every_way_of_saying_confirmed_becomes_a_preview(confirmed):
    args = await _sent_args(_call({"eventinstance_id": "9372", "confirmed": confirmed}), [])

    assert args["confirmed"] is False


async def test_a_value_that_is_not_a_boolean_passes_through_for_the_tool_to_reject():
    args = await _sent_args(_call({"eventinstance_id": "9372", "confirmed": "maybe"}), [])

    assert args["confirmed"] == "maybe"


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


async def test_a_real_agent_sends_the_downgraded_call_to_the_tool():
    received: list[Any] = []

    @tool
    def register_for_event(eventinstance_id: str, confirmed: bool = False) -> str:
        """Register for an event."""
        received.append(confirmed)
        return PREVIEW

    call = _call({"eventinstance_id": "9372", "confirmed": "true"}, "c1")
    model = _ScriptedModel(
        messages=iter([AIMessage(content="", tool_calls=[call]), AIMessage(content="Confirm?")])
    )
    agent = create_agent(
        model=model, tools=[register_for_event], middleware=[PreviewFirstMiddleware()]
    )

    await agent.ainvoke({"messages": [HumanMessage(content="Sign me up.")]})

    assert received == [False]


async def test_a_real_agent_lets_a_confirmed_call_through_after_a_reply():
    received: list[Any] = []

    @tool
    def register_for_event(eventinstance_id: str, confirmed: bool = False) -> str:
        """Register for an event."""
        received.append(confirmed)
        return REGISTERED

    call = _call({"eventinstance_id": "9372", "confirmed": True}, "c9")
    model = _ScriptedModel(
        messages=iter([AIMessage(content="", tool_calls=[call]), AIMessage(content="Submitted.")])
    )
    agent = create_agent(
        model=model, tools=[register_for_event], middleware=[PreviewFirstMiddleware()]
    )

    await agent.ainvoke({"messages": [*_previewed(), HumanMessage(content="Yes.")]})

    assert received == [True]


async def _result(tool_call: dict[str, Any], messages: list[Any], content: str = PREVIEW) -> Any:
    async def handler(request: Any) -> ToolMessage:
        return ToolMessage(
            content=content, tool_call_id=request.tool_call["id"], name="register_for_event"
        )

    return await PreviewFirstMiddleware().awrap_tool_call(_Request(tool_call, messages), handler)


async def test_a_downgraded_call_tells_the_model_the_preview_is_expected():
    """Without this the model reads a preview it didn't ask for as a backend fault."""
    result = await _result(_call({"eventinstance_id": "9372", "confirmed": True}), [])

    body = json.loads(result.content)
    assert body["status"] == "preview"
    assert "not an error" in body["note"]
    assert "confirm" in body["note"]


async def test_a_downgraded_preview_still_counts_as_a_preview():
    result = await _result(_call({"eventinstance_id": "9372", "confirmed": True}, "c1"), [])
    messages = [
        HumanMessage(content="Sign me up."),
        AIMessage(
            content="", tool_calls=[_call({"eventinstance_id": "9372", "confirmed": True}, "c1")]
        ),
        result,
        HumanMessage(content="Yes."),
    ]

    assert PreviewFirstMiddleware._may_confirm(messages, "9372")


async def test_a_preview_the_model_asked_for_gets_no_note():
    result = await _result(_call({"eventinstance_id": "9372"}), [])

    assert "note" not in json.loads(result.content)


async def test_a_confirmed_call_that_goes_through_gets_no_note():
    done = json.dumps({"status": "registered", "approval": "pending"})
    messages = [*_previewed(), HumanMessage(content="Yes, confirm.")]

    result = await _result(_call({"eventinstance_id": "9372", "confirmed": True}), messages, done)

    assert json.loads(result.content) == {"status": "registered", "approval": "pending"}


async def test_a_downgraded_call_with_an_error_result_is_left_alone():
    result = await _result(_call({"eventinstance_id": "9372", "confirmed": True}), [], "not json")

    assert result.content == "not json"
