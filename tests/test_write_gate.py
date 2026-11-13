"""A write runs only from the user's Confirm click, never from the model (D8-3088)."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command

from src.agent.middleware.local_times import LocalTimesMiddleware
from src.agent.middleware.requires_auth import RequiresAuthMiddleware
from src.agent.middleware.turn_budget import TurnBudgetMiddleware
from src.agent.middleware.write_gate import (
    CONFIRM_FAILED_NOTE,
    NOT_FROM_ASSISTANT_TEXT,
    PREVIEW_NOTE,
    SERVICE_DOWN_TEXT,
    TYPED_CONFIRM_NOTE,
    UNCONFIRMABLE_NOTE,
    UNCONFIRMED_WRITE_NOTE,
    WriteGateMiddleware,
    preview_body,
)
from src.agent.state import human_turn
from src.pending_previews import get_pending_preview_store
from tests.agent_runtime import context_for, run_loop

THREAD = "thread-1"
WRITER = context_for("user123", can_write=True)
PREVIEW = {
    "action": "register",
    "status": "preview",
    "executed": False,
    "data": {"title": "GPU Workshop", "start_date": "2026-11-20T18:00:00Z"},
}
REGISTERED = {"action": "register", "status": "registered", "executed": True, "data": {}}
HEX32 = re.compile(r"^[0-9a-f]{32}$")


@pytest.fixture(autouse=True)
def _sqlite_db(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.pending_previews as previews
    import src.thread_owners as owners

    owners._store = None
    previews._store = None


def register_for_event(eventinstance_id: str, confirmed: bool = False) -> str:
    """Register for an event."""
    return ""


def create_support_ticket(summary: str) -> str:
    """File a ticket."""
    return ""


REGISTER_TOOL = StructuredTool.from_function(register_for_event)
NO_CONFIRM_TOOL = StructuredTool.from_function(create_support_ticket)


def _call(args: dict[str, Any], name: str = "register_for_event", call_id: str = "c1") -> Any:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def _request(call: dict[str, Any], tool_: Any = REGISTER_TOOL) -> ToolCallRequest:
    return ToolCallRequest(tool_call=call, tool=tool_, state={}, runtime=Runtime(context=WRITER))


def _gate(confirm_ids: list[str] | None = None) -> WriteGateMiddleware:
    return WriteGateMiddleware(thread_id=THREAD, confirm_ids=confirm_ids or [])


async def _run(
    call: dict[str, Any],
    content: Any = None,
    *,
    tool_: Any = REGISTER_TOOL,
    gate: WriteGateMiddleware | None = None,
) -> tuple[list[dict[str, Any]], Any]:
    """The args the tool received and the result the model gets back."""
    sent: list[dict[str, Any]] = []
    body = json.dumps(PREVIEW) if content is None else content

    async def handler(request: Any) -> Any:
        sent.append(request.tool_call["args"])
        if isinstance(body, Command):
            return body
        if isinstance(body, ToolMessage):
            return body
        return ToolMessage(content=body, tool_call_id=request.tool_call["id"], name=call["name"])

    result = await (gate or _gate()).awrap_tool_call(_request(call, tool_), handler)
    return sent, result


# ── Model-made calls ─────────────────────────────────────────────────────────


async def test_a_tool_that_does_not_write_passes_through_untouched():
    call = _call({"confirmed": True}, name="search_events")

    sent, result = await _run(call, '{"status": "preview"}', tool_=None)

    assert sent == [{"confirmed": True}]
    assert result.content == '{"status": "preview"}'


async def test_a_write_tool_with_no_confirmed_argument_is_refused_without_running():
    async def handler(request: Any) -> ToolMessage:
        raise AssertionError("a write that can't preview must never run")

    call = _call({"summary": "help"}, name="create_support_ticket")
    result = await _gate().awrap_tool_call(_request(call, NO_CONFIRM_TOOL), handler)

    assert result.content == NOT_FROM_ASSISTANT_TEXT
    assert result.status == "error"
    assert result.tool_call_id == "c1"


async def test_a_write_tool_the_agent_does_not_have_is_refused_without_running():
    async def handler(request: Any) -> ToolMessage:
        raise AssertionError("must not run")

    result = await _gate().awrap_tool_call(_request(_call({"confirmed": True}), None), handler)

    assert result.content == NOT_FROM_ASSISTANT_TEXT


async def test_a_model_confirmed_call_runs_as_a_preview_that_is_recorded_and_stamped():
    sent, result = await _run(_call({"eventinstance_id": "9372", "confirmed": True}))

    assert sent == [{"eventinstance_id": "9372", "confirmed": False}]
    assert isinstance(result.content, str)
    body = json.loads(result.content)
    assert HEX32.match(body["preview_id"])
    assert body["note"] == PREVIEW_NOTE
    assert {k: body[k] for k in PREVIEW} == PREVIEW
    claimed = get_pending_preview_store().claim(THREAD, body["preview_id"])
    assert claimed == ("register_for_event", {"eventinstance_id": "9372"})


@pytest.mark.parametrize("confirmed", [True, "true", "True", "yes", 1, "maybe", False])
async def test_every_value_of_confirmed_from_the_model_becomes_false(confirmed):
    sent, _ = await _run(_call({"eventinstance_id": "9372", "confirmed": confirmed}))

    assert sent[0]["confirmed"] is False


async def test_a_call_without_confirmed_is_sent_with_it_false():
    sent, _ = await _run(_call({"eventinstance_id": "9372"}))

    assert sent == [{"eventinstance_id": "9372", "confirmed": False}]


async def test_the_original_call_is_not_mutated():
    call = _call({"eventinstance_id": "9372", "confirmed": True})

    await _run(call)

    assert call["args"]["confirmed"] is True


async def test_a_preview_with_no_changes_keeps_the_flag():
    preview = {**PREVIEW, "no_changes": True}

    _, result = await _run(_call({"eventinstance_id": "1"}), json.dumps(preview))

    body = json.loads(result.content)
    assert body["no_changes"] is True
    assert "preview_id" in body


async def test_a_preview_in_content_blocks_is_stamped_as_a_json_string():
    blocks = [{"type": "text", "text": json.dumps(PREVIEW)}]

    _, result = await _run(_call({"eventinstance_id": "1"}), blocks)

    body = json.loads(result.content)
    assert body["status"] == "preview"
    assert HEX32.match(body["preview_id"])


async def test_a_preview_whose_record_fails_says_so_and_has_no_id(monkeypatch):
    monkeypatch.setattr(get_pending_preview_store(), "record", lambda *a: None)

    _, result = await _run(_call({"eventinstance_id": "1"}))

    body = json.loads(result.content)
    assert body["note"] == UNCONFIRMABLE_NOTE
    assert "preview_id" not in body
    assert body["status"] == "preview"


@pytest.mark.parametrize(
    "content",
    [
        json.dumps(REGISTERED),
        json.dumps({**PREVIEW, "executed": True}),
        json.dumps(["not", "an", "object"]),
    ],
)
async def test_a_forced_call_that_wrote_anyway_is_logged_and_noted(caplog, content):
    caplog.set_level(logging.ERROR)

    _, result = await _run(_call({"eventinstance_id": "1"}), content)

    assert "WRITE_WITHOUT_CONFIRM" in caplog.text
    assert "register_for_event" in caplog.text
    assert UNCONFIRMED_WRITE_NOTE in result.content
    assert "preview_id" not in result.content


async def test_a_noted_object_result_stays_json():
    _, result = await _run(_call({"eventinstance_id": "1"}), json.dumps(REGISTERED))

    body = json.loads(result.content)
    assert body["note"] == UNCONFIRMED_WRITE_NOTE
    assert body["status"] == "registered"


async def test_a_result_that_is_not_json_is_not_a_preview(caplog):
    caplog.set_level(logging.ERROR)

    _, result = await _run(_call({"eventinstance_id": "1"}), "Registered!")

    assert "WRITE_WITHOUT_CONFIRM" in caplog.text
    assert result.content == f"Registered!\n\n{UNCONFIRMED_WRITE_NOTE}"


@pytest.mark.parametrize(
    "message",
    [
        ToolMessage(content="Not run.", tool_call_id="c1", status="error"),
        ToolMessage(content=json.dumps({"error": "event is full"}), tool_call_id="c1"),
    ],
)
async def test_a_failed_call_is_left_alone(caplog, message):
    """A failure wrote nothing, so it gets no warning that it may have."""
    caplog.set_level(logging.ERROR)

    _, result = await _run(_call({"eventinstance_id": "1"}), message)

    assert result is message
    assert "WRITE_WITHOUT_CONFIRM" not in caplog.text


async def test_a_command_result_is_left_alone():
    command = Command(update={})

    _, result = await _run(_call({"eventinstance_id": "1"}), command)

    assert result is command


# ── Confirm turns ────────────────────────────────────────────────────────────


def _recorded(args: dict[str, Any] | None = None, thread: str = THREAD) -> str:
    preview_id = get_pending_preview_store().record(
        thread, "register_for_event", args or {"eventinstance_id": "9372"}
    )
    assert preview_id is not None
    return preview_id


async def _before(gate: WriteGateMiddleware) -> dict[str, Any] | None:
    return await gate.abefore_model({"messages": []}, Runtime(context=WRITER))


async def _view(gate: WriteGateMiddleware, messages: list[Any]) -> list[Any]:
    seen: list[Any] = []

    async def handler(request: ModelRequest) -> Any:
        seen.extend(request.messages)
        return MagicMock()

    request = ModelRequest(
        model=MagicMock(),
        messages=messages,
        system_message=None,
        tool_choice=None,
        tools=[],
        response_format=None,
        state={"messages": messages},
        runtime=Runtime(context=WRITER),
    )
    await gate.awrap_model_call(request, handler)
    return seen


async def test_a_confirm_claims_the_preview_and_jumps_to_its_stored_call():
    preview_id = _recorded()
    gate = _gate([preview_id])

    update = await _before(gate)

    assert update is not None
    assert update["jump_to"] == "tools"
    [message] = update["messages"]
    assert isinstance(message, AIMessage)
    assert [(c["id"], c["name"], c["args"]) for c in message.tool_calls] == [
        (
            f"confirm-{preview_id}",
            "register_for_event",
            {"eventinstance_id": "9372", "confirmed": True},
        )
    ]
    assert get_pending_preview_store().claim(THREAD, preview_id) is None


async def test_the_confirmed_call_passes_the_gate_unchanged():
    preview_id = _recorded()
    gate = _gate([preview_id])
    call = (await _before(gate))["messages"][0].tool_calls[0]  # type: ignore[index]

    sent, result = await _run(call, json.dumps(REGISTERED), gate=gate)

    assert sent == [{"eventinstance_id": "9372", "confirmed": True}]
    assert json.loads(result.content) == REGISTERED


async def test_a_confirmed_call_id_runs_confirmed_only_once():
    preview_id = _recorded()
    gate = _gate([preview_id])
    call = (await _before(gate))["messages"][0].tool_calls[0]  # type: ignore[index]
    await _run(call, json.dumps(REGISTERED), gate=gate)

    sent, _ = await _run(call, gate=gate)

    assert sent[0]["confirmed"] is False


async def test_two_confirm_ids_make_two_calls_in_one_message():
    first = _recorded({"eventinstance_id": "1"})
    second = _recorded({"eventinstance_id": "2"})

    update = await _before(_gate([first, second]))

    [message] = update["messages"]  # type: ignore[index]
    assert [c["args"]["eventinstance_id"] for c in message.tool_calls] == ["1", "2"]
    assert [c["id"] for c in message.tool_calls] == [f"confirm-{first}", f"confirm-{second}"]


async def test_a_repeated_confirm_id_makes_one_call_and_no_failure_note():
    preview_id = _recorded()
    gate = _gate([preview_id, preview_id])

    update = await _before(gate)

    assert len(update["messages"][0].tool_calls) == 1  # type: ignore[index]
    view = await _view(gate, [HumanMessage(content="Confirm")])
    assert CONFIRM_FAILED_NOTE not in view[-1].content


@pytest.mark.parametrize("kind", ["spent", "foreign", "forged"])
async def test_an_unclaimable_id_makes_no_jump_and_tells_the_model(kind):
    if kind == "spent":
        preview_id = _recorded()
        get_pending_preview_store().claim(THREAD, preview_id)
    elif kind == "foreign":
        preview_id = _recorded(thread="someone-else")
    else:
        preview_id = "0" * 32
    gate = _gate([preview_id])

    assert await _before(gate) is None

    view = await _view(gate, [HumanMessage(content="Confirm")])
    assert view[-1].content == f"Confirm\n\n{CONFIRM_FAILED_NOTE}"


async def test_a_foreign_id_stays_claimable_by_its_own_thread():
    preview_id = _recorded(thread="someone-else")

    await _before(_gate([preview_id]))

    assert get_pending_preview_store().claim("someone-else", preview_id) is not None


async def test_one_good_and_one_spent_id_run_the_good_one_and_note_the_other():
    good = _recorded()
    gate = _gate([good, "f" * 32])

    update = await _before(gate)

    assert len(update["messages"][0].tool_calls) == 1  # type: ignore[index]
    view = await _view(gate, [HumanMessage(content="Confirm")])
    assert view[-1].content.count(CONFIRM_FAILED_NOTE) == 1
    assert "1 of the 2 changes" in view[-1].content


async def test_when_every_confirmed_change_failed_the_note_says_how_many():
    gate = _gate(["e" * 32, "f" * 32])

    assert await _before(gate) is None

    view = await _view(gate, [HumanMessage(content="Confirm")])
    assert view[-1].content.count(CONFIRM_FAILED_NOTE) == 1
    assert "2 of the 2 changes" in view[-1].content


async def test_without_a_failed_id_the_model_view_is_unchanged():
    messages = [HumanMessage(content="Hi")]

    view = await _view(_gate(), messages)

    assert view == messages


async def test_the_failure_note_does_not_touch_the_stored_message():
    gate = _gate(["0" * 32])
    await _before(gate)
    question = HumanMessage(content="Confirm")

    await _view(gate, [question])

    assert question.content == "Confirm"


async def test_the_second_model_call_of_a_turn_claims_nothing_more():
    preview_id = _recorded()
    gate = _gate([preview_id])
    assert await _before(gate) is not None
    later = _recorded()
    gate.confirm_ids.append(later)

    assert await _before(gate) is None
    assert get_pending_preview_store().claim(THREAD, later) is not None


async def test_a_turn_with_no_confirm_ids_never_jumps():
    assert await _before(_gate()) is None


# ── Through a real agent ─────────────────────────────────────────────────────


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedModel:
        return self


class _Sees(AgentMiddleware):
    """Records each tool call that reaches its place in the stack."""

    def __init__(self, label: str, seen: list[tuple[str, str, Any]]) -> None:
        super().__init__()
        self.label = label
        self.seen = seen

    @property
    def name(self) -> str:
        return f"sees-{self.label}"

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        call = request.tool_call
        self.seen.append((self.label, call["id"], call["args"].get("confirmed")))
        return await handler(request)


def _stack(gate: WriteGateMiddleware, seen: list[tuple[str, str, Any]]) -> list[Any]:
    """The loop's tool-call middleware, in its order, with a witness around each."""
    return [
        _Sees("outer", seen),
        RequiresAuthMiddleware(),
        _Sees("after-auth", seen),
        gate,
        _Sees("after-gate", seen),
        LocalTimesMiddleware("America/New_York"),
        TurnBudgetMiddleware(time.monotonic() + 60),
        _Sees("inside-budget", seen),
    ]


def _agent(answers: list[AIMessage], ran: list[Any], gate: WriteGateMiddleware, seen: list) -> Any:
    @tool("register_for_event")
    def stub(eventinstance_id: str, confirmed: bool = False) -> str:
        """Register for an event."""
        ran.append((eventinstance_id, confirmed))
        return json.dumps(REGISTERED if confirmed else PREVIEW)

    model = _ScriptedModel(messages=iter(answers))
    return create_agent(model=model, tools=[stub], middleware=_stack(gate, seen))


async def test_jumping_to_tools_runs_the_confirmed_call_through_every_tool_middleware():
    """The replay relies on this: the injected call must meet the same gates
    as any other, including the acting-user check."""
    preview_id = _recorded({"eventinstance_id": "9372"})
    ran: list[Any] = []
    seen: list[tuple[str, str, Any]] = []
    agent = _agent([AIMessage(content="You're registered.")], ran, _gate([preview_id]), seen)

    result = await agent.ainvoke(
        {"messages": [human_turn("Confirm", [preview_id])]}, context=WRITER
    )

    call_id = f"confirm-{preview_id}"
    assert ran == [("9372", True)]
    assert [label for label, cid, _ in seen if cid == call_id] == [
        "outer",
        "after-auth",
        "after-gate",
        "inside-budget",
    ]
    tool_result = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert tool_result.tool_call_id == call_id
    assert result["messages"][-1].content == "You're registered."


async def test_the_acting_user_check_still_refuses_a_confirmed_call():
    preview_id = _recorded()
    ran: list[Any] = []
    agent = _agent([AIMessage(content="Sign in first.")], ran, _gate([preview_id]), [])

    result = await agent.ainvoke(
        {"messages": [human_turn("Confirm", [preview_id])]}, context=context_for(None)
    )

    assert ran == []
    refusal = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert refusal.status == "error"


async def test_replaying_a_confirm_turn_writes_nothing_more():
    """A Retry or Edit resends the same ids in a new turn, with a new gate."""
    preview_id = _recorded()
    ran: list[Any] = []
    question = human_turn("Confirm", [preview_id])
    for _ in range(2):
        agent = _agent([AIMessage(content="Done.")], ran, _gate([preview_id]), [])
        await agent.ainvoke({"messages": [question]}, context=WRITER)

    assert ran == [("9372", True)]


async def test_a_typed_yes_only_ever_previews():
    """The model can't confirm a write, however the user says yes."""
    ran: list[Any] = []
    seen_by_model: list[Any] = []
    call = _call({"eventinstance_id": "9372", "confirmed": True})
    gate = _gate()
    agent = _agent(
        [AIMessage(content="", tool_calls=[call]), AIMessage(content="Please click Confirm.")],
        ran,
        gate,
        seen_by_model,
    )

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content="Yes, register me.")]}, context=WRITER
    )

    assert ran == [("9372", False)]
    body = preview_body(next(m for m in result["messages"] if isinstance(m, ToolMessage)))
    assert body is not None
    assert body["note"] == PREVIEW_NOTE
    assert body["data"]["start_local"] == "Fri, Nov 20, 2026, 1:00 PM EST"
    assert get_pending_preview_store().claim(THREAD, body["preview_id"]) is not None


# ── The loop builds it ───────────────────────────────────────────────────────


async def test_the_loop_builds_the_gate_from_the_thread_and_the_latest_question():
    built: dict[str, Any] = {}

    def capture(**kwargs: Any) -> MagicMock:
        built.update(kwargs)
        return MagicMock(ainvoke=AsyncMock(return_value={"messages": [AIMessage(content="ok")]}))

    messages = [
        human_turn("Confirm", ["a" * 32]),
        AIMessage(content="Done."),
        human_turn("Confirm", ["b" * 32, "c" * 32]),
    ]
    with (
        patch("src.agent.nodes.tool_calling_loop.create_agent", side_effect=capture),
        patch("src.agent.nodes.tool_calling_loop.get_llm", return_value=MagicMock()),
    ):
        await run_loop({"messages": messages, "tool_catalog": {}, "session_id": THREAD})

    kinds = [type(m) for m in built["middleware"]]
    gate = built["middleware"][kinds.index(WriteGateMiddleware)]
    assert gate.thread_id == THREAD
    assert gate.confirm_ids == ["b" * 32, "c" * 32]
    assert (
        kinds.index(RequiresAuthMiddleware)
        < kinds.index(WriteGateMiddleware)
        < kinds.index(LocalTimesMiddleware)
        < kinds.index(TurnBudgetMiddleware)
    )


async def test_the_loop_gives_a_plain_question_no_confirm_ids():
    built: dict[str, Any] = {}

    def capture(**kwargs: Any) -> MagicMock:
        built.update(kwargs)
        return MagicMock(ainvoke=AsyncMock(return_value={"messages": [AIMessage(content="ok")]}))

    messages = [human_turn("Confirm", ["a" * 32]), AIMessage(content="Done."), human_turn("Hi")]
    with (
        patch("src.agent.nodes.tool_calling_loop.create_agent", side_effect=capture),
        patch("src.agent.nodes.tool_calling_loop.get_llm", return_value=MagicMock()),
    ):
        await run_loop({"messages": messages, "tool_catalog": {}, "session_id": THREAD})

    gate = next(m for m in built["middleware"] if isinstance(m, WriteGateMiddleware))
    assert gate.confirm_ids == []


# ── Fix round 1: fail closed, and approval tied to the replayed call ─────────


def update_widget(widget_id: str, confirmed: bool = False) -> str:
    """Update a widget."""
    return ""


def update_gadget(gadget_id: str) -> str:
    """Update a gadget."""
    return ""


class _HintedTool(StructuredTool):
    read_only: bool | None = None


UNLISTED_WRITE = _HintedTool.from_function(update_widget, read_only=False)
UNLISTED_NO_CONFIRM = _HintedTool.from_function(update_gadget, read_only=False)


async def test_an_unlisted_write_is_still_forced_to_a_preview():
    sent, result = await _run(
        _call({"widget_id": "w1", "confirmed": True}, name="update_widget"),
        tool_=UNLISTED_WRITE,
    )

    assert sent == [{"widget_id": "w1", "confirmed": False}]
    assert HEX32.match(json.loads(result.content)["preview_id"])


async def test_an_unlisted_write_without_confirmed_is_refused():
    async def handler(request: Any) -> ToolMessage:
        raise AssertionError("must not run")

    call = _call({"gadget_id": "g1"}, name="update_gadget")
    result = await _gate().awrap_tool_call(_request(call, UNLISTED_NO_CONFIRM), handler)

    assert result.content == NOT_FROM_ASSISTANT_TEXT


async def test_a_tool_the_server_marks_as_a_write_is_gated_whatever_its_name():
    tool_ = _HintedTool.from_function(register_for_event, name="sign_up", read_only=False)

    sent, _ = await _run(
        _call({"eventinstance_id": "1", "confirmed": True}, name="sign_up"), tool_=tool_
    )

    assert sent[0]["confirmed"] is False


async def _approved_call(gate: WriteGateMiddleware) -> dict[str, Any]:
    update = await _before(gate)
    assert update is not None
    return dict(update["messages"][0].tool_calls[0])


async def test_a_reused_confirm_id_with_other_args_is_forced_to_a_preview():
    gate = _gate([_recorded({"eventinstance_id": "9372"})])
    call = await _approved_call(gate)
    changed = {**call, "args": {"eventinstance_id": "1111", "confirmed": True}}

    sent, _ = await _run(changed, gate=gate)

    assert sent == [{"eventinstance_id": "1111", "confirmed": False}]


async def test_a_reused_confirm_id_for_another_tool_is_forced_to_a_preview():
    gate = _gate([_recorded({"eventinstance_id": "9372"})])
    call = await _approved_call(gate)
    other = {**call, "name": "cancel_registration"}

    sent, _ = await _run(other, gate=gate)

    assert sent[0]["confirmed"] is False


async def test_approval_ends_with_the_first_tool_round():
    """A call refused before reaching the gate must not leave its approval behind."""
    gate = _gate([_recorded()])
    call = await _approved_call(gate)

    assert await _before(gate) is None
    sent, _ = await _run(call, gate=gate)

    assert sent[0]["confirmed"] is False


async def test_a_later_model_call_reusing_the_confirm_id_is_forced_to_a_preview():
    preview_id = _recorded({"eventinstance_id": "9372"})
    ran: list[Any] = []
    reuse = _call({"eventinstance_id": "9372", "confirmed": True}, call_id=f"confirm-{preview_id}")
    agent = _agent(
        [AIMessage(content="", tool_calls=[reuse]), AIMessage(content="Done.")],
        ran,
        _gate([preview_id]),
        [],
    )

    await agent.ainvoke({"messages": [human_turn("Confirm", [preview_id])]}, context=WRITER)

    assert ran == [("9372", True), ("9372", False)]


@pytest.mark.parametrize(
    "content",
    [json.dumps(PREVIEW), json.dumps({"status": "already_registered", "executed": False})],
)
async def test_a_confirmed_call_that_did_not_execute_is_logged(caplog, content):
    caplog.set_level(logging.WARNING)
    gate = _gate([_recorded()])
    call = await _approved_call(gate)

    _, result = await _run(call, content, gate=gate)

    assert "register_for_event" in caplog.text
    assert "CONFIRMED_NOT_EXECUTED" in caplog.text
    assert result.content == content


async def test_a_confirmed_call_that_errored_is_logged(caplog):
    caplog.set_level(logging.WARNING)
    gate = _gate([_recorded()])
    call = await _approved_call(gate)
    error = ToolMessage(
        content="timed out", tool_call_id=call["id"], name=call["name"], status="error"
    )

    await _run(call, error, gate=gate)

    assert "CONFIRMED_NOT_EXECUTED" in caplog.text


async def test_a_confirmed_call_that_executed_logs_nothing(caplog):
    caplog.set_level(logging.WARNING)
    gate = _gate([_recorded()])
    call = await _approved_call(gate)

    await _run(call, json.dumps(REGISTERED), gate=gate)

    assert "CONFIRMED_NOT_EXECUTED" not in caplog.text


# ── Review fixes ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "text",
    ["Confirm", "confirm", " Yes. ", "y", "OK!", "okay", "Go ahead", "do it", "Proceed."],
)
async def test_a_typed_confirmation_with_no_ids_tells_the_model_nothing_changed(text):
    view = await _view(_gate(), [HumanMessage(content=text)])

    assert view[-1].content.count(TYPED_CONFIRM_NOTE) == 1
    assert TYPED_CONFIRM_NOTE == (
        "The user typed a confirmation instead of clicking Confirm, so nothing was "
        "changed. Ask them to click Confirm on the card."
    )


@pytest.mark.parametrize(
    "text", ["yes please register me for the other one too", "Confirm the date is Tuesday", ""]
)
async def test_a_longer_sentence_is_not_flagged_as_a_typed_confirmation(text):
    messages = [HumanMessage(content=text)]

    assert await _view(_gate(), messages) == messages


async def test_a_confirm_click_with_ids_gets_no_typed_confirmation_note():
    gate = _gate(["e" * 32])
    await _before(gate)

    view = await _view(gate, [HumanMessage(content="Confirm")])

    assert TYPED_CONFIRM_NOTE not in view[-1].content
    assert CONFIRM_FAILED_NOTE in view[-1].content


def test_the_failure_note_only_says_this_click_changed_nothing():
    assert CONFIRM_FAILED_NOTE == (
        "The user clicked Confirm, but that preview was already used or has expired, so "
        "this click changed nothing. Tell them so and offer to preview it again."
    )


async def test_an_approved_replay_of_a_tool_missing_from_the_catalog_says_the_service_is_down():
    gate = _gate([_recorded({"eventinstance_id": "9372"})])
    call = await _approved_call(gate)

    async def handler(request: Any) -> ToolMessage:
        raise AssertionError("must not run")

    result = await gate.awrap_tool_call(_request(call, None), handler)

    assert result.content == SERVICE_DOWN_TEXT
    assert SERVICE_DOWN_TEXT == (
        "That service isn't available right now, so nothing was changed. Tell the user "
        "and offer to preview it again later."
    )
    assert result.status == "error"
    assert result.tool_call_id == call["id"]
