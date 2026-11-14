"""Tests for the model's view of tool results from earlier turns.

The unit tests build the thread shapes that reach the model: reused call ids,
compacted threads, context-edited results. The graph tests run a real
create_agent with a scripted model that records the messages it receives, since
what the model is sent is the behavior under test.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import ClearToolUsesEdit, ContextEditingMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.domains.capabilities import (
    LIVE_DATA_MCP_TOOL_NAMES,
    WRITE_MCP_TOOL_NAMES,
    _parse_live_data_tools,
    live_data_tool_names,
)
from src.agent.middleware.earlier_results import (
    EARLIER_FAILURE_TEXT,
    EARLIER_NOTES_PREFIX,
    EARLIER_RESULT_PREFIX,
    EARLIER_TURN_NOTE,
    EarlierResultsMiddleware,
    earlier_results_view,
)
from src.agent.middleware.requires_auth import (
    SIGNED_IN_NOTE,
    RequiresAuthMiddleware,
    refusal_artifact,
    refusal_text,
)
from src.agent.nodes.tool_calling_loop import (
    _build_context_editing_middleware,
    _edit_aware_token_counter,
)
from src.agent.token_estimate import estimate_messages
from src.config import settings
from tests.agent_runtime import context_for, run_loop

LIVE = "get_chart_data"
STATIC = "search_software"
TIMEOUT = json.dumps({"error": "XDMoD request timed out"})
REUSED_ID = "chatcmpl-tool-0"


def _ai_call(name: str, call_id: str = REUSED_ID, args: dict[str, Any] | None = None) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args or {}, "id": call_id, "type": "tool_call"}],
    )


def _result(name: str, content: str = '{"jobs": 12}', **fields: Any) -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id=REUSED_ID, **fields)


def _earlier_turn(name: str, content: str = '{"jobs": 12}', **fields: Any) -> list[Any]:
    return [
        HumanMessage(content="Any usage data from ACES?"),
        _ai_call(name),
        _result(name, content, **fields),
        AIMessage(content="XDMoD is currently timing out."),
    ]


def _asked_again(text: str = "Any usage data from ACES?") -> HumanMessage:
    return HumanMessage(content=text)


def _noted(text: str = "Any usage data from ACES?") -> str:
    return f"{text}\n\n{EARLIER_TURN_NOTE}"


def _summary(text: str = "XDMoD timed out.") -> AIMessage:
    return AIMessage(content=text, additional_kwargs={"lc_source": "summarization"})


def test_a_first_question_is_unchanged():
    messages = [_asked_again()]

    assert earlier_results_view(messages) is messages


def test_a_thread_with_no_question_is_unchanged():
    messages = [_ai_call(LIVE), _result(LIVE, TIMEOUT)]

    assert earlier_results_view(messages) is messages


def test_a_result_from_this_turn_is_unchanged():
    messages = [_asked_again(), _ai_call(LIVE), _result(LIVE, TIMEOUT)]

    assert earlier_results_view(messages) is messages


def test_an_earlier_live_result_keeps_its_data_and_is_marked_as_old():
    view = earlier_results_view([*_earlier_turn(LIVE), _asked_again()])

    assert view[2].content == f'{EARLIER_RESULT_PREFIX}\n\n{{"jobs": 12}}'
    assert view[-1].content == _noted()


@pytest.mark.parametrize(
    ("content", "fields"),
    [(TIMEOUT, {}), ("Error: tool timed out", {"status": "error"})],
    ids=["mcp-error-json", "error-status"],
)
def test_an_earlier_failure_reads_as_past_and_loses_its_wording(content, fields):
    view = earlier_results_view([*_earlier_turn(LIVE, content, **fields), _asked_again()])

    assert view[2].content == EARLIER_FAILURE_TEXT
    assert "timed out" not in view[2].content
    assert view[-1].content == _noted()


def test_an_earlier_result_of_a_static_tool_is_unchanged():
    messages = [*_earlier_turn(STATIC), _asked_again()]

    assert earlier_results_view(messages) is messages


def test_an_earlier_failure_of_a_static_tool_reads_as_past():
    view = earlier_results_view([*_earlier_turn(STATIC, TIMEOUT), _asked_again()])

    assert view[2].content == EARLIER_FAILURE_TEXT


@pytest.mark.parametrize(
    "content",
    ["not json", "[1, 2]", '{"items": []}'],
    ids=["text", "json-list", "json-object"],
)
def test_a_static_result_that_is_not_an_error_is_not_a_failure(content):
    messages = [*_earlier_turn(STATIC, content), _asked_again()]

    assert earlier_results_view(messages) is messages


def test_a_failed_write_keeps_its_reason_and_does_not_invite_a_retry():
    messages = [*_earlier_turn("register_for_event", TIMEOUT), _asked_again()]

    assert earlier_results_view(messages) is messages


def test_a_refusal_that_still_applies_is_unchanged():
    refused = _earlier_turn(
        "get_my_rp_accounts",
        refusal_text("private_data"),
        status="error",
        artifact=refusal_artifact("private_data"),
    )
    messages = [*refused, _asked_again()]

    assert earlier_results_view(messages) is messages


def test_a_result_with_no_tool_name_is_unchanged():
    message = ToolMessage(content=TIMEOUT, tool_call_id=REUSED_ID)
    messages = [_ai_call(LIVE), message, _asked_again()]

    assert earlier_results_view(messages) is messages


def test_a_reused_call_id_does_not_lend_one_tool_the_others_class():
    messages = [
        HumanMessage(content="Is Python on Delta?"),
        _ai_call(STATIC),
        _result(STATIC, '{"software": "python"}'),
        AIMessage(content="Yes."),
        HumanMessage(content="And usage?"),
        _ai_call(LIVE),
        _result(LIVE),
        AIMessage(content="12 jobs."),
        _asked_again(),
    ]

    view = earlier_results_view(messages)

    assert view[2].content == '{"software": "python"}'
    assert view[6].content.startswith(EARLIER_RESULT_PREFIX)


def test_the_stored_messages_are_not_changed():
    messages = [*_earlier_turn(LIVE, TIMEOUT), _asked_again()]

    earlier_results_view(messages)

    assert messages[2].content == TIMEOUT
    assert messages[-1].content == "Any usage data from ACES?"


def test_a_rewritten_message_keeps_its_identity_fields():
    messages = [*_earlier_turn(LIVE, TIMEOUT, id="t-1"), _asked_again()]

    rewritten = earlier_results_view(messages)[2]

    assert (rewritten.id, rewritten.name, rewritten.tool_call_id) == ("t-1", LIVE, REUSED_ID)


def test_a_question_in_content_blocks_still_gets_the_rewrite_without_a_note():
    blocks = [{"type": "text", "text": "Any usage data?"}]
    view = earlier_results_view([*_earlier_turn(LIVE), HumanMessage(content=blocks)])

    assert view[2].content.startswith(EARLIER_RESULT_PREFIX)
    assert view[-1].content == blocks


def test_compaction_notes_from_an_earlier_turn_are_marked_as_old():
    view = earlier_results_view([HumanMessage(content="Usage?"), _summary(), _asked_again()])

    assert view[1].content == f"{EARLIER_NOTES_PREFIX}\n\nXDMoD timed out."
    assert view[-1].content == _noted()


def test_compaction_mid_turn_leaves_this_turns_notes_and_results_alone():
    """Compaction moves the questions ahead of the notes and the kept tail."""
    messages = [
        HumanMessage(content="Usage?"),
        _asked_again(),
        _summary(),
        _ai_call(LIVE),
        _result(LIVE, TIMEOUT),
    ]

    assert earlier_results_view(messages) is messages


def _context_edited(thread: list[Any]) -> list[Any]:
    """The thread after ClearToolUsesEdit, which also drops the artifact."""
    ClearToolUsesEdit(trigger=0, keep=0).apply(thread, count_tokens=count_tokens_approximately)
    return thread


def test_a_cleared_live_result_is_still_marked_as_old():
    thread = _context_edited([*_earlier_turn(LIVE, TIMEOUT), _asked_again()])

    view = earlier_results_view(thread)

    assert view[2].content.startswith(EARLIER_RESULT_PREFIX)
    assert view[-1].content == _noted()


def test_a_cleared_refusal_is_not_taken_for_a_failure():
    thread = _context_edited(
        [
            *_earlier_turn(
                "get_software_details",
                refusal_text("private_data"),
                status="error",
                artifact=refusal_artifact("private_data"),
            ),
            _asked_again(),
        ]
    )

    assert thread[2].artifact is None
    assert earlier_results_view(thread) is thread


def test_the_live_tools_are_all_reads():
    assert not LIVE_DATA_MCP_TOOL_NAMES & WRITE_MCP_TOOL_NAMES


def test_the_live_tools_default_to_the_shared_list():
    assert live_data_tool_names() == LIVE_DATA_MCP_TOOL_NAMES


def test_an_operator_can_replace_the_live_tools(monkeypatch):
    monkeypatch.setattr(settings, "LIVE_DATA_TOOLS", " program_stats, ,queue_depth ")

    assert live_data_tool_names() == {"program_stats", "queue_depth"}

    view = earlier_results_view([*_earlier_turn("queue_depth"), _asked_again()])
    assert view[2].content.startswith(EARLIER_RESULT_PREFIX)
    assert earlier_results_view(thread := [*_earlier_turn(LIVE), _asked_again()]) is thread


def test_the_summarization_counter_measures_the_view_the_model_gets():
    """A failure's body is replaced, so counting it would overstate the thread."""
    body = json.dumps({"error": "X" * 4000})
    thread = [*_earlier_turn(LIVE, body), _asked_again()]

    counted = _edit_aware_token_counter()(thread)

    assert counted == estimate_messages(earlier_results_view(thread))
    assert counted < estimate_messages(thread)


def test_the_summarization_counter_copes_with_a_tail_that_has_no_question():
    """Upstream searches for the cutoff by counting suffixes of the thread."""
    tail = [_ai_call(LIVE), _result(LIVE, TIMEOUT)]

    assert _edit_aware_token_counter()(tail) == estimate_messages(tail)


def _recording_model(script: list[AIMessage], seen: list[list[Any]]) -> GenericFakeChatModel:
    class Model(GenericFakeChatModel):
        def bind_tools(self, tools: Any, **kwargs: Any) -> Model:
            return self

        def _generate(self, messages: list[Any], *args: Any, **kwargs: Any) -> Any:
            seen.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

    return Model(messages=iter(script))


def _tool_results(messages: list[Any]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


async def test_a_real_agent_sends_the_rewritten_view_and_stores_the_original():
    """Two turns through one checkpointed thread, with the failure on the first."""
    outcomes = iter([TIMEOUT, '{"jobs": 12}'])

    @tool
    def get_chart_data() -> str:
        """Usage data."""
        return next(outcomes)

    seen: list[list[Any]] = []
    model = _recording_model(
        [
            AIMessage(content="", tool_calls=[_ai_call(LIVE).tool_calls[0]]),
            AIMessage(content="XDMoD is currently timing out."),
            AIMessage(content="", tool_calls=[_ai_call(LIVE).tool_calls[0]]),
            AIMessage(content="12 jobs."),
        ],
        seen,
    )
    agent = create_agent(
        model=model,
        tools=[get_chart_data],
        middleware=[_build_context_editing_middleware(), EarlierResultsMiddleware()],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t"}}

    await agent.ainvoke({"messages": [_asked_again()]}, config)
    second = await agent.ainvoke({"messages": [_asked_again()]}, config)

    first_call_of_second_turn, last_call = seen[2], seen[3]
    assert _tool_results(first_call_of_second_turn)[0].content == EARLIER_FAILURE_TEXT
    assert first_call_of_second_turn[-1].content == _noted()
    assert [m.content for m in _tool_results(last_call)] == [EARLIER_FAILURE_TEXT, '{"jobs": 12}']

    stored = second["messages"]
    assert [m.content for m in _tool_results(stored)] == [TIMEOUT, '{"jobs": 12}']
    assert [m.content for m in stored if isinstance(m, HumanMessage)] == [
        "Any usage data from ACES?",
        "Any usage data from ACES?",
    ]


async def test_a_real_agent_keeps_the_data_for_a_follow_up_about_an_earlier_answer():
    seen: list[list[Any]] = []
    model = _recording_model([AIMessage(content="In short, 12 jobs.")], seen)
    agent = create_agent(model=model, tools=[], middleware=[EarlierResultsMiddleware()])

    await agent.ainvoke({"messages": [*_earlier_turn(LIVE), _asked_again("Summarize that.")]})

    assert '{"jobs": 12}' in _tool_results(seen[0])[0].content
    assert seen[0][-1].content == _noted("Summarize that.")


async def test_a_real_agent_composes_with_the_signed_in_view():
    """A cleared refusal is rewritten by the gate, and both notes reach the model."""
    seen: list[list[Any]] = []
    model = _recording_model([AIMessage(content="Your accounts.")], seen)
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[EarlierResultsMiddleware(), RequiresAuthMiddleware()],
        context_schema=type(context_for("user123")),
    )
    refused = _earlier_turn(
        "get_my_rp_accounts",
        refusal_text("private_data"),
        status="error",
        artifact=refusal_artifact("private_data"),
    )

    await agent.ainvoke(
        {"messages": [*_earlier_turn(LIVE), *refused, _asked_again()]},
        context=context_for("user123"),
    )

    assert seen[0][-1].content == f"{_noted()}\n\n{SIGNED_IN_NOTE}"


async def test_the_loop_rewrites_after_context_editing():
    """Inside it, so the view sees cleared results and the counter stays faithful."""
    built: dict[str, Any] = {}

    def capture(**kwargs: Any) -> MagicMock:
        built.update(kwargs)
        return MagicMock(ainvoke=AsyncMock(return_value={"messages": [AIMessage(content="ok")]}))

    with (
        patch("src.agent.nodes.tool_calling_loop.create_agent", side_effect=capture),
        patch("src.agent.nodes.tool_calling_loop.get_llm", return_value=MagicMock()),
    ):
        await run_loop({"messages": [_asked_again()], "tool_catalog": {}})

    kinds = [type(m) for m in built["middleware"]]
    assert kinds.index(EarlierResultsMiddleware) > kinds.index(ContextEditingMiddleware)


def test_the_note_says_an_unmarked_preview_still_stands():
    assert "still stands, including a preview" in EARLIER_TURN_NOTE
    assert "from before this message" not in EARLIER_TURN_NOTE


def test_a_write_known_only_by_its_hint_keeps_its_failure_reason():
    failed = ToolMessage(
        content=json.dumps({"error": "event is locked"}), tool_call_id="c1", name="publish_event"
    )

    view = earlier_results_view([failed, HumanMessage(content="Again.")], {"publish_event": False})

    assert "event is locked" in view[0].content


def test_a_write_in_the_live_list_is_never_marked():
    done = ToolMessage(content="{}", tool_call_id="c1", name="register_for_event")

    with patch(
        "src.agent.middleware.earlier_results.live_data_tool_names",
        return_value=frozenset({"register_for_event"}),
    ):
        view = earlier_results_view([done, HumanMessage(content="Again.")])

    assert view[0].content == "{}"


def test_live_tool_config_drops_writes_and_malformed_names(caplog):
    parsed = _parse_live_data_tools('get_chart_data, register_for_event, ["x"]')

    assert parsed == frozenset({"get_chart_data"})
    assert "ignoring" in caplog.text


def test_live_tool_config_warns_about_a_likely_typo(caplog):
    assert _parse_live_data_tools("get_chart_dat") == frozenset({"get_chart_dat"})
    assert "check spelling" in caplog.text


def test_live_tool_config_falls_back_when_nothing_is_usable():
    with patch("src.config.settings.LIVE_DATA_TOOLS", "register_for_event"):
        assert live_data_tool_names() == LIVE_DATA_MCP_TOOL_NAMES


async def test_a_real_agent_tells_the_model_an_empty_reply_was_cut_off():
    from src.agent.stopped_replies import CUT_OFF_NOTE

    seen: list[list[Any]] = []
    model = _recording_model([AIMessage(content="Delta is a GPU system.")], seen)
    agent = create_agent(model=model, tools=[], middleware=[EarlierResultsMiddleware()])
    cut = AIMessage(content="", id="a1", response_metadata={"stopped": True})

    result = await agent.ainvoke(
        {"messages": [HumanMessage("Anvil?", id="h1"), cut, HumanMessage("Delta?", id="h2")]}
    )

    assert seen[0][1].content == CUT_OFF_NOTE
    assert result["messages"][1].content == ""
