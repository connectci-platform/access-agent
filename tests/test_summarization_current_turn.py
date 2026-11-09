"""Compaction leaves the turn being answered alone and never shows as an answer (D8-3055).

QA saw compaction fire inside one turn after five identical searches. It kept two
of the five results, and its note, holding only upstream's "Previous conversation
was too long to summarize." placeholder, reached the user as an answer bubble.
"""

from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.nodes.tool_calling_loop import (
    DO_NOT_RENDER_ID_PREFIX,
    _edit_aware_token_counter,
    _FlaggingSummarizationMiddleware,
)
from src.config import settings


def _turn(question: str, prefix: str) -> list[Any]:
    """A turn big enough to trip summarization by itself, after context editing."""
    keep = settings.CONTEXT_EDIT_KEEP_TOOL_RESULTS
    per_result = (settings.SUMMARIZATION_TRIGGER_TOKENS * 4) // keep + 8000
    messages: list[Any] = [HumanMessage(question, id=f"{prefix}-q")]
    for i in range(keep + 6):
        call = {"name": "search_access_documents", "args": {"query": "q"}, "id": f"{prefix}{i}"}
        messages.append(AIMessage("", id=f"{prefix}-ai-{i}", tool_calls=[call]))
        messages.append(
            ToolMessage("X" * per_result, tool_call_id=f"{prefix}{i}", id=f"{prefix}-t-{i}")
        )
    return messages


def _middleware() -> _FlaggingSummarizationMiddleware:
    return _FlaggingSummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([AIMessage("a summary")] * 50)),
        token_counter=_edit_aware_token_counter(),
        trigger=("tokens", settings.SUMMARIZATION_TRIGGER_TOKENS),
        keep=("tokens", settings.SUMMARIZATION_KEEP_TOKENS),
    )


def _summary(text: str = "earlier notes") -> AIMessage:
    return AIMessage(text, id="sum-1", additional_kwargs={"lc_source": "summarization"})


def test_every_message_of_the_current_turn_is_kept():
    current = _turn("my username shows pending", "now")
    thread = [*_turn("what is Expanse?", "before"), *current]

    result = _middleware().before_model({"messages": thread}, None)

    assert result is not None, "summarization did not fire; fixture is undersized"
    kept = [m.id for m in result["messages"][1:]]
    assert kept[-len(current) :] == [m.id for m in current]
    assert not any(m.id.startswith("before-t-") for m in result["messages"][1:])


async def test_the_async_path_keeps_the_current_turn_too():
    current = _turn("my username shows pending", "now")
    thread = [*_turn("what is Expanse?", "before"), *current]

    result = await _middleware().abefore_model({"messages": thread}, None)

    assert result is not None
    assert [m.id for m in result["messages"][-len(current) :]] == [m.id for m in current]


def test_a_single_long_turn_is_not_compacted():
    assert _middleware().before_model({"messages": _turn("pending?", "now")}, None) is None


def test_only_questions_and_an_earlier_summary_are_not_summarized_again():
    thread = [HumanMessage("first", id="q-0"), _summary(), *_turn("pending?", "now")]

    assert _middleware().before_model({"messages": thread}, None) is None


def test_upstream_s_empty_placeholder_adds_nothing():
    for placeholder in (
        "Previous conversation was too long to summarize.",
        "No previous conversation history.",
    ):
        assert _FlaggingSummarizationMiddleware._build_new_messages(placeholder) == []


def test_a_compaction_with_no_summary_keeps_the_questions_and_adds_no_note(monkeypatch):
    middleware = _middleware()
    monkeypatch.setattr(
        middleware,
        "_create_summary",
        lambda messages: "Previous conversation was too long to summarize.",
    )
    thread = [*_turn("what is Expanse?", "before"), *_turn("pending?", "now")]

    result = middleware.before_model({"messages": thread}, None)

    assert result is not None
    texts = [m.text for m in result["messages"][1:]]
    assert not any("too long to summarize" in t for t in texts)
    humans = [m.content for m in result["messages"] if isinstance(m, HumanMessage)]
    assert humans == ["what is Expanse?", "pending?"]


def test_the_summary_is_marked_for_clients_not_to_render():
    (message,) = _FlaggingSummarizationMiddleware._build_new_messages("SUMMARY BODY")

    # The prefix the full-screen client filters on (src/lib/ensure-tool-responses.ts).
    assert DO_NOT_RENDER_ID_PREFIX == "do-not-render-"
    assert message.id.startswith(DO_NOT_RENDER_ID_PREFIX)
    assert message.additional_kwargs == {"lc_source": "summarization"}
