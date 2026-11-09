"""Compaction leaves the turn being answered alone and never shows as an answer (D8-3055).

QA saw compaction fire inside one turn after five identical searches. It kept two
of the five results, and its note, holding only upstream's "Previous conversation
was too long to summarize." placeholder, reached the user as an answer bubble.
"""

from typing import Any

from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.nodes.tool_calling_loop import (
    DO_NOT_RENDER_ID_PREFIX,
    _edit_aware_token_counter,
    _FlaggingSummarizationMiddleware,
)
from src.config import settings


def _turn(question: str, prefix: str) -> list[Any]:
    """A turn of many searches whose four kept results come to ~40k tokens, under
    the ceiling, so it is kept whole."""
    messages: list[Any] = [HumanMessage(question, id=f"{prefix}-q")]
    for i in range(settings.CONTEXT_EDIT_KEEP_TOOL_RESULTS + 6):
        call = {"name": "search_access_documents", "args": {"query": "q"}, "id": f"{prefix}{i}"}
        messages.append(AIMessage("", id=f"{prefix}-ai-{i}", tool_calls=[call]))
        messages.append(
            ToolMessage("Y" * 40_000, tool_call_id=f"{prefix}{i}", id=f"{prefix}-t-{i}")
        )
    return messages


def _earlier(question: str, prefix: str) -> list[Any]:
    """An earlier turn whose long answer, which context editing can't clear, takes
    the thread over the trigger."""
    return [
        HumanMessage(question, id=f"{prefix}-q"),
        AIMessage("Y" * 160_000, id=f"{prefix}-answer"),
    ]


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
    thread = [*_earlier("what is Expanse?", "before"), *current]
    middleware = _middleware()
    # Upstream would cut inside the current turn, which is what this guards.
    assert SummarizationMiddleware._determine_cutoff_index(middleware, thread) > len(thread) - len(
        current
    )

    result = middleware.before_model({"messages": thread}, None)

    assert result is not None, "summarization did not fire; fixture is undersized"
    kept = [m.id for m in result["messages"][1:]]
    assert kept[-len(current) :] == [m.id for m in current]
    assert "before-answer" not in kept


async def test_the_async_path_keeps_the_current_turn_too():
    current = _turn("my username shows pending", "now")
    thread = [*_earlier("what is Expanse?", "before"), *current]

    result = await _middleware().abefore_model({"messages": thread}, None)

    assert result is not None
    assert [m.id for m in result["messages"][-len(current) :]] == [m.id for m in current]


def test_only_questions_and_an_earlier_summary_are_not_summarized_again():
    thread = [HumanMessage("first", id="q-0"), _summary("Y" * 160_000), *_turn("pending?", "now")]

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
    thread = [*_earlier("what is Expanse?", "before"), *_turn("pending?", "now")]

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


def test_a_single_turn_too_big_for_the_window_is_compacted_as_a_last_resort():
    # Six ~25k-token results: context editing keeps four whole, ~100k, which with
    # the request's fixed overhead would overflow the model's window.
    messages: list[Any] = [HumanMessage("list everything on Anvil", id="q")]
    for i in range(6):
        call = {"name": "list_all_software", "args": {"page": i}, "id": f"c{i}"}
        messages.append(AIMessage("", id=f"ai-{i}", tool_calls=[call]))
        messages.append(ToolMessage("Y" * 100_000, tool_call_id=f"c{i}", id=f"t-{i}"))
    counter = _edit_aware_token_counter()
    assert counter(messages) > settings.SUMMARIZATION_TURN_CEILING_TOKENS

    result = _middleware().before_model({"messages": messages}, None)

    assert result is not None
    after = result["messages"][1:]
    assert counter(after) < settings.SUMMARIZATION_TURN_CEILING_TOKENS
    assert [m.content for m in after if isinstance(m, HumanMessage)] == ["list everything on Anvil"]


def test_a_follow_up_keeps_a_summary_of_the_earlier_answer():
    """One ordinary tool result in the last earlier turn used to leave upstream's
    4,000-token trim empty, so its placeholder replaced every earlier answer."""
    earlier = [
        HumanMessage("what GPUs does Jetstream2 have?", id="q-0"),
        AIMessage("", id="ai-0", tool_calls=[{"name": "s", "args": {}, "id": "c0"}]),
        ToolMessage("Y" * 20_000, tool_call_id="c0", id="t-0"),
        AIMessage("Jetstream2's g3.xl flavor has one A100.", id="ans-0"),
    ]
    middleware = _FlaggingSummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([AIMessage("g3.xl has one A100")])),
        token_counter=_edit_aware_token_counter(),
        trigger=("tokens", 1000),
        keep=("tokens", 100),
    )

    trimmed = middleware._trim_messages_for_summary(earlier)
    result = middleware.before_model(
        {"messages": [*earlier, HumanMessage("and its price?", id="q-1")]}, None
    )

    assert any(m.id == "ans-0" for m in trimmed)
    assert result is not None
    summaries = [m for m in result["messages"] if m.id and m.id.startswith(DO_NOT_RENDER_ID_PREFIX)]
    assert len(summaries) == 1
    assert "g3.xl has one A100" in summaries[0].text


def test_a_trim_error_falls_back_to_the_newest_messages(monkeypatch):
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("cannot split")

    monkeypatch.setattr("src.agent.nodes.tool_calling_loop.trim_messages", broken)
    messages = [AIMessage(f"answer {i}", id=f"a{i}") for i in range(20)]

    trimmed = _middleware()._trim_messages_for_summary(messages)

    assert [m.id for m in trimmed] == [f"a{i}" for i in range(5, 20)]
