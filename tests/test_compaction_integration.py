"""Drive the real SummarizationMiddleware and assert the request stays valid.

Every other test of this behaviour exercises `_restore_questions` directly or
against a hand-built result dict. That is how the original bug shipped: the
pieces were tested in isolation and the composition was not. These run the
actual middleware — real partitioning, real summary generation, real return
shape — and assert on what it hands back.

The bug: upstream rebuilds the thread as RemoveMessage(ALL) + summary + a
positional tail, so a single-turn question at index 0 is always summarized
away. With `_build_new_messages` emitting an AIMessage rather than upstream's
HumanMessage, nothing user-role survives and UKY's vLLM rejects the next
request with `400 No user query found in messages`.
"""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.nodes.tool_calling_loop import (
    _edit_aware_token_counter,
    _FlaggingSummarizationMiddleware,
)
from src.config import settings


def _oversized_thread(question: str = "who is eligible to use ACCESS?") -> list:
    """A thread big enough to trip summarization *after* context editing.

    Sizing matters: editing keeps the most recent CONTEXT_EDIT_KEEP_TOOL_RESULTS
    verbatim and blanks the rest, and the trigger is measured on the edited size
    (see _edit_aware_token_counter). So the kept window alone has to exceed the
    trigger, or summarization never fires and the test passes vacuously.
    """
    keep = settings.CONTEXT_EDIT_KEEP_TOOL_RESULTS
    per_result = (settings.SUMMARIZATION_TRIGGER_TOKENS * 4) // keep + 8000
    messages: list = [HumanMessage(question, id="q-1")]
    for i in range(keep + 6):
        messages.append(
            AIMessage("", id=f"ai-{i}", tool_calls=[{"name": "s", "args": {}, "id": f"c{i}"}])
        )
        messages.append(ToolMessage("X" * per_result, tool_call_id=f"c{i}", id=f"t-{i}"))
    return messages


def _middleware() -> _FlaggingSummarizationMiddleware:
    return _FlaggingSummarizationMiddleware(
        model=GenericFakeChatModel(messages=iter([AIMessage("a summary")] * 50)),
        token_counter=_edit_aware_token_counter(),
        trigger=("tokens", settings.SUMMARIZATION_TRIGGER_TOKENS),
        keep=("tokens", settings.SUMMARIZATION_KEEP_TOKENS),
    )


def test_a_compacted_request_still_carries_a_user_turn():
    """The end-to-end property vLLM actually requires."""
    result = _middleware().before_model({"messages": _oversized_thread()}, None)

    # Guard against a vacuous pass: if compaction did not fire there is nothing
    # to assert, and the fixture needs resizing.
    assert result is not None, "summarization did not fire; fixture is undersized"
    assert any(isinstance(m, HumanMessage) for m in result["messages"])


def test_the_original_question_text_survives_compaction():
    question = "what are the different queues on Delta GPU?"
    result = _middleware().before_model({"messages": _oversized_thread(question)}, None)

    assert result is not None
    humans = [m.content for m in result["messages"] if isinstance(m, HumanMessage)]
    assert humans == [question]


def test_every_question_survives_a_multi_turn_compaction():
    thread = _oversized_thread()
    thread.append(HumanMessage("and how much do they cost?", id="q-2"))
    result = _middleware().before_model({"messages": thread}, None)

    assert result is not None
    humans = [m.content for m in result["messages"] if isinstance(m, HumanMessage)]
    assert "who is eligible to use ACCESS?" in humans
    assert "and how much do they cost?" in humans


def test_a_small_thread_is_left_alone():
    """Compaction must not fire on ordinary traffic."""
    small = [
        HumanMessage("who is eligible?", id="q-1"),
        AIMessage("", id="ai-0", tool_calls=[{"name": "s", "args": {}, "id": "c0"}]),
        ToolMessage("a short answer", tool_call_id="c0", id="t-0"),
    ]
    assert _middleware().before_model({"messages": small}, None) is None
