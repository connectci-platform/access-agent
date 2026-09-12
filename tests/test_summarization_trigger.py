"""Summarization must measure the thread as the model will receive it.

ContextEditingMiddleware hooks `wrap_model_call`; SummarizationMiddleware hooks
`before_model`. Different hooks, so the middleware list order does not sequence
them — the edits rewrite one request via `request.override` and never reach
state. Summarization therefore counted the raw thread and fired on fan-outs the
edits had already reclaimed, evicting the user's question on the way (the
"backstop" firing as a tripwire).

The fix is an edit-aware token counter. These tests pin the property, not the
implementation: the trigger decision must reflect the edited size.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately

from src.agent.nodes.tool_calling_loop import _edit_aware_token_counter
from src.config import settings


def _fanout(calls: int, result_chars: int) -> list:
    """A single-turn thread with `calls` tool results, the production shape."""
    msgs: list = [HumanMessage("who is eligible to use ACCESS?")]
    for i in range(calls):
        msgs.append(AIMessage("", tool_calls=[{"name": "search", "args": {}, "id": f"c{i}"}]))
        msgs.append(ToolMessage("X" * result_chars, tool_call_id=f"c{i}"))
    return msgs


def test_editing_is_what_keeps_a_big_fanout_under_the_trigger():
    """Sized off the configured trigger, not a fixed magnitude.

    The point is the relationship: a fan-out large enough to trip summarization
    on its raw size must fall under it once edited. Hardcoding token counts ties
    the test to whatever the thresholds happen to be — this one keeps holding
    when they are retuned.
    """
    per_result = 9000
    # Enough calls that the raw thread exceeds the trigger with margin.
    calls = (settings.SUMMARIZATION_TRIGGER_TOKENS * 4 * 2) // per_result
    msgs = _fanout(calls, per_result)

    assert count_tokens_approximately(msgs) > settings.SUMMARIZATION_TRIGGER_TOKENS
    assert _edit_aware_token_counter()(msgs) < settings.SUMMARIZATION_TRIGGER_TOKENS


def test_counting_does_not_mutate_the_caller_s_messages():
    # The counter applies edits to decide; it must not strip the real thread,
    # which the model still needs to answer from.
    msgs = _fanout(14, 9000)
    before = [str(m.content) for m in msgs]
    _edit_aware_token_counter()(msgs)
    assert [str(m.content) for m in msgs] == before


def test_a_genuinely_long_thread_still_trips_summarization():
    """Summarization must stay a real backstop, not be disabled by editing.

    Editing keeps the most recent CONTEXT_EDIT_KEEP_TOOL_RESULTS verbatim, so
    results large enough in that preserved window still exceed the trigger.
    """
    # Size each preserved result so the kept window alone clears the trigger.
    per_result = (settings.SUMMARIZATION_TRIGGER_TOKENS * 4) // (
        settings.CONTEXT_EDIT_KEEP_TOOL_RESULTS
    ) + 4000
    huge = _fanout(settings.CONTEXT_EDIT_KEEP_TOOL_RESULTS, per_result)
    assert _edit_aware_token_counter()(huge) > settings.SUMMARIZATION_TRIGGER_TOKENS


def test_small_threads_are_unaffected():
    msgs = _fanout(2, 500)
    assert _edit_aware_token_counter()(msgs) == count_tokens_approximately(msgs)
