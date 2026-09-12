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


def test_heavy_fanout_no_longer_trips_summarization():
    # The observed production case: 14 calls of ~9k chars. Raw this is ~31.5k
    # against a 24k trigger; edited it is ~9k and should not fire.
    msgs = _fanout(14, 9000)
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
    # Summarization must remain a real backstop. Editing keeps the most recent
    # CONTEXT_EDIT_KEEP_TOOL_RESULTS verbatim, so enough large recent results
    # still exceed the trigger even after editing.
    huge = _fanout(settings.CONTEXT_EDIT_KEEP_TOOL_RESULTS, 40000)
    assert _edit_aware_token_counter()(huge) > settings.SUMMARIZATION_TRIGGER_TOKENS


def test_small_threads_are_unaffected():
    msgs = _fanout(2, 500)
    assert _edit_aware_token_counter()(msgs) == count_tokens_approximately(msgs)
