"""A heavy tool fan-out must not cost the model the user's question.

Production bug (2026-09-10, deterministic 3/3): a single-turn request that
fanned out over ~14 `get_resource_hardware` calls produced enough tool-result
context to cross ``SUMMARIZATION_TRIGGER_TOKENS``. ``SummarizationMiddleware``
keeps only the most recent ``SUMMARIZATION_KEEP_TOKENS`` **by recency**
(``conversation_messages[cutoff_index:]``), so the oldest message — the user's
question — was evicted. It then injected the summary as
``HumanMessage("Here is a summary of the conversation to date: ...")``.

The model, shown a human narrating a conversation that never happened and no
actual question, replied "I don't have access to the previous conversation
history. What would you like help with?" on a FIRST-TURN request. That is a
rational reading of a broken transcript, not a model failure.

These tests pin the two invariants that were violated.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent.nodes.tool_calling_loop import _build_context_editing_middleware


def _fat_fanout_messages(n_calls: int | None = None, chars_per_result: int = 9000) -> list:
    """A single-turn thread: one question, then n fat tool results.

    Mirrors the production shape at the sizes UKY's retrieve-docs actually
    returns. Measured 2026-09-10: individual documents run 5-17KB (one outlier
    at 99KB), a typical whole response is ~33KB, so 9000 chars/result is
    mid-range.

    The CALL COUNT is derived from the configured trigger rather than fixed at
    the 14 that crossed it in production, so these tests keep exercising the
    over-threshold path when the thresholds are retuned. (They did not: raising
    the trigger to 70,000 left the old fixed fixture at 31,510 tokens, below
    both the summarization and context-edit triggers, and the tests failed
    while the behaviour they cover was fine.)
    """
    if n_calls is None:
        from src.config import settings

        # 4 chars/token, doubled for comfortable margin over the trigger.
        n_calls = max(14, (settings.SUMMARIZATION_TRIGGER_TOKENS * 4 * 2) // chars_per_result)
    messages: list = [HumanMessage(content="What GPU types do ACCESS resources have?", id="q-1")]
    for i in range(n_calls):
        messages.append(
            AIMessage(
                content="",
                id=f"ai-{i}",
                tool_calls=[
                    {"name": "get_resource_hardware", "args": {"id": f"res{i}"}, "id": f"call-{i}"}
                ],
            )
        )
        messages.append(
            ToolMessage(content="x" * chars_per_result, tool_call_id=f"call-{i}", id=f"tm-{i}")
        )
    return messages


def _count_tokens(messages: list) -> int:
    """Crude char/4 estimate — enough to trip a token trigger deterministically."""
    return sum(len(str(getattr(m, "content", ""))) for m in messages) // 4


class TestContextEditingKeepsTheQuestion:
    def test_fat_fanout_would_exceed_the_summarization_trigger(self):
        """Establish the premise: this thread is big enough to trigger
        compaction, which is what evicted the question in production."""
        from src.config import settings

        assert _count_tokens(_fat_fanout_messages()) > settings.SUMMARIZATION_TRIGGER_TOKENS

    def test_context_editing_clears_tool_output_not_the_question(self):
        """ClearToolUsesEdit trims oldest TOOL results and never touches the
        human turn, so the question survives a heavy fan-out."""
        messages = _fat_fanout_messages()
        middleware = _build_context_editing_middleware()
        edit = middleware.edits[0]

        edit.apply(messages, count_tokens=_count_tokens)

        humans = [m for m in messages if isinstance(m, HumanMessage)]
        assert len(humans) == 1
        assert humans[0].content == "What GPU types do ACCESS resources have?"
        assert humans[0].id == "q-1"

    def test_clearing_reclaims_context(self):
        """The edit must actually shrink the thread, or compaction still fires
        and we are back to the original bug."""
        messages = _fat_fanout_messages()
        before = _count_tokens(messages)
        edit = _build_context_editing_middleware().edits[0]

        edit.apply(messages, count_tokens=_count_tokens)

        assert _count_tokens(messages) < before

    def test_message_ids_survive_editing(self):
        """`_build_tool_results` pairs ToolMessages to tool_calls by id, and the
        multi-turn eval slices on message_id. Editing must preserve identity —
        ClearToolUsesEdit uses model_copy, so ids are kept."""
        messages = _fat_fanout_messages()
        ids_before = [m.id for m in messages]
        edit = _build_context_editing_middleware().edits[0]

        edit.apply(messages, count_tokens=_count_tokens)

        assert [m.id for m in messages] == ids_before

    def test_recent_tool_results_are_kept_verbatim(self):
        """Trimming the oldest is fine; the model still needs the most recent
        results to answer from."""
        messages = _fat_fanout_messages()
        edit = _build_context_editing_middleware().edits[0]

        edit.apply(messages, count_tokens=_count_tokens)

        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        assert tool_messages, "tool results must not be wholly discarded"
        # The last `keep` results retain their original payload.
        assert any(len(str(m.content)) > 1000 for m in tool_messages)


class TestSummaryFraming:
    def test_summary_is_not_framed_as_a_user_turn(self):
        """The upstream default injects the summary as
        HumanMessage("Here is a summary of the conversation to date"). On a
        single-turn request that invents a conversation the user never had and
        is what prompted "I don't have access to the previous conversation
        history". Our subclass must not narrate a fake user turn.
        """
        from src.agent.nodes.tool_calling_loop import _FlaggingSummarizationMiddleware

        built = _FlaggingSummarizationMiddleware._build_new_messages("SUMMARY BODY")

        assert len(built) == 1
        message = built[0]
        assert not isinstance(message, HumanMessage), (
            "a summary is the agent's own recall, not something the user said"
        )
        assert "conversation to date" not in str(message.content)
        assert "SUMMARY BODY" in str(message.content)
