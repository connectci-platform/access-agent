"""Compaction must never produce a request with no user turn.

SummarizationMiddleware rebuilds the thread as RemoveMessage(REMOVE_ALL_MESSAGES)
plus the summary plus a positional tail, and partitions purely by index. On a
single-turn request the question is index 0, so it is always summarized away.
Upstream compensates by making the summary a HumanMessage; _build_new_messages
deliberately does not, which left requests carrying no user turn at all and UKY's
vLLM rejected them with "No user query found in messages".

Diagnosed from a Honeycomb `gen_ai.input.messages` payload on eval run
loop-20260912-034618-b05580: assistant x2, tool x1, no user role, summary present.
"""

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from src.agent.nodes.tool_calling_loop import _restore_questions


def _compacted(*tail):
    """What upstream returns: the remove sentinel, the summary, a preserved tail."""
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            AIMessage("Notes from my earlier tool calls in this turn", id="sum-1"),
            *tail,
        ]
    }


def _humans(messages):
    return [m.content for m in messages if isinstance(m, HumanMessage)]


def test_the_question_survives_a_single_turn_compaction():
    # The production failure: one question, a fat fan-out, question evicted.
    result = _compacted(ToolMessage("docs", tool_call_id="c1"))
    state = {"messages": [HumanMessage("who is eligible?", id="q-1")]}
    _restore_questions(result, state)

    assert _humans(result["messages"]) == ["who is eligible?"]


def test_the_question_precedes_the_summary():
    # "Here is what you were asked, here is what you learned" — the reverse
    # reads as notes about a question the model has not seen yet.
    result = _compacted()
    state = {"messages": [HumanMessage("who is eligible?", id="q-1")]}
    _restore_questions(result, state)

    msgs = result["messages"]
    human_at = next(i for i, m in enumerate(msgs) if isinstance(m, HumanMessage))
    summary_at = next(i for i, m in enumerate(msgs) if getattr(m, "id", None) == "sum-1")
    assert human_at < summary_at
    # And after the remove sentinel, which must stay first to clear the thread.
    assert isinstance(msgs[0], RemoveMessage)


def test_every_question_survives_a_multi_turn_thread():
    # Earlier questions are the conversation's structure: "how do I get an
    # allocation" then "what about for a student" only parses as a pair.
    # Restoring all of them also makes a first-vs-last mistake unreachable.
    result = _compacted()
    state = {
        "messages": [
            HumanMessage("how do I get an allocation?", id="q-1"),
            AIMessage("...", id="a-1"),
            HumanMessage("what about for a student?", id="q-2"),
            HumanMessage("and how long does it take?", id="q-3"),
        ],
    }
    _restore_questions(result, state)

    assert _humans(result["messages"]) == [
        "how do I get an allocation?",
        "what about for a student?",
        "and how long does it take?",
    ]


def test_original_message_ids_are_reused():
    # Re-adding an id after REMOVE_ALL_MESSAGES is clean; splicing copies with
    # fresh ids accumulates duplicate human turns across repeated compactions.
    result = _compacted()
    original = HumanMessage("who is eligible?", id="q-1")
    _restore_questions(result, {"messages": [original]})

    restored = next(m for m in result["messages"] if isinstance(m, HumanMessage))
    assert restored is original
    assert restored.id == "q-1"


def test_repeated_compaction_does_not_duplicate_questions():
    result = _compacted()
    state = {"messages": [HumanMessage("who is eligible?", id="q-1")]}
    _restore_questions(result, state)
    _restore_questions(result, state)

    assert _humans(result["messages"]) == ["who is eligible?"]


def test_older_questions_are_restored_even_when_the_tail_kept_a_newer_one():
    """The all-or-nothing guard this replaces silently dropped the older ones.

    The positional tail can happen to preserve the newest question while older
    ones are summarized away. Bailing out on finding any human left the thread
    with a valid user turn — so no 400 — but missing the context that makes a
    follow-up parse. That is the quieter failure the 400 at least announced.
    """
    q3 = HumanMessage("and how long does it take?", id="q-3")
    result = {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            AIMessage("Notes from my earlier tool calls in this turn", id="sum-1"),
            q3,
            ToolMessage("docs", tool_call_id="c1"),
        ]
    }
    state = {
        "messages": [
            HumanMessage("how do I get an allocation?", id="q-1"),
            HumanMessage("what about for a student?", id="q-2"),
            q3,
        ]
    }
    _restore_questions(result, state)

    assert _humans(result["messages"]) == [
        "how do I get an allocation?",
        "what about for a student?",
        "and how long does it take?",
    ]


def test_the_sentinel_stays_first_so_the_reducer_keeps_the_questions():
    """Index 1 is the only correct insertion point.

    langgraph's add_messages discards everything up to and including
    REMOVE_ALL_MESSAGES, so anything placed at index 0 is thrown away.
    """
    result = _compacted(ToolMessage("docs", tool_call_id="c1"))
    _restore_questions(result, {"messages": [HumanMessage("who is eligible?", id="q-1")]})

    msgs = result["messages"]
    assert isinstance(msgs[0], RemoveMessage)
    assert isinstance(msgs[1], HumanMessage)


def test_an_untouched_result_is_left_alone():
    # When upstream already preserved a human turn there is nothing to repair,
    # and re-inserting would duplicate it.
    kept = HumanMessage("who is eligible?", id="q-1")
    result = {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), kept, AIMessage("sum")]}
    _restore_questions(result, {"messages": [kept]})

    assert _humans(result["messages"]) == ["who is eligible?"]
