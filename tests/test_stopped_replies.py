"""The model is told a reply was cut off; the stored reply doesn't change (D8-3047)."""

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.stopped_replies import CUT_OFF_NOTE, is_stopped, stopped_replies_view

META = {"stopped": True, "reason": "user"}


def test_a_thread_with_no_cut_off_reply_is_unchanged():
    thread = [HumanMessage(content="q", id="h"), AIMessage(content="a", id="a")]
    assert stopped_replies_view(thread) is thread


def test_only_an_empty_cut_off_reply_gets_the_note():
    """A note on partial text sent Qwen into a reasoning loop on the next
    question; an empty reply still needs one so it doesn't read as blank."""
    part = AIMessage(content="Anvil is ", id="a", response_metadata=META)
    empty = AIMessage(content="", id="b", response_metadata=META)
    blocks = AIMessage(content=[{"type": "text", "text": "x"}], id="c", response_metadata=META)
    empty_blocks = AIMessage(content=[], id="d", response_metadata=META)
    thread = [HumanMessage(content="q", id="h"), part, blocks]
    assert stopped_replies_view(thread) is thread

    view = stopped_replies_view(
        [HumanMessage(content="q", id="h"), part, empty, blocks, empty_blocks]
    )
    assert view[1] is part
    assert view[2].content == CUT_OFF_NOTE
    assert view[3] is blocks
    assert view[4].content == CUT_OFF_NOTE
    assert empty.content == ""
    assert is_stopped(view[2])
