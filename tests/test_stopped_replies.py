"""The model is told a reply was cut off; the stored reply doesn't change (D8-3047)."""

from langchain_core.messages import AIMessage, HumanMessage

from src.agent.stopped_replies import CUT_OFF_NOTE, is_stopped, stopped_replies_view

META = {"stopped": True, "reason": "user"}


def test_a_thread_with_no_cut_off_reply_is_unchanged():
    thread = [HumanMessage(content="q", id="h"), AIMessage(content="a", id="a")]
    assert stopped_replies_view(thread) is thread


def test_a_cut_off_reply_gets_the_note_and_the_stored_one_keeps_its_text():
    part = AIMessage(content="Anvil is ", id="a", response_metadata=META)
    empty = AIMessage(content="", id="b", response_metadata=META)
    blocks = AIMessage(content=[{"type": "text", "text": "x"}], id="c", response_metadata=META)
    view = stopped_replies_view([HumanMessage(content="q", id="h"), part, empty, blocks])
    assert view[1].content == f"Anvil is\n\n{CUT_OFF_NOTE}"
    assert view[2].content == CUT_OFF_NOTE
    assert view[3].content[-1] == {"type": "text", "text": CUT_OFF_NOTE}
    assert part.content == "Anvil is "
    assert is_stopped(view[1])
