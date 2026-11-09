"""A stopped turn is saved into the thread, and Retry/Edit replace it (D8-3047)."""

import asyncio

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.graph import create_checkpointed_graph
from src.api import stopped_turns
from src.api.stopped_turns import (
    drop_unanswered_turn,
    is_stopped,
    keep_stopped_turn,
    stopped_messages,
    stopped_turn_saved,
    unanswered_turn,
)

CONFIG = {"configurable": {"thread_id": "t"}}


def _stopped(text: str = "part", id: str = "a2") -> AIMessage:
    return AIMessage(content=text, id=id, response_metadata={"stopped": True})


async def _thread(saver: InMemorySaver) -> list:
    state = await create_checkpointed_graph(saver).aget_state(CONFIG)
    return (state.values or {}).get("messages", [])


async def _seed(saver: InMemorySaver, messages: list) -> None:
    graph = create_checkpointed_graph(saver)
    await graph.aupdate_state(CONFIG, {"messages": messages}, as_node="suggest_follow_ups")


def test_is_stopped():
    assert is_stopped(_stopped())
    assert not is_stopped(AIMessage(content="done", id="a"))
    assert not is_stopped(HumanMessage(content="q", id="h"))


def test_stopped_messages_follow_a_saved_question():
    saved = [HumanMessage(content="q", id="h")]
    streamed = {
        "r1": AIMessageChunk(content="Here is", id="r1"),
        "r2": AIMessageChunk(content="  ", id="r2"),
        "": AIMessageChunk(content="no id", id=None),
    }
    out = stopped_messages(saved, "q", streamed)
    assert [(m.id, m.content) for m in out] == [("r1", "Here is")]
    assert all(is_stopped(m) for m in out)


def test_stopped_messages_with_no_text_is_an_empty_stopped_answer():
    out = stopped_messages([HumanMessage(content="q", id="h")], "q", {})
    assert len(out) == 1
    assert out[0].content == ""
    assert is_stopped(out[0])


def test_stopped_messages_add_the_question_when_it_wasnt_saved():
    saved = [HumanMessage(content="old", id="h0"), AIMessage(content="old answer", id="a0")]
    out = stopped_messages(saved, "new", {"r1": AIMessageChunk(content="x", id="r1")})
    assert isinstance(out[0], HumanMessage)
    assert out[0].content == "new"
    assert out[1].id == "r1"
    assert stopped_messages([], "new", {})[0].content == "new"


def test_unanswered_turn():
    h0, a0 = HumanMessage(content="q0", id="h0"), AIMessage(content="a0", id="a0")
    h1 = HumanMessage(content="q1", id="h1")
    assert unanswered_turn([]) == []
    assert unanswered_turn([h0, a0]) == []
    assert unanswered_turn([h0, a0, h1]) == ["h1"]
    assert unanswered_turn([h0, a0, h1, _stopped()]) == ["h1", "a2"]
    assert unanswered_turn([h0, a0, h1, _stopped(), AIMessage(content="late", id="a3")]) == []


async def test_keep_stopped_turn_saves_the_partial_answer_after_the_question():
    saver = InMemorySaver()
    await _seed(saver, [HumanMessage(content="q", id="h")])
    keep_stopped_turn(saver, "t", "q", {"r1": AIMessageChunk(content="Here is", id="r1")})
    await stopped_turn_saved("t")
    thread = await _thread(saver)
    assert [m.id for m in thread] == ["h", "r1"]
    assert is_stopped(thread[-1])
    # Nothing left to run, so the head doesn't read as a breakpoint.
    state = await create_checkpointed_graph(saver).aget_state(CONFIG)
    assert state.next == ()
    assert "t" not in stopped_turns._pending


async def test_keep_stopped_turn_without_a_checkpointer_does_nothing():
    keep_stopped_turn(None, "t", "q", {})
    assert "t" not in stopped_turns._pending
    await stopped_turn_saved("t")


async def test_a_failed_save_is_logged_not_raised(caplog):
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("db down")

    keep_stopped_turn(Broken(), "t", "q", {})
    await stopped_turn_saved("t")
    assert "Saving a stopped turn failed" in caplog.text


async def test_a_later_save_on_the_thread_stays_pending_when_an_earlier_one_ends():
    saver = InMemorySaver()
    gate = asyncio.Event()
    real = stopped_turns._save

    async def _slow(*args):
        await gate.wait()
        await real(*args)

    stopped_turns._pending.clear()
    keep_stopped_turn(saver, "t", "q", {})
    first = stopped_turns._pending["t"]
    stopped_turns._save = _slow  # type: ignore[assignment]
    try:
        keep_stopped_turn(saver, "t", "q2", {})
        second = stopped_turns._pending["t"]
        await first
        assert stopped_turns._pending.get("t") is second
        gate.set()
        await stopped_turn_saved("t")
    finally:
        stopped_turns._save = real  # type: ignore[assignment]
    assert "t" not in stopped_turns._pending


async def test_drop_unanswered_turn_removes_the_stopped_question_and_answer():
    saver = InMemorySaver()
    await _seed(
        saver,
        [
            HumanMessage(content="q0", id="h0"),
            AIMessage(content="a0", id="a0"),
            HumanMessage(content="q1", id="h1"),
            _stopped(),
        ],
    )
    await drop_unanswered_turn(saver, "t")
    assert [m.id for m in await _thread(saver)] == ["h0", "a0"]


async def test_drop_unanswered_turn_keeps_an_answered_thread():
    saver = InMemorySaver()
    await _seed(saver, [HumanMessage(content="q0", id="h0"), AIMessage(content="a0", id="a0")])
    await drop_unanswered_turn(saver, "t")
    assert [m.id for m in await _thread(saver)] == ["h0", "a0"]
    await drop_unanswered_turn(None, "t")
