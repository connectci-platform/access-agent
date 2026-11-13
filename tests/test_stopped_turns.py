"""A stopped turn is saved into the thread, and Retry/Edit replace it (D8-3047)."""

import asyncio
import contextlib

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from src.agent.graph import create_checkpointed_graph
from src.api import stopped_turns
from src.api.stopped_turns import (
    answer_delivered,
    drain_stopped_turns,
    drop_unanswered_turn,
    is_stopped,
    keep_stopped_turn,
    keeping_cut_off_turn,
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
    # A chunk with no id still keeps its text, under an id of its own.
    assert [m.content for m in out] == ["Here is", "no id"]
    assert out[0].id == "r1"
    assert out[1].id
    assert all(is_stopped(m) for m in out)
    assert out[0].response_metadata == {"stopped": True, "reason": "user"}


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


def test_stopped_messages_carry_the_reason():
    out = stopped_messages([HumanMessage(content="q", id="h")], "q", {}, "timeout")
    assert out[0].response_metadata == {"stopped": True, "reason": "timeout"}


def test_a_different_question_left_at_the_end_isnt_taken_for_this_one():
    """A failed run before this fix left its question unanswered at the end."""
    saved = [HumanMessage(content="old question", id="h0")]
    out = stopped_messages(saved, "new question", {})
    assert isinstance(out[0], HumanMessage)
    assert out[0].content == "new question"
    # The same text is this run's question, saved when it started.
    assert len(stopped_messages([HumanMessage(content=" q ", id="h")], "q", {})) == 1


async def test_waiting_for_a_save_is_bounded(caplog):
    gate = asyncio.Event()
    stopped_turns._pending["t"] = asyncio.ensure_future(gate.wait())
    try:
        await stopped_turn_saved("t", timeout=0.01)
        assert "going on without it" in caplog.text
        assert not stopped_turns._pending["t"].done()
    finally:
        gate.set()
        await stopped_turns._pending.pop("t")


async def test_drain_waits_for_saves_still_running():
    await drain_stopped_turns()  # nothing pending
    done: list[str] = []

    async def _save():
        await asyncio.sleep(0.01)
        done.append("saved")

    stopped_turns._pending["t"] = asyncio.ensure_future(_save())
    try:
        await drain_stopped_turns(timeout=1)
    finally:
        stopped_turns._pending.clear()
    assert done == ["saved"]


def test_answer_delivered_needs_an_answer_and_its_message():
    answer = AIMessage("a", id="ans")
    assert answer_delivered({"final_answer": "a", "messages": [answer]})
    assert not answer_delivered({})


async def _drive(events, saver, final_state=None, **kw):
    streamed = {"r1": AIMessageChunk(content="Half", id="r1")}
    out = []
    async for e in keeping_cut_off_turn(
        events,
        checkpointer=saver,
        thread_id="t",
        query="q",
        final_state=final_state if final_state is not None else {},
        streamed=streamed,
        **kw,
    ):
        out.append(e)
    return out


async def test_a_failed_run_keeps_its_turn_marked_error():
    saver = InMemorySaver()
    await _seed(saver, [HumanMessage(content="q", id="h")])

    async def _boom():
        yield 1
        raise RuntimeError("vLLM 500")

    with contextlib.suppress(RuntimeError):
        await _drive(_boom(), saver)
    await stopped_turn_saved("t")
    thread = await _thread(saver)
    assert [m.id for m in thread] == ["h", "r1"]
    assert thread[-1].response_metadata["reason"] == "error"


async def test_a_cancel_past_the_deadline_is_a_timeout(monkeypatch):
    import time

    saver = InMemorySaver()
    await _seed(saver, [HumanMessage(content="q", id="h")])
    monkeypatch.setattr(
        stopped_turns, "get_turn_capture", lambda: {"turn_deadline": time.monotonic() - 1}
    )

    async def _cancelled():
        yield 1
        raise asyncio.CancelledError

    with contextlib.suppress(asyncio.CancelledError):
        await _drive(_cancelled(), saver)
    await stopped_turn_saved("t")
    assert (await _thread(saver))[-1].response_metadata["reason"] == "timeout"


async def test_a_cancel_before_the_deadline_is_a_stop(monkeypatch):
    import time

    saver = InMemorySaver()
    await _seed(saver, [HumanMessage(content="q", id="h")])
    monkeypatch.setattr(
        stopped_turns, "get_turn_capture", lambda: {"turn_deadline": time.monotonic() + 60}
    )

    async def _cancelled():
        raise asyncio.CancelledError
        yield

    with contextlib.suppress(asyncio.CancelledError):
        await _drive(_cancelled(), saver)
    await stopped_turn_saved("t")
    assert (await _thread(saver))[-1].response_metadata["reason"] == "user"


async def test_a_stop_while_waiting_for_the_last_save_still_keeps_the_question():
    """The run's question isn't saved yet, so it goes in with an empty answer."""
    saver = InMemorySaver()
    await _seed(saver, [HumanMessage(content="old", id="h0"), AIMessage(content="a", id="a0")])
    earlier = asyncio.ensure_future(asyncio.sleep(10))
    stopped_turns._pending["t"] = earlier

    async def _never():
        yield 1

    task = asyncio.ensure_future(_drive(_never(), saver))
    await asyncio.sleep(0.01)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    earlier.cancel()
    await stopped_turn_saved("t")
    thread = await _thread(saver)
    assert [m.content for m in thread] == ["old", "a", "q", "Half"]
    assert is_stopped(thread[-1])


async def test_a_delivered_answer_isnt_kept_again():
    saver = InMemorySaver()
    delivered = {"final_answer": "a", "messages": [AIMessage(content="a", id="ans")]}

    async def _cancelled():
        raise asyncio.CancelledError
        yield

    with contextlib.suppress(asyncio.CancelledError):
        await _drive(_cancelled(), saver, final_state=delivered)
    assert "t" not in stopped_turns._pending


# ---------------------------------------------------------------------------
# A save that lands late doesn't write over a later run (D8-3047)
# ---------------------------------------------------------------------------


def test_still_this_turn():
    from src.api.stopped_turns import UNKNOWN_START, still_this_turn

    h0, a0 = HumanMessage(content="old", id="h0"), AIMessage(content="a", id="a0")
    q = HumanMessage(content="q", id="h1")
    later = [HumanMessage(content="q2", id="h2"), AIMessage(content="a2", id="a2")]
    assert still_this_turn([h0, a0], "q", "a0")
    assert still_this_turn([h0, a0, q], "q", "a0")
    assert still_this_turn([], "q", None)
    assert still_this_turn([q], "q", None)
    assert not still_this_turn([h0, a0, q, *later], "q", "a0")
    assert not still_this_turn([h0, a0, *later], "q", "a0")
    assert not still_this_turn([HumanMessage(content="x", id="hx")], "q", "a0")
    assert still_this_turn([h0, a0, *later], "q", UNKNOWN_START)


async def test_a_late_save_after_the_next_run_is_skipped(caplog):
    saver = InMemorySaver()
    await _seed(saver, [HumanMessage(content="old", id="h0"), AIMessage(content="a", id="a0")])
    # This run's question, then the next run's question and answer, all saved
    # before this run's slow save got to write.
    await _seed(
        saver,
        [
            HumanMessage(content="q", id="h1"),
            HumanMessage(content="q2", id="h2"),
            AIMessage(content="a2", id="a2"),
        ],
    )
    keep_stopped_turn(saver, "t", "q", {"r1": AIMessageChunk(content="Half", id="r1")}, start="a0")
    await stopped_turn_saved("t")
    assert [m.id for m in await _thread(saver)] == ["h0", "a0", "h1", "h2", "a2"]
    assert "a later run has written the thread" in caplog.text


async def test_the_run_records_where_it_began():
    """keeping_cut_off_turn reads the thread's end before the run starts."""
    saver = InMemorySaver()
    await _seed(saver, [HumanMessage(content="old", id="h0"), AIMessage(content="a", id="a0")])
    graph = create_checkpointed_graph(saver)

    async def _next_run_wins():
        # The next run's turn lands before this run's save does.
        await graph.aupdate_state(
            CONFIG,
            {"messages": [HumanMessage(content="q2", id="h2"), AIMessage(content="a2", id="a2")]},
            as_node="suggest_follow_ups",
        )
        yield 1
        raise asyncio.CancelledError

    with contextlib.suppress(asyncio.CancelledError):
        await _drive(_next_run_wins(), saver)
    await stopped_turn_saved("t")
    assert [m.id for m in await _thread(saver)] == ["h0", "a0", "h2", "a2"]


# ---------------------------------------------------------------------------
# A cut-off turn keeps the writes it already made (D8-3047)
# ---------------------------------------------------------------------------


def _calls(*names_ids, text=""):
    return AIMessage(
        content=text,
        id="m-" + "-".join(i for _, i in names_ids),
        tool_calls=[{"name": n, "args": {}, "id": i, "type": "tool_call"} for n, i in names_ids],
    )


def test_write_record_keeps_writes_paired_with_their_results():
    from langchain_core.messages import ToolMessage

    from src.api.stopped_turns import WRITE_IN_FLIGHT_TEXT, write_record

    read_round = _calls(("search_resources", "c1"))
    read_result = ToolMessage(content="found", tool_call_id="c1", name="search_resources")
    mixed = _calls(("create_announcement", "c2"), ("search_events", "c3"), text="Posting it.")
    done = ToolMessage(content="created #7", tool_call_id="c2", name="create_announcement")
    in_flight = _calls(("update_announcement", "c4"))

    record = write_record([read_round, read_result, mixed, done, in_flight, "junk"])

    assert [type(m).__name__ for m in record] == [
        "AIMessage",
        "ToolMessage",
        "AIMessage",
        "ToolMessage",
    ]
    assert [c["id"] for c in record[0].tool_calls] == ["c2"]
    assert record[0].content == "Posting it."
    assert record[1].content == "created #7"
    assert record[3].tool_call_id == "c4"
    assert record[3].content == WRITE_IN_FLIGHT_TEXT
    assert record[3].status == "error"
    assert write_record([read_round, read_result]) == []


def test_write_record_leaves_out_calls_that_only_previewed():
    import json

    from langchain_core.messages import ToolMessage

    from src.api.stopped_turns import WRITE_IN_FLIGHT_TEXT, write_record

    preview = json.dumps({"status": "preview", "executed": False, "data": {}})
    ran = json.dumps({"status": "created", "executed": True})
    previewed = _calls(("create_announcement", "c1"))
    previewed_result = ToolMessage(content=preview, tool_call_id="c1", name="create_announcement")
    mixed = _calls(("update_announcement", "c2"), ("delete_announcement", "c3"))
    ran_result = ToolMessage(content=ran, tool_call_id="c2", name="update_announcement")
    cut_off = ToolMessage(content=preview, tool_call_id="x", name="x")

    assert write_record([previewed, previewed_result]) == []

    record = write_record([mixed, ran_result, _calls(("create_event", "c4")), cut_off])

    assert [c["id"] for c in record[0].tool_calls] == ["c2", "c3"]
    assert record[1].content == ran
    assert record[2].content == WRITE_IN_FLIGHT_TEXT
    assert [c["id"] for c in record[3].tool_calls] == ["c4"]


def test_stopped_messages_put_the_writes_before_the_stopped_text():
    from langchain_core.messages import ToolMessage

    call = _calls(("create_announcement", "c2"), text="Posting it.")
    result = ToolMessage(content="created #7", tool_call_id="c2", name="create_announcement")
    streamed = {
        call.id: AIMessageChunk(content="Posting it.", id=call.id),
        "r2": AIMessageChunk(content="Done, it is", id="r2"),
    }
    saved = [HumanMessage(content="q", id="h")]
    out = stopped_messages(saved, "q", streamed, "user", [call, result])
    assert [m.id for m in out[:2]] == [call.id, result.id]
    assert out[1].tool_call_id == "c2"
    # The call's own text isn't repeated as stopped text.
    assert [(m.id, m.content) for m in out[2:]] == [("r2", "Done, it is")]
    assert is_stopped(out[-1]) and not is_stopped(out[0])
    # With the question not yet saved, it comes first.
    out = stopped_messages([], "q", {}, "user", [call, result])
    assert isinstance(out[0], HumanMessage)
    assert out[1].tool_calls


async def test_without_tool_messages_sets_them_aside():
    from src.api.stopped_turns import without_tool_messages

    async def _events():
        yield "messages", 1
        yield "tool_messages", ["call", "result"]
        yield "updates", 2

    kept: list = []
    out = [e async for e in without_tool_messages(kept, _events())]
    assert out == [("messages", 1), ("updates", 2)]
    assert kept == ["call", "result"]
