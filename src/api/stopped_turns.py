"""A turn cut off before its answer was delivered, kept in the thread as it was.

The graph saves the question when a run starts, but an answer only when the tool
loop returns it. A stop, a timeout or a failure in between used to leave the
thread ending in a question nobody answered, so the next run's model saw two
questions in a row and answered the first. The routes now save the answer text
that had streamed after the question, marked stopped with the reason, so
history, a reload and the next question all see the same turn the person saw.
Retry and Edit replace that turn instead of asking the question a second time.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, TypeVar

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    ToolMessage,
)

from ..agent.auth_tools import ToolIdentityClass, tool_identity_class
from ..agent.graph import create_checkpointed_graph
from ..agent.stopped_replies import REASON, STOPPED, is_stopped
from ..agent.turn_capture import get_turn_capture

T = TypeVar("T")

logger = logging.getLogger(__name__)

# How long a run or a history read waits for a save before going on without it.
SAVE_WAIT_S = 5.0

# Written as this node so the saved checkpoint has nothing left to run: its only
# edge goes to END, as a finished turn's does.
_AS_NODE = "suggest_follow_ups"

# Stands for a write the turn sent whose result never came back.
WRITE_IN_FLIGHT_TEXT = (
    "This request was sent, but the turn was cut off before its result came back. "
    "It may have gone through: check before doing it again."
)

# A run's start when the stop came before the run read the thread.
UNKNOWN_START = object()

# Saves started by a stop, by thread, so the next run and a history read wait
# for them instead of reading the thread before the stopped turn is in it.
_pending: dict[str, asyncio.Task[None]] = {}


def _question_saved(saved: list[AnyMessage], query: str) -> bool:
    """Whether the graph saved this run's question: the thread ends in it.

    Compared by text, so a question an earlier failed run left at the end of
    the thread isn't taken for this one.
    """
    last = saved[-1] if saved else None
    return isinstance(last, HumanMessage) and last.text.strip() == query.strip()


def _is_write(name: str) -> bool:
    return tool_identity_class(name) is ToolIdentityClass.WRITE


def write_record(tool_messages: list[Any]) -> list[BaseMessage]:
    """The turn's write calls, each followed by its result, in the order they
    happened. A write still in flight gets a result saying it may have gone
    through, so a Retry doesn't repeat it blind. Read calls are left out: they
    changed nothing, and asking again reads again.
    """
    results = {m.tool_call_id: m for m in tool_messages if isinstance(m, ToolMessage)}
    record: list[BaseMessage] = []
    for message in tool_messages:
        if not isinstance(message, AIMessage):
            continue
        calls = [c for c in message.tool_calls if _is_write(c["name"])]
        if not calls:
            continue
        record.append(
            AIMessage(content=message.content, id=message.id or str(uuid.uuid4()), tool_calls=calls)
        )
        for call in calls:
            result = results.get(call["id"] or "")
            record.append(
                result
                if result is not None
                else ToolMessage(
                    content=WRITE_IN_FLIGHT_TEXT,
                    tool_call_id=call["id"] or "",
                    name=call["name"],
                    status="error",
                    id=str(uuid.uuid4()),
                )
            )
    return record


def stopped_messages(
    saved: list[AnyMessage],
    query: str,
    streamed: dict[str, AIMessageChunk],
    reason: str = "user",
    tool_messages: list[Any] | None = None,
) -> list[BaseMessage]:
    """What a cut-off turn adds to the saved thread: the question when the turn
    ended before the graph saved it, then its write calls and their results
    (see write_record), then the answer text that had streamed, one message per
    streamed id as the client showed it.

    With no text yet the answer is an empty message, so the question still
    reads as answered by a stopped turn rather than waiting for an answer.
    ``reason`` is "user" (Stop or a closed tab), "timeout" or "error".
    """
    meta = {STOPPED: True, REASON: reason}
    record = write_record(tool_messages or [])
    # A write call's own text went with it, so it isn't repeated as stopped text.
    in_record = {m.id for m in record}
    partial: list[BaseMessage] = [
        AIMessage(content=chunk.content, id=key or str(uuid.uuid4()), response_metadata=meta)
        for key, chunk in streamed.items()
        if chunk.text.strip() and key not in in_record
    ]
    if not partial:
        partial = [AIMessage(content="", id=str(uuid.uuid4()), response_metadata=meta)]
    if _question_saved(saved, query):
        return [*record, *partial]
    return [HumanMessage(content=query, id=str(uuid.uuid4())), *record, *partial]


def still_this_turn(saved: list[AnyMessage], query: str, start: Any) -> bool:
    """Whether the saved thread is still where this run left it: nothing after
    the message it began from but, at most, its own question. A save slower
    than the next run's wait would otherwise land after that run's answer, or
    in the middle of it. ``start`` is that message's id, None for an empty
    thread, or UNKNOWN_START when the run never read the thread."""
    if start is UNKNOWN_START:
        return True
    ids = [m.id for m in saved]
    if start is not None and start not in ids:
        return False
    tail = saved[ids.index(start) + 1 :] if start is not None else saved
    return not tail or (len(tail) == 1 and _question_saved(tail, query))


def unanswered_turn(messages: list[AnyMessage]) -> list[str]:
    """Ids of the thread's last question and its stopped answer, when nothing
    else answered it, or an empty list. A question left without any answer, as
    a failed run leaves it, counts too."""
    at = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], HumanMessage)),
        None,
    )
    if at is None or not all(is_stopped(m) for m in messages[at + 1 :]):
        return []
    return [m.id for m in messages[at:] if m.id]


async def _save(
    checkpointer: Any,
    thread_id: str,
    query: str,
    streamed: dict[str, AIMessageChunk],
    reason: str,
    tool_messages: list[Any],
    start: Any,
) -> None:
    try:
        graph = create_checkpointed_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(config)
        saved = (state.values or {}).get("messages", [])
        if not still_this_turn(saved, query, start):
            logger.warning(
                "Not saving the stopped turn on %s: a later run has written the thread",
                thread_id,
            )
            return
        update = stopped_messages(saved, query, streamed, reason, tool_messages)
        await graph.aupdate_state(config, {"messages": update}, as_node=_AS_NODE)
    except Exception:
        logger.exception("Saving a stopped turn failed")


def keep_stopped_turn(
    checkpointer: Any,
    thread_id: str,
    query: str,
    streamed: dict[str, AIMessageChunk],
    reason: str = "user",
    *,
    tool_messages: list[Any] | None = None,
    start: Any = UNKNOWN_START,
) -> None:
    """Save the cut-off turn from a task of its own, so the cancel that ended
    the run can't cut the write short."""
    if checkpointer is None:
        return
    task = asyncio.ensure_future(
        _save(checkpointer, thread_id, query, streamed, reason, tool_messages or [], start)
    )
    _pending[thread_id] = task

    def _done(done: asyncio.Task[None]) -> None:
        if _pending.get(thread_id) is done:
            del _pending[thread_id]

    task.add_done_callback(_done)


async def stopped_turn_saved(thread_id: str, timeout: float = SAVE_WAIT_S) -> None:
    """Wait for a stopped turn still being saved on this thread, if any, for at
    most ``timeout`` seconds. A save that takes longer goes on by itself."""
    task = _pending.get(thread_id)
    if task is None:
        return
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout)
    except TimeoutError:
        logger.warning("Saving a stopped turn on %s is slow; going on without it", thread_id)


async def drain_stopped_turns(timeout: float = SAVE_WAIT_S) -> None:
    """At shutdown, give saves still running a moment to finish before the
    checkpointer closes."""
    if _pending:
        await asyncio.wait(list(_pending.values()), timeout=timeout)


async def drop_unanswered_turn(checkpointer: Any, thread_id: str) -> None:
    """Remove the thread's last question if nothing but a stopped answer
    followed it, so asking it again replaces that turn."""
    if checkpointer is None:
        return
    graph = create_checkpointed_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    state = await graph.aget_state(config)
    ids = unanswered_turn((state.values or {}).get("messages", []))
    if ids:
        removals = [RemoveMessage(id=i) for i in ids]
        await graph.aupdate_state(config, {"messages": removals}, as_node=_AS_NODE)


async def _thread_end(checkpointer: Any, thread_id: str) -> Any:
    """The id of the thread's last message as this run begins, or None."""
    if checkpointer is None:
        return UNKNOWN_START
    graph = create_checkpointed_graph(checkpointer)
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    messages = (state.values or {}).get("messages", [])
    return messages[-1].id if messages else None


def answer_delivered(final_state: dict[str, Any]) -> bool:
    """Whether the turn's answer is complete: a node has returned it, with its id.

    By then it has streamed and is in the checkpoint, so a cancel after this
    point (a Stop or a closed tab while follow-ups are suggested) is not a lost
    turn.
    """
    messages = final_state.get("messages") or []
    return (
        bool(final_state.get("final_answer"))
        and bool(messages)
        and isinstance(messages[-1], AIMessage)
        and bool(messages[-1].id)
    )


def _past_turn_deadline() -> bool:
    """Whether the turn's time is up, which makes a cancel its timeout."""
    deadline = get_turn_capture().get("turn_deadline")
    return deadline is not None and time.monotonic() >= deadline


async def without_tool_messages(
    into: list[Any], events: AsyncIterator[tuple[str, Any]]
) -> AsyncGenerator[tuple[str, Any], None]:
    """``events`` from stream_agent without its "tool_messages" events, whose
    messages go into ``into`` for a cut-off turn's write record."""
    async for stream_type, chunk in events:
        if stream_type == "tool_messages":
            into.extend(chunk)
        else:
            yield stream_type, chunk


async def keeping_cut_off_turn(
    events: AsyncIterator[T],
    *,
    checkpointer: Any,
    thread_id: str,
    query: str,
    final_state: dict[str, Any],
    streamed: dict[str, AIMessageChunk],
    tool_messages: list[Any] | None = None,
    replace_stopped_turn: bool = False,
) -> AsyncGenerator[T, None]:
    """``events``, one run's stream, keeping its turn in the thread when it ends
    before its answer was delivered. The caller fills ``final_state`` and
    ``streamed`` (answer chunks by message id) and ``tool_messages`` (the
    "tool_messages" events stream_agent sends) as it reads the events.

    A stop lands here as a CancelledError, or as GeneratorExit when it came
    while an event was being sent and the generator is closed afterwards. The
    turn timeout is a CancelledError too, told apart by the turn's deadline. A
    failure is any other exception. Each is re-raised for the route to report.
    """

    start: Any = UNKNOWN_START

    def _keep(reason: str) -> None:
        if not answer_delivered(final_state):
            keep_stopped_turn(
                checkpointer,
                thread_id,
                query,
                streamed,
                reason,
                tool_messages=tool_messages,
                start=start,
            )

    try:
        # A turn cut off just before this run may still be being saved.
        await stopped_turn_saved(thread_id)
        if replace_stopped_turn:
            # Retry or Edit: the question asked again replaces its stopped turn.
            await drop_unanswered_turn(checkpointer, thread_id)
        start = await _thread_end(checkpointer, thread_id)
        async for event in events:
            yield event
    except (asyncio.CancelledError, GeneratorExit):
        _keep("timeout" if _past_turn_deadline() else "user")
        raise
    except Exception:
        _keep("error")
        raise
