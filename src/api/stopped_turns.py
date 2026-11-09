"""A turn the person stopped, kept in the thread as it was when they stopped it.

The graph saves the question when a run starts, but an answer only when the tool
loop returns it. A stop in between used to leave the thread ending in a question
nobody answered, so the next run's model saw two questions in a row and answered
the first. On a stop the route now saves the answer text that had streamed after
the question, marked stopped, so history, a reload and the next question all see
the same turn the person saw. Retry and Edit replace that turn instead of asking
the question a second time.
"""

import asyncio
import logging
import uuid
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
)

from ..agent.graph import create_checkpointed_graph

logger = logging.getLogger(__name__)

# The answer message's response_metadata key the client reads to show the
# Stopped note. response_metadata is never sent to the model.
STOPPED = "stopped"

# Written as this node so the saved checkpoint has nothing left to run: its only
# edge goes to END, as a finished turn's does.
_AS_NODE = "suggest_follow_ups"

# Saves started by a stop, by thread, so the next run and a history read wait
# for them instead of reading the thread before the stopped turn is in it.
_pending: dict[str, asyncio.Task[None]] = {}


def is_stopped(message: Any) -> bool:
    """Whether the message is an answer the person stopped."""
    return isinstance(message, AIMessage) and message.response_metadata.get(STOPPED) is True


def stopped_messages(
    saved: list[AnyMessage], query: str, streamed: dict[str, AIMessageChunk]
) -> list[BaseMessage]:
    """What a stop adds to the saved thread: the answer text that had streamed,
    one message per streamed id as the client showed it, and the question too
    when the stop came before the graph saved it.

    With no text yet the answer is an empty message, so the question still
    reads as answered by a stopped turn rather than waiting for an answer.
    """
    partial: list[BaseMessage] = [
        AIMessage(content=chunk.content, id=key, response_metadata={STOPPED: True})
        for key, chunk in streamed.items()
        if key and chunk.text.strip()
    ]
    if not partial:
        partial = [AIMessage(content="", id=str(uuid.uuid4()), response_metadata={STOPPED: True})]
    if saved and isinstance(saved[-1], HumanMessage):
        return partial
    return [HumanMessage(content=query, id=str(uuid.uuid4())), *partial]


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
    checkpointer: Any, thread_id: str, query: str, streamed: dict[str, AIMessageChunk]
) -> None:
    try:
        graph = create_checkpointed_graph(checkpointer)
        config = {"configurable": {"thread_id": thread_id}}
        state = await graph.aget_state(config)
        saved = (state.values or {}).get("messages", [])
        update = stopped_messages(saved, query, streamed)
        await graph.aupdate_state(config, {"messages": update}, as_node=_AS_NODE)
    except Exception:
        logger.exception("Saving a stopped turn failed")


def keep_stopped_turn(
    checkpointer: Any, thread_id: str, query: str, streamed: dict[str, AIMessageChunk]
) -> None:
    """Save the stopped turn from a task of its own, so the cancel that ended
    the run can't cut the write short."""
    if checkpointer is None:
        return
    task = asyncio.ensure_future(_save(checkpointer, thread_id, query, streamed))
    _pending[thread_id] = task

    def _done(done: asyncio.Task[None]) -> None:
        if _pending.get(thread_id) is done:
            del _pending[thread_id]

    task.add_done_callback(_done)


async def stopped_turn_saved(thread_id: str) -> None:
    """Wait for a stopped turn still being saved on this thread, if any."""
    task = _pending.get(thread_id)
    if task is not None:
        await asyncio.shield(task)


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
