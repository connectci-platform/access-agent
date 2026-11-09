"""Replies that were cut off: how the thread marks them, and how the model sees them.

The thread route saves the text that had streamed when a turn is stopped, times
out or fails before its answer is delivered (see src/api/stopped_turns.py). The
message is marked in its ``response_metadata``, which the client reads and the
model never sees, so the model is told in its own view of the thread instead.
"""

from typing import Any

from langchain_core.messages import AIMessage, AnyMessage

# The response_metadata key, and why the reply stopped.
STOPPED = "stopped"
REASON = "reason"
REASONS = ("user", "timeout", "error")

CUT_OFF_NOTE = "[This reply was cut off before it finished.]"


def is_stopped(message: Any) -> bool:
    """Whether the message is a reply that was cut off."""
    return isinstance(message, AIMessage) and message.response_metadata.get(STOPPED) is True


def _with_note(content: str | list[Any]) -> str | list[Any]:
    if isinstance(content, list):
        return [*content, {"type": "text", "text": CUT_OFF_NOTE}]
    return f"{content.strip()}\n\n{CUT_OFF_NOTE}" if content.strip() else CUT_OFF_NOTE


def stopped_replies_view(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The thread with a note on each cut-off reply, so the model doesn't take a
    partial reply for a whole one, or an empty one for no reply. The stored
    messages don't change."""
    if not any(is_stopped(m) for m in messages):
        return messages
    return [
        m.model_copy(update={"content": _with_note(m.content)}) if is_stopped(m) else m
        for m in messages
    ]
