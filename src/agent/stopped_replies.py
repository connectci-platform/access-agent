"""Replies that were cut off: how the thread marks them, and how the model sees them.

The thread route saves the text that had streamed when a turn is stopped, times
out or fails before its answer is delivered (see src/api/stopped_turns.py). The
message is marked in its ``response_metadata``, which the client reads and the
model never sees.

The model is told only when the reply is empty, so the turn doesn't read as a
blank reply. A reply with text is shown as it is: its unfinished last sentence
says enough, and a note on it sent Qwen into a loop on the next question.
Replaying a real stopped thread, a note appended to the partial text left the
next answer reasoning-only, out of tokens before it chose a tool, in 6 of 36
runs; in the person's next message it did so in 2 of 16. Without the note it
was 0 of 32, and an empty reply with the note 0 of 16.
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


def _empty(message: Any) -> bool:
    return not message.text.strip()


def stopped_replies_view(messages: list[AnyMessage]) -> list[AnyMessage]:
    """The thread with the note in place of each empty cut-off reply, so it
    doesn't read as a blank reply. A cut-off reply with text is left as it is
    (see the module docstring). The stored messages don't change."""
    if not any(is_stopped(m) and _empty(m) for m in messages):
        return messages
    return [
        m.model_copy(update={"content": CUT_OFF_NOTE}) if is_stopped(m) and _empty(m) else m
        for m in messages
    ]
