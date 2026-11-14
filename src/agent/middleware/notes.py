"""A note added to a message in the model's view."""

from __future__ import annotations

from typing import TypeVar

from langchain_core.messages import BaseMessage

M = TypeVar("M", bound=BaseMessage)


def with_note(message: M, note: str) -> M:
    """``message`` with ``note`` after its content: a new paragraph on string
    content, or a text block on list content, which a question that shares
    files has."""
    if isinstance(message.content, str):
        return message.model_copy(update={"content": f"{message.content}\n\n{note}"})
    blocks = [*message.content, {"type": "text", "text": note}]
    return message.model_copy(update={"content": blocks})
