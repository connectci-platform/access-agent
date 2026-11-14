"""Middleware: the model sees the files the user shared (D8-3086).

A question that shares files carries a reference block per file, built from
the stored row, never the file itself. On the way to the model, the
references of the questions in the upload window become what the model reads:
an image as an ``image_url`` data URL, and a PDF or text file as a ``<file>``
text block. Every other reference becomes a short note. ``upload_window``
decides all of it, through the same helpers the token estimate uses, so the
budget's count and the view agree.

Only the request changes. The thread in state, and so checkpoints, history
and turn reports, keep only the references. Nothing here logs what it loads.

It runs inside the turn budget, which counts references by what they will
cost. Outside it, the budget would see image blocks, which it can't count.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage

from ...tool_attachments import get_tool_attachment_store
from ..upload_window import (
    NO_MESSAGE_NOTE,
    NO_TEXT_BODY,
    cut_note,
    earlier_note,
    file_block,
    file_name,
    has_typed_text,
    is_reference,
    left_out_note,
    missing_note,
    references,
    shown_in_full,
    upload_window,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping

    from langchain.agents.middleware.types import ModelRequest, ModelResponse

# The image types processing stores. A row with any other type is not one
# this code wrote, so it reads as missing.
_IMAGE_TYPES = frozenset({"image/png", "image/jpeg"})

_Block = dict[str, Any]


def _text(text: str) -> _Block:
    return {"type": "text", "text": text}


def _file_text(ref: Mapping[str, Any], row: Mapping[str, Any]) -> str | None:
    """A PDF's or text file's body as the model reads it, or None when the row
    has none."""
    if ref.get("kind") == "pdf" and row.get("no_text") is True:
        return NO_TEXT_BODY
    text = row.get("text")
    if not isinstance(text, str):
        return None
    return f"{text}\n{cut_note()}" if row.get("text_truncated") is True else text


def _shown(ref: Mapping[str, Any], row: Any) -> _Block:
    """A reference in the window, shown in full from its stored row."""
    if not isinstance(row, dict) or row.get("kind") != ref.get("kind"):
        return _text(missing_note(ref))
    if ref.get("kind") == "image":
        mime_type, data = row.get("mime_type"), row.get("base64")
        if mime_type in _IMAGE_TYPES and isinstance(data, str) and data:
            return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}}
        return _text(missing_note(ref))
    body = _file_text(ref, row)
    if body is None:
        return _text(missing_note(ref))
    return _text(file_block(file_name(ref), body))


class UploadsMiddleware(AgentMiddleware):
    """Send the model the files shared in the upload window.

    The loop node builds one per run, so ``_loaded`` caches each row's block
    for the run's model calls and no longer.
    """

    def __init__(self, thread_id: str) -> None:
        super().__init__()
        self.thread_id = thread_id
        self._loaded: dict[str, _Block] = {}

    async def _block(self, ref: Mapping[str, Any]) -> _Block:
        attachment_id = ref.get("id")
        if not isinstance(attachment_id, str):
            return _text(missing_note(ref))
        if attachment_id not in self._loaded:
            store = get_tool_attachment_store()
            row = await asyncio.to_thread(store.get_payload, self.thread_id, attachment_id)
            self._loaded[attachment_id] = _shown(ref, row)
        return self._loaded[attachment_id]

    async def _view(self, message: HumanMessage, window: frozenset[str]) -> HumanMessage:
        in_window = message.id in window
        full = iter(shown_in_full(references(message.content)) if in_window else [])
        blocks: list[Any] = [] if has_typed_text(message.content) else [_text(NO_MESSAGE_NOTE)]
        for block in message.content:
            if not is_reference(block):
                blocks.append(block)
            elif not in_window:
                blocks.append(_text(earlier_note(block)))
            elif not next(full):
                blocks.append(_text(left_out_note(block)))
            else:
                blocks.append(await self._block(block))
        return message.model_copy(update={"content": blocks})

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if not any(isinstance(m, HumanMessage) and references(m.content) for m in request.messages):
            return await handler(request)
        # The window the turn budget counted this same list by.
        window = upload_window(request.messages)
        view = [
            await self._view(m, window)
            if isinstance(m, HumanMessage) and references(m.content)
            else m
            for m in request.messages
        ]
        return await handler(request.override(messages=view))
