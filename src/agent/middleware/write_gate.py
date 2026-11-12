"""Middleware: a write runs only from the user's Confirm click.

Every write tool previews unless called with ``confirmed: true``, and the model
chose which. Now the model can only preview: its write calls are always sent
with ``confirmed: false``, and each preview is stored with an id the client
shows beside a Confirm button. A Confirm turn carries those ids on its
HumanMessage. Before the model runs, the gate claims them, once ever, and
replays the stored calls with ``confirmed: true`` by jumping straight to the
tool node, so they still pass the acting-user check and every other tool
middleware. What runs is what the user saw, and a double click or a resent
turn can't write twice.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ...pending_previews import get_pending_preview_store
from ..domains.capabilities import WRITE_MCP_TOOL_NAMES

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.runtime import Runtime
    from langgraph.types import Command

logger = logging.getLogger(__name__)

PREVIEW_NOTE = (
    "This is a preview. Nothing has changed. The user will see these details with a "
    "Confirm button; ask them to review and click Confirm. Do not call this tool "
    "again to confirm."
)
UNCONFIRMABLE_NOTE = (
    "This is a preview, but it can't be confirmed right now. Tell the user to try "
    "again in a moment."
)
UNCONFIRMED_WRITE_NOTE = (
    "The server may have made this change without confirmation. Tell the user "
    "exactly what the result says."
)
CONFIRM_FAILED_NOTE = (
    "The user clicked Confirm, but that preview was already used or has expired, so "
    "nothing was changed. Tell them so and offer to preview it again."
)
NOT_FROM_ASSISTANT_TEXT = "This change can't be made from the assistant."


def _text(content: Any) -> str | None:
    """A result's text, from a string or from content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            b if isinstance(b, str) else str(b.get("text", ""))
            for b in content
            if isinstance(b, str) or (isinstance(b, dict) and b.get("type") == "text")
        )
    return None


def _json_object(message: ToolMessage) -> dict[str, Any] | None:
    text = _text(message.content)
    if text is None:
        return None
    try:
        body = json.loads(text)
    except json.JSONDecodeError:
        return None
    return body if isinstance(body, dict) else None


def preview_body(message: Any) -> dict[str, Any] | None:
    """The result as an object when it is an unexecuted preview, else None."""
    if not isinstance(message, ToolMessage):
        return None
    body = _json_object(message)
    if body is None or body.get("status") != "preview" or body.get("executed") is not False:
        return None
    return body


def _is_failure(message: ToolMessage) -> bool:
    """A failure wrote nothing, so it isn't reported as an unconfirmed write."""
    if message.status == "error":
        return True
    body = _json_object(message)
    return body is not None and "error" in body and body.get("executed") is not True


def _with_body(message: ToolMessage, body: dict[str, Any]) -> ToolMessage:
    # Always a JSON string: the client finds previews by parsing string content.
    return message.model_copy(update={"content": json.dumps(body, ensure_ascii=False)})


def _with_note(message: ToolMessage, note: str) -> ToolMessage:
    body = _json_object(message)
    if body is not None:
        return _with_body(message, {**body, "note": note})
    return message.model_copy(update={"content": f"{_text(message.content) or ''}\n\n{note}"})


def _noted(message: HumanMessage, notes: list[str]) -> HumanMessage:
    extra = "\n\n".join(notes)
    if isinstance(message.content, str):
        return message.model_copy(update={"content": f"{message.content}\n\n{extra}"})
    return message.model_copy(
        update={"content": [*message.content, {"type": "text", "text": extra}]}
    )


class WriteGateMiddleware(AgentMiddleware):
    """Force model writes to previews, and replay the previews the user confirmed.

    Built once per turn with the thread's id and the latest question's confirm
    ids, so ``approved`` and the failure notes are this turn's.
    """

    def __init__(self, thread_id: str, confirm_ids: list[str]) -> None:
        super().__init__()
        self.thread_id = thread_id
        self.confirm_ids = confirm_ids
        self.approved: set[str] = set()
        self.failed_notes: list[str] = []
        self._claimed = False

    @hook_config(can_jump_to=["tools"])
    async def abefore_model(
        self,
        state: Any,  # noqa: ARG002
        runtime: Runtime[Any],  # noqa: ARG002
    ) -> dict[str, Any] | None:
        # The loop comes back here after every tool round; only the first may claim.
        if self._claimed:
            return None
        self._claimed = True
        store = get_pending_preview_store()
        calls = []
        for preview_id in dict.fromkeys(self.confirm_ids):
            claimed = await asyncio.to_thread(store.claim, self.thread_id, preview_id)
            if claimed is None:
                self.failed_notes.append(CONFIRM_FAILED_NOTE)
                continue
            tool, args = claimed
            call_id = f"confirm-{preview_id}"
            calls.append({"id": call_id, "name": tool, "args": {**args, "confirmed": True}})
            self.approved.add(call_id)
        if not calls:
            return None
        return {"messages": [AIMessage(content="", tool_calls=calls)], "jump_to": "tools"}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if not self.failed_notes:
            return await handler(request)
        view = list(request.messages)
        for i in range(len(view) - 1, -1, -1):
            message = view[i]
            if isinstance(message, HumanMessage):
                view[i] = _noted(message, self.failed_notes)
                break
        return await handler(request.override(messages=view))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        name = call["name"]
        if name not in WRITE_MCP_TOOL_NAMES:
            return await handler(request)
        # A write that can't preview would run on the model's word alone.
        if "confirmed" not in (getattr(request.tool, "args", None) or {}):
            return ToolMessage(
                content=NOT_FROM_ASSISTANT_TEXT, name=name, tool_call_id=call["id"], status="error"
            )
        if call["id"] in self.approved:
            # Once: the id is ours, but a later call could reuse it.
            self.approved.discard(call["id"])
            return await handler(request)
        args = {k: v for k, v in call["args"].items() if k != "confirmed"}
        result = await handler(
            request.override(tool_call={**call, "args": {**args, "confirmed": False}})
        )
        if not isinstance(result, ToolMessage):
            logger.error("WRITE_WITHOUT_CONFIRM: %s returned no tool result to a preview", name)
            return result
        body = preview_body(result)
        if body is not None:
            store = get_pending_preview_store()
            preview_id = await asyncio.to_thread(store.record, self.thread_id, name, args)
            if preview_id is None:
                return _with_body(result, {**body, "note": UNCONFIRMABLE_NOTE})
            return _with_body(result, {**body, "preview_id": preview_id, "note": PREVIEW_NOTE})
        if _is_failure(result):
            return result
        logger.error("WRITE_WITHOUT_CONFIRM: %s did not preview when called unconfirmed", name)
        return _with_note(result, UNCONFIRMED_WRITE_NOTE)
