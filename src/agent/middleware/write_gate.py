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
import re
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ...pending_previews import ClaimStatus, get_pending_preview_store
from ..auth_tools import ToolIdentityClass, tool_identity_class

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
    "this click changed nothing. Tell them so and offer to preview it again."
)
ALREADY_CONFIRMED_NOTE = (
    "That change was already confirmed earlier in this conversation; don't preview "
    "it again. Point the user to the earlier result, and if that result shows it "
    "failed, offer to preview it again."
)
CONFIRM_ERROR_NOTE = (
    "The user clicked Confirm, but it couldn't be confirmed right now, so nothing was "
    "changed by this click. Tell them to try again in a moment."
)
PARTIAL_CONFIRM_NOTE = (
    "The user clicked Confirm on {total} changes. {failed} of them had expired or "
    "couldn't be found, so those weren't made; the rest ran, and their results are "
    "the latest tool results. Tell them which changes were made and offer to preview "
    "the others again."
)
TYPED_CONFIRM_NOTE = (
    "The user typed a confirmation instead of clicking Confirm, so nothing was "
    "changed. Ask them to click Confirm on the card."
)
SERVICE_DOWN_TEXT = (
    "That service isn't available right now, so nothing was changed. Tell the user "
    "and offer to preview it again later."
)
_TYPED_CONFIRMATIONS = frozenset(
    {"confirm", "yes", "y", "ok", "okay", "go ahead", "do it", "proceed"}
)
NOT_FROM_ASSISTANT_TEXT = "This change can't be made from the assistant."


def _is_typed_confirmation(message: HumanMessage) -> bool:
    """A bare "yes" or "confirm" typed in place of the button, nothing longer."""
    text = (_text(message.content) or "").strip().casefold().rstrip(".!?, ")
    return text in _TYPED_CONFIRMATIONS


def _answers_a_preview(messages: list[Any], human_index: int) -> bool:
    """Whether the turn before this message left a preview waiting for Confirm."""
    for message in reversed(messages[:human_index]):
        if isinstance(message, HumanMessage):
            return False
        if preview_body(message) is not None:
            return True
    return False


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


MAX_CONFIRM_IDS = 10
_PREVIEW_ID = re.compile(r"[0-9a-f]{32}")


def clean_confirm_ids(raw: Any) -> list[str]:
    """Client-sent preview ids, filtered: only exactly-32-lowercase-hex strings
    survive, deduplicated in order, at most 10. Anything malformed is dropped."""
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    for item in raw:
        if isinstance(item, str) and _PREVIEW_ID.fullmatch(item) and item not in ids:
            ids.append(item)
            if len(ids) == MAX_CONFIRM_IDS:
                break
    return ids


def turn_previews(messages: list[Any]) -> list[dict[str, Any]]:
    """The stamped previews among a turn's tool messages, in order, as
    ``{"preview_id", "tool", "data", "no_changes"}``. A preview with no valid
    id (its record failed) can't be confirmed, so it is left out, and so is a
    result from a tool whose name doesn't classify as a write."""
    previews: list[dict[str, Any]] = []
    for message in messages:
        body = preview_body(message)
        if body is None:
            continue
        if tool_identity_class(message.name or "") is not ToolIdentityClass.WRITE:
            continue
        preview_id = body.get("preview_id")
        if not isinstance(preview_id, str) or not _PREVIEW_ID.fullmatch(preview_id):
            continue
        data = body.get("data")
        previews.append(
            {
                "preview_id": preview_id,
                "tool": message.name or "",
                "data": data,
                "no_changes": body.get("no_changes") is True
                or (isinstance(data, dict) and data.get("no_changes") is True),
            }
        )
    return previews


def _is_failure(message: ToolMessage) -> bool:
    """A failure wrote nothing, so it isn't reported as an unconfirmed write."""
    if message.status == "error":
        return True
    body = _json_object(message)
    return body is not None and "error" in body and body.get("executed") is not True


def _with_body(message: ToolMessage, body: dict[str, Any]) -> ToolMessage:
    # Always a JSON string: the client finds previews by parsing string content.
    return message.model_copy(update={"content": json.dumps(body, ensure_ascii=False)})


def _without_preview_id(message: Any) -> Any:
    """The model's copy of a stamped result: the id is the client's, not the model's."""
    if not isinstance(message, ToolMessage):
        return message
    body = _json_object(message)
    if body is None or "preview_id" not in body:
        return message
    return _with_body(message, {k: v for k, v in body.items() if k != "preview_id"})


def _acting_user(runtime: Any) -> str | None:
    user = getattr(getattr(runtime, "context", None), "acting_user", None)
    return user if isinstance(user, str) else None


def _partial_note(total: int, used: int, gone: int, errors: int) -> str:
    if not used and not errors:
        return PARTIAL_CONFIRM_NOTE.format(total=total, failed=gone)
    parts = [f"The user clicked Confirm on {total} changes."]
    if used:
        parts.append(
            f"{used} of them had already been confirmed earlier in this conversation, so "
            "they weren't made again; don't preview those again."
        )
    if gone:
        parts.append(
            f"{gone} of them had expired or couldn't be found, so those weren't made; "
            "offer to preview them again."
        )
    if errors:
        parts.append(
            f"{errors} of them couldn't be confirmed right now, so those weren't made; "
            "they can try those again in a moment."
        )
    parts.append(
        "The rest ran, and their results are the latest tool results. Tell them which "
        "changes were made."
    )
    return " ".join(parts)


def _with_note(message: ToolMessage, note: str) -> ToolMessage:
    body = _json_object(message)
    if body is not None:
        return _with_body(message, {**body, "note": note})
    return message.model_copy(update={"content": f"{_text(message.content) or ''}\n\n{note}"})


def _noted(message: HumanMessage, extra: str) -> HumanMessage:
    if isinstance(message.content, str):
        return message.model_copy(update={"content": f"{message.content}\n\n{extra}"})
    return message.model_copy(
        update={"content": [*message.content, {"type": "text", "text": extra}]}
    )


def _warn_if_not_executed(name: str, result: Any) -> None:
    """A confirmed replay that didn't write means the server and the stored
    preview disagree; the user clicked Confirm and nothing changed."""
    if not isinstance(result, ToolMessage):
        return
    body = _json_object(result)
    failed = result.status == "error"
    if failed or (
        body is not None and (body.get("status") == "preview" or body.get("executed") is False)
    ):
        logger.warning("CONFIRMED_NOT_EXECUTED: %s did not write when confirmed", name)


class WriteGateMiddleware(AgentMiddleware):
    """Force model writes to previews, and replay the previews the user confirmed.

    Built once per turn with the thread's id and the latest question's confirm
    ids, so ``approved`` and the failure notes are this turn's.
    """

    def __init__(self, thread_id: str, confirm_ids: list[str]) -> None:
        super().__init__()
        self.thread_id = thread_id
        self.confirm_ids = confirm_ids
        # call id -> (tool, args): only that exact call runs confirmed.
        self.approved: dict[str, tuple[str, dict[str, Any]]] = {}
        self.failed_note: str | None = None
        self._claimed = False
        # (tool, args) -> preview id recorded this turn, so identical calls
        # share one id and Confirm all can't write the same change twice. The
        # lock covers parallel calls from one model step.
        self._recorded: dict[tuple[str, str], str] = {}
        self._record_lock = asyncio.Lock()

    @hook_config(can_jump_to=["tools"])
    async def abefore_model(
        self,
        state: Any,  # noqa: ARG002
        runtime: Runtime[Any],
    ) -> dict[str, Any] | None:
        # The loop comes back here after every tool round; only the first may
        # claim. Approval ends with that round, so a replayed call refused before
        # reaching this gate can't leave one behind for the model to reuse.
        if self._claimed:
            self.approved.clear()
            return None
        self._claimed = True
        store = get_pending_preview_store()
        acting_user = _acting_user(runtime)
        preview_ids = list(dict.fromkeys(self.confirm_ids))
        calls = []
        failures = dict.fromkeys(ClaimStatus, 0)
        for preview_id in preview_ids:
            claim = await asyncio.to_thread(
                store.claim, self.thread_id, preview_id, acting_user=acting_user
            )
            if claim.status is not ClaimStatus.CLAIMED or claim.tool is None:
                failures[claim.status] += 1
                continue
            call_id = f"confirm-{preview_id}"
            args = {**(claim.args or {}), "confirmed": True}
            calls.append({"id": call_id, "name": claim.tool, "args": args})
            self.approved[call_id] = (claim.tool, args)
        used = failures[ClaimStatus.ALREADY_USED]
        gone = failures[ClaimStatus.EXPIRED_OR_UNKNOWN]
        errors = failures[ClaimStatus.ERROR]
        if (used or gone or errors) and calls:
            self.failed_note = _partial_note(len(preview_ids), used, gone, errors)
        elif used or gone or errors:
            notes = [
                note
                for count, note in (
                    (used, ALREADY_CONFIRMED_NOTE),
                    (gone, CONFIRM_FAILED_NOTE),
                    (errors, CONFIRM_ERROR_NOTE),
                )
                if count
            ]
            if len(preview_ids) > 1:
                notes.append(f"None of the {len(preview_ids)} changes they confirmed were made.")
            self.failed_note = " ".join(notes)
        if not calls:
            return None
        return {"messages": [AIMessage(content="", tool_calls=calls)], "jump_to": "tools"}

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        view = [_without_preview_id(m) for m in request.messages]
        changed = any(a is not b for a, b in zip(view, request.messages, strict=True))
        for i in range(len(view) - 1, -1, -1):
            message = view[i]
            if not isinstance(message, HumanMessage):
                continue
            note = self.failed_note
            if (
                note is None
                and not self.confirm_ids
                and _is_typed_confirmation(message)
                and _answers_a_preview(view, i)
            ):
                note = TYPED_CONFIRM_NOTE
            if note is not None:
                view[i] = _noted(message, note)
                changed = True
            break
        if not changed:
            return await handler(request)
        return await handler(request.override(messages=view))

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        name = call["name"]
        # Taken on first sight, so an approval is used once at most.
        approved = self.approved.pop(call["id"] or "", None)
        # The shared classifier, so a write missing from the name list still
        # fails closed when its server marks it as one.
        read_only = getattr(request.tool, "read_only", None)
        if tool_identity_class(name, read_only) is not ToolIdentityClass.WRITE:
            return await handler(request)
        # The user approved this call but its service is gone from the catalog.
        if request.tool is None and approved == (name, call["args"]):
            return ToolMessage(
                content=SERVICE_DOWN_TEXT, name=name, tool_call_id=call["id"], status="error"
            )
        # A write that can't preview would run on the model's word alone.
        if "confirmed" not in (getattr(request.tool, "args", None) or {}):
            return ToolMessage(
                content=NOT_FROM_ASSISTANT_TEXT, name=name, tool_call_id=call["id"], status="error"
            )
        if approved == (name, call["args"]):
            result = await handler(request)
            _warn_if_not_executed(name, result)
            return result
        args = {k: v for k, v in call["args"].items() if k != "confirmed"}
        result = await handler(
            request.override(tool_call={**call, "args": {**args, "confirmed": False}})
        )
        if not isinstance(result, ToolMessage):
            logger.error("WRITE_WITHOUT_CONFIRM: %s returned no tool result to a preview", name)
            return result
        body = preview_body(result)
        if body is not None:
            preview_id = await self._record(name, args, _acting_user(request.runtime))
            if preview_id is None:
                return _with_body(result, {**body, "note": UNCONFIRMABLE_NOTE})
            return _with_body(result, {**body, "preview_id": preview_id, "note": PREVIEW_NOTE})
        if _is_failure(result):
            return result
        logger.error("WRITE_WITHOUT_CONFIRM: %s did not preview when called unconfirmed", name)
        return _with_note(result, UNCONFIRMED_WRITE_NOTE)

    async def _record(self, name: str, args: dict[str, Any], acting_user: str | None) -> str | None:
        """This turn's id for the call, recording it on first sight."""
        key = (name, json.dumps(args, sort_keys=True, default=str))
        async with self._record_lock:
            preview_id = self._recorded.get(key)
            if preview_id is None:
                store = get_pending_preview_store()
                preview_id = await asyncio.to_thread(
                    store.record, self.thread_id, name, args, acting_user
                )
                if preview_id is not None:
                    self._recorded[key] = preview_id
            return preview_id
