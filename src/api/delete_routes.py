"""Deleting conversations from the full-screen client (D8-3126).

DELETE /threads/{thread_id} deletes one conversation for its owner, or for
the signed-out browser holding its claim key. DELETE /me/conversations
deletes every conversation of the signed-in caller. Neither touches a
conversation while an answer is streaming in it: that's a 409, and the
client asks again once the answer ends.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from ..conversation_delete import (
    ConversationBusyError,
    DeletionUnavailableError,
    delete_conversations,
    owned_thread_ids,
)
from .identity import ClaimKey, Identity, require_identity
from .thread_access import require_thread_access
from .thread_runs import acquire_thread_run, release_thread_run

delete_router = APIRouter()

_UNAVAILABLE = "Conversations can't be deleted right now. Try again shortly."
_BUSY = "An answer is still being written. Try again when it finishes."


def _busy() -> JSONResponse:
    return JSONResponse({"detail": _BUSY}, status_code=409)


def _checkpointer(request: Request) -> Any:
    return getattr(request.app.state, "checkpointer", None)


def _hold(thread_ids: list[str], run_id: str) -> None:
    """Hold every thread's run lock, or none of them and 409."""
    held: list[str] = []
    try:
        for thread_id in thread_ids:
            acquire_thread_run(thread_id, run_id)
            held.append(thread_id)
    except HTTPException:
        for thread_id in held:
            release_thread_run(thread_id, run_id)
        raise HTTPException(status_code=409, detail=_BUSY) from None


def _release(thread_ids: list[str], run_id: str) -> None:
    for thread_id in thread_ids:
        release_thread_run(thread_id, run_id)


@delete_router.delete("/threads/{thread_id}", status_code=204, response_model=None)
async def delete_thread(
    thread_id: str,
    request: Request,
    identity: Identity,
    claim_key: ClaimKey = None,
) -> Response:
    caller = identity.sub if identity else None
    # 404 for a thread the caller doesn't own, the same as reading it.
    require_thread_access(thread_id, caller, claim_key)
    run_id = f"delete-{uuid.uuid4()}"
    _hold([thread_id], run_id)
    try:
        await delete_conversations(
            [thread_id], user=caller, scope="one", checkpointer=_checkpointer(request)
        )
    except DeletionUnavailableError:
        return JSONResponse({"error": _UNAVAILABLE}, status_code=503)
    except ConversationBusyError:
        return _busy()
    finally:
        _release([thread_id], run_id)
    return Response(status_code=204)


@delete_router.delete("/me/conversations", response_model=None)
async def delete_my_conversations(request: Request, identity: Identity) -> JSONResponse:
    caller = require_identity(identity)
    try:
        thread_ids = owned_thread_ids(caller)
    except DeletionUnavailableError:
        return JSONResponse({"error": _UNAVAILABLE}, status_code=503)
    run_id = f"delete-{uuid.uuid4()}"
    _hold(thread_ids, run_id)
    try:
        deleted = await delete_conversations(
            thread_ids, user=caller, scope="all", checkpointer=_checkpointer(request)
        )
    except ConversationBusyError:
        return _busy()
    finally:
        _release(thread_ids, run_id)
    return JSONResponse({"deleted": deleted}, headers={"Cache-Control": "no-store"})
