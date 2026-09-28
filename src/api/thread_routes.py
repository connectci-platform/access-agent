"""LangGraph thread/run protocol endpoints: create + history (resume) + run.

All routes sit behind auth (SESSaccess_auth cookie) and the thread-ownership
gate from src/thread_owners.py. Non-owner and unknown-thread requests return
an identical 404 body so thread existence is not observable to a caller who
doesn't own it.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from ..agent.graph import create_checkpointed_graph, stream_agent
from ..auth import get_acting_user_from_cookie
from ..config import settings
from ..thread_owners import get_thread_owner_store
from ..turn_reporter import _hash_user, get_turn_reporter
from .sse import format_sse_event
from .thread_runs import (
    acquire_thread_run,
    cancel_run,
    held_run,
    release_thread_run,
    run_with_timeout,
)

logger = logging.getLogger(__name__)

thread_router = APIRouter()


def _require_user(raw_request: Request) -> str:
    user, _ = get_acting_user_from_cookie(raw_request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user


def _require_access(thread_id: str, caller: str) -> None:
    if not get_thread_owner_store().check_access(thread_id, caller):
        raise HTTPException(status_code=404, detail="Thread not found")  # never 403


class HistoryRequest(BaseModel):
    limit: int = 10
    before: str | None = None


class SearchRequest(BaseModel):
    """SDK thread-search body. ``metadata`` (e.g. graph_id/assistant_id) is
    accepted for protocol compatibility and ignored — single-agent deployment,
    scope is already per-user via the caller's identity."""

    metadata: dict[str, Any] | None = None
    limit: int = 10


class RunInput(BaseModel):
    messages: list[dict[str, Any]] = []


class RunRequest(BaseModel):
    """SDK run body. Only ``input.messages`` is load-bearing here; the rest are
    accepted for protocol compatibility and ignored (the agent resolves its own
    tools, and profile/resource context are accepted-loss). A run always
    creates-and-owns a non-existent thread, so ``if_not_exists`` is accepted and
    ignored rather than honored."""

    input: RunInput = RunInput()
    assistant_id: str | None = None
    stream_mode: Any = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    if_not_exists: str | None = None


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """The query the agent runs is the latest user turn in the run input.

    Checkpoint resume rebuilds prior context from the thread; the run body only
    needs to carry the new turn.
    """
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            return content if isinstance(content, str) else str(content)
    return ""


@thread_router.post("/threads")
async def create_thread(raw_request: Request) -> dict[str, Any]:
    _require_user(raw_request)
    now = datetime.now(UTC).isoformat()
    return {
        "thread_id": str(uuid.uuid4()),
        "created_at": now,
        "metadata": {},
        "status": "idle",
        "values": {},
    }


@thread_router.post("/threads/search")
async def search_threads(body: SearchRequest, raw_request: Request) -> list[dict[str, Any]]:
    """The conversation sidebar: the caller's own threads, most recent first.

    Authed-only IN EFFECT, not by a 401 gate: a caller with no verified
    identity has no user_hash to match, so there is nothing to enumerate and
    the correct response is an empty list, not a 401 (unlike every other
    route in this module).
    """
    user, _ = get_acting_user_from_cookie(raw_request)
    user_hash = _hash_user(user)
    if not user_hash:
        return []
    rows = get_turn_reporter().list_threads_for_user(user_hash, body.limit)
    return [
        {
            "thread_id": row["session_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "metadata": {},
            "status": "idle",
            "interrupts": {},
            "values": {"messages": [{"type": "human", "content": row["label"]}]},
        }
        for row in rows
    ]


@thread_router.post("/threads/{thread_id}/history")
async def thread_history(
    thread_id: str, body: HistoryRequest, raw_request: Request
) -> list[dict[str, Any]]:
    caller = _require_user(raw_request)
    _require_access(thread_id, caller)
    checkpointer = getattr(raw_request.app.state, "checkpointer", None)
    if checkpointer is None:
        return []
    graph = create_checkpointed_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    states = []
    async for snap in graph.aget_state_history(config, limit=body.limit):
        states.append(
            {
                "values": _serialize_values(snap.values),
                "checkpoint": snap.config.get("configurable", {}),
                "parent_checkpoint": (snap.parent_config or {}).get("configurable", {}),
                "metadata": snap.metadata or {},
                "created_at": snap.created_at,
            }
        )
    return states


def _serialize_values(values: dict[str, Any] | None) -> dict[str, Any]:
    if not values:
        return {}
    msgs = values.get("messages", [])
    return {"messages": [m.model_dump() if hasattr(m, "model_dump") else m for m in msgs]}


@thread_router.post("/threads/{thread_id}/runs/stream")
async def thread_run_stream(
    thread_id: str, body: RunRequest, raw_request: Request
) -> StreamingResponse:
    """Stream one agent turn on a thread as the LangGraph run protocol SSE.

    Ownership is atomic on the first run: ``claim_thread`` writes the ownership
    row transactionally (UNIQUE(thread_id)); a genuinely new thread's caller wins
    it, an existing thread must pass the ownership check. All status-code-bearing
    checks (401/404/409) run HERE, before StreamingResponse is returned — a status
    raised inside the generator would land after Starlette flushed 200.
    """
    # 1. Auth → 401.
    caller = _require_user(raw_request)

    # 2. Atomic ownership. claim_thread returns True iff this call created the row.
    #    "Create means create": new thread → caller owns it; existing thread →
    #    claim fails → must be the existing owner (else 404, indistinguishable
    #    from unknown).
    owned = get_thread_owner_store().claim_thread(thread_id, caller)
    if owned is False:
        _require_access(thread_id, caller)

    # 3. + 4. Acquire the per-thread run lock in the handler so a busy thread is a
    #    real 409 status, not an event flushed after a 200.
    run_id = str(uuid.uuid4())
    acquire_thread_run(thread_id, run_id)

    # The lock is released by held_run's finally, but ONLY once the generator is
    # actually started. An async generator that is never iterated never runs its
    # finally (verified against CPython/Starlette 1.0: GC of an un-started async
    # gen does not run finally), so any exception on the synchronous path between
    # acquire and returning a *started* response would leak the lock and 409-wedge
    # the thread until restart. Guard that whole gap in one try: release on any
    # exception before hand-off. release_thread_run is idempotent, so held_run's
    # own release on the normal (generator-started) path is unaffected.
    query = _last_user_text(body.input.messages)

    # Client-supplied UI hint, not server state: LangGraph SDK convention puts
    # per-run values under config.configurable; context is the newer top-level
    # alternative. Absent on both → None (accepted context-loss, per spec).
    cfg = (body.config or {}).get("configurable", {}) if body.config else {}
    resource_context = cfg.get("resource_context") or (body.context or {}).get("resource_context")

    async def _gen() -> AsyncGenerator[str, None]:
        async with held_run(thread_id, run_id):
            try:
                async with run_with_timeout(settings.AGENT_TURN_TIMEOUT_S):
                    yield format_sse_event("metadata", {"run_id": run_id, "attempt": 1})

                    from .routes import get_registry

                    registry = await get_registry()
                    accumulated: AIMessageChunk | None = None
                    final_messages: list[dict[str, Any]] = []

                    async for stream_type, chunk in stream_agent(
                        query=query,
                        session_id=thread_id,
                        question_id=run_id,
                        tool_catalog=registry.catalog,
                        acting_user=caller,
                        resource_context=resource_context,
                        profile=None,
                        checkpointer=getattr(raw_request.app.state, "checkpointer", None),
                    ):
                        if stream_type == "messages":
                            msg, metadata = chunk
                            if (
                                isinstance(msg, AIMessageChunk)
                                and metadata.get("langgraph_node") == "tool_calling_loop"
                            ):
                                accumulated = msg if accumulated is None else accumulated + msg
                                yield format_sse_event("messages/partial", [msg.model_dump()])
                        elif stream_type == "updates" and isinstance(chunk, dict):
                            for node_output in chunk.values():
                                if isinstance(node_output, dict):
                                    msgs = node_output.get("messages")
                                    if msgs:
                                        final_messages = [
                                            m.model_dump() if hasattr(m, "model_dump") else m
                                            for m in msgs
                                        ]

                    if accumulated is not None:
                        # Complete message reuses the stable streamed id.
                        final = AIMessage(content=accumulated.content, id=accumulated.id)
                        dumped = final.model_dump()
                        yield format_sse_event("messages/complete", [dumped])
                        if not final_messages:
                            final_messages = [dumped]

                    yield format_sse_event("values", {"messages": final_messages})
            except asyncio.CancelledError:
                # Real cancel (client disconnect / cancel_run). Emit, then propagate
                # so the task truly cancels; held_run's finally still releases.
                yield format_sse_event("error", {"error": "cancelled"})
                raise
            except TimeoutError:
                # asyncio.timeout expiry (a TimeoutError in 3.11, NOT CancelledError).
                # Emit and return; do not synthesize a CancelledError.
                yield format_sse_event("error", {"error": "timeout"})
                return
            except Exception as exc:
                logger.exception("Run stream failed: %s", exc)
                yield format_sse_event("error", {"error": "agent_error"})
                return

    # The lock was acquired synchronously above; it is released by held_run's
    # finally, but ONLY once the generator is actually started. An async
    # generator that is never iterated never runs its finally (verified against
    # CPython/Starlette 1.0: GC of an un-started async gen does not run finally),
    # so any failure on the synchronous path between acquire and returning a
    # started response would leak the lock and 409-wedge the thread until
    # restart. Guard that gap: release on any exception before hand-off.
    try:
        return StreamingResponse(
            _gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except BaseException:
        release_thread_run(thread_id, run_id)
        raise


@thread_router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_thread_run(thread_id: str, run_id: str, raw_request: Request) -> dict[str, str]:
    caller = _require_user(raw_request)
    _require_access(thread_id, caller)  # 404 if not the owner (never 403)
    if not cancel_run(thread_id, run_id):  # no such in-flight run ON THIS THREAD
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "cancelled"}
