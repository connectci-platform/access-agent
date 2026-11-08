"""LangGraph thread/run protocol endpoints: create + history (resume) + run.

The Bearer agent token is optional, not required: a caller without one is
anonymous, not rejected. Every route goes through require_thread_access in
src/api/thread_access.py, which is where access is decided.
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, AIMessageChunk
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from ..agent.context import AgentContext
from ..agent.graph import create_checkpointed_graph, stream_agent
from ..agent.turn_capture import get_turn_capture, reset_turn_capture
from ..config import settings
from ..turn_reporter import _hash_user, get_turn_reporter
from ..turnstile import charge_query, check_turnstile
from .answer_tokens import ANSWER_MODEL_NODE, is_answer_token
from .handoff_tokens import (
    DEFAULT_TTL_SECONDS,
    exchange_handoff_token,
    mint_handoff_token,
)
from .identity import ClaimKey, Identity, require_identity
from .sse import format_sse_event
from .thread_access import require_thread_access
from .thread_runs import (
    acquire_thread_run,
    cancel_run,
    held_run,
    release_thread_run,
    run_with_timeout,
)

logger = logging.getLogger(__name__)

thread_router = APIRouter()


class HistoryRequest(BaseModel):
    limit: int = 10
    before: str | None = None


class ExchangeRequest(BaseModel):
    token: str


class SearchRequest(BaseModel):
    """SDK thread-search body. ``metadata`` (e.g. graph_id/assistant_id) is
    accepted for protocol compatibility and ignored — single-agent deployment,
    scope is already per-user via the caller's identity."""

    metadata: dict[str, Any] | None = None
    limit: int = 10


class RunInput(BaseModel):
    messages: list[dict[str, Any]] = []


class RunRequest(BaseModel):
    """SDK run body. The route reads ``input.messages``, ``turnstile_token`` (top
    level, or in ``context`` because the SDK's submit can't add a top-level field),
    and ``resource_context`` from ``config.configurable`` or ``context``; everything
    else is accepted for protocol compatibility and ignored. The body's
    ``context`` never reaches the agent's runtime context, which comes from the
    verified token only, so a body claiming a user or scopes has no effect. A
    run always creates-and-owns a non-existent thread, so ``if_not_exists`` is
    accepted and ignored rather than honored."""

    input: RunInput = RunInput()
    assistant_id: str | None = None
    stream_mode: Any = None
    config: dict[str, Any] | None = None
    context: dict[str, Any] | None = None
    if_not_exists: str | None = None
    turnstile_token: str | None = None


async def _write_turn_report(
    *,
    final_state: dict[str, Any],
    thread_id: str,
    run_id: str,
    query: str,
    start_time: float,
    caller: str | None,
    success: bool,
) -> None:
    """Turn report (denormalized read model). Off the response path; never
    raises into the stream. Mirrors routes.py:454-483 (success) /
    routes.py:500-527 (failure). acting_user == `caller` here; session_id ==
    thread_id (the thread/run path keys the reporter by thread_id, exactly as
    stream_agent is called with session_id=thread_id). Called OUTSIDE
    run_with_timeout/held_run — see thread_run_stream's placement note.
    """
    try:
        from ..agent.domains.capabilities import get_capability_registry as _cap_reg
        from ..services.resource_matcher import resources_for_turn
        from ..turn_reporter import get_turn_reporter

        reporter = get_turn_reporter()
        prior_turns = await asyncio.to_thread(reporter.count_turns_for_session, thread_id)
        turn_index = prior_turns + 1 if prior_turns is not None else None  # None on DB error → NULL
        resources = await resources_for_turn(query, str(final_state.get("final_answer") or ""))
        await asyncio.to_thread(
            reporter.log_turn_report,
            final_state=final_state,
            session_id=thread_id,
            turn_index=turn_index,
            question_id=run_id,
            query_text=query,
            duration_ms=(time.time() - start_time) * 1000,
            acting_user=caller,
            success=success,
            capabilities=_cap_reg().infer_capability_ids(final_state.get("tool_results", [])),
            resources=resources,
            turn_capture=get_turn_capture(),
            judge=None,
        )
    except Exception:
        logger.exception("Turn report write failed")


async def _stream_run_events(
    *,
    thread_id: str,
    run_id: str,
    query: str,
    context: AgentContext,
    resource_context: str | None,
    checkpointer: Any,
    final_state: dict[str, Any],
) -> AsyncGenerator[str, None]:
    """The metadata/messages/custom/values SSE sequence for one run, mutating
    ``final_state`` in place (mirrors turn_capture's in-place-mutation pattern)
    so the caller can read it after this generator is exhausted. Extracted
    from thread_run_stream's ``_gen`` to keep both under ruff's branch/statement
    caps once the turn-report write was added.
    """
    yield format_sse_event("metadata", {"run_id": run_id, "attempt": 1})

    from .routes import get_registry

    registry = await get_registry()
    # Answer chunks by message id, in arrival order, for the values fallback below.
    streamed: dict[str, AIMessageChunk] = {}
    final_messages: list[dict[str, Any]] = []

    async for stream_type, chunk in stream_agent(
        query=query,
        session_id=thread_id,
        question_id=run_id,
        tool_catalog=registry.catalog,
        context=context,
        resource_context=resource_context,
        profile=None,
        checkpointer=checkpointer,
    ):
        if stream_type == "custom":
            if (
                isinstance(chunk, dict)
                and chunk.get("type") == "status"
                and isinstance(chunk.get("message"), str)
            ):
                yield format_sse_event("custom", {"type": "status", "message": chunk["message"]})
        elif stream_type == "messages":
            msg, metadata = chunk
            if is_answer_token(msg, metadata):
                key = msg.id or ""
                streamed[key] = streamed[key] + msg if key in streamed else msg
                if not streamed[key].text.strip():
                    # Whitespace before any visible text (such as the "\n\n" after
                    # a think block) would open a blank bubble. Held back, and not
                    # resent: the final values event replaces the text.
                    continue
                # The SDK reads event `messages` with data [message, metadata].
                yield format_sse_event(
                    "messages", [msg.model_dump(), {"langgraph_node": ANSWER_MODEL_NODE}]
                )
        elif stream_type == "updates" and isinstance(chunk, dict):
            for node_output in chunk.values():
                if isinstance(node_output, dict):
                    final_state.update(node_output)
                    msgs = node_output.get("messages")
                    if msgs:
                        final_messages = [
                            m.model_dump() if hasattr(m, "model_dump") else m for m in msgs
                        ]

    refusals = final_state.get("tool_refusals") or []
    if refusals:
        # The caller asked for something they may not do here. Sent as a
        # LangGraph `custom` event because useStream drops event names it doesn't
        # know; `type` keeps it apart from the ui events on the same channel. The
        # turn still completes below with the model's reply.
        yield format_sse_event("custom", {"type": "requires_auth", "reason": refusals[0]["reason"]})

    if streamed and not final_messages:
        # The model produced tokens but no node update carried messages. One AI
        # message per streamed id, so separate rounds are never merged.
        final_messages = [
            AIMessage(content=c.content, id=c.id or None).model_dump() for c in streamed.values()
        ]

    yield format_sse_event("values", {"messages": final_messages})


def _extract_text_content(content: Any) -> str:
    """Render a message's ``content`` field as plain text.

    Callers send content as either a plain string or a LangChain-style list of
    content blocks (``[{"type": "text", "text": "..."}, ...]``). Join the text
    of each text block; non-text blocks (images, etc.) are ignored. Anything
    else (None, a dict, ...) yields "" rather than a stringified repr.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(part for part in parts if part)
    return ""


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    """The query the agent runs is the latest user turn in the run input.

    Checkpoint resume rebuilds prior context from the thread; the run body only
    needs to carry the new turn.

    A message is a user turn if it has ``role: "user"`` (the widget's shape)
    or ``type: "human"`` (the LangChain message shape the full-screen client
    sends). Its ``content`` may be a plain string or a block-array (the
    full-screen client sends ``[{"type": "text", "text": "..."}]``); only the
    text blocks are extracted.
    """
    for msg in reversed(messages):
        if msg.get("type") == "human" or msg.get("role") == "user":
            return _extract_text_content(msg.get("content", ""))
    return ""


@thread_router.post("/threads")
async def create_thread() -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "thread_id": str(uuid.uuid4()),
        "created_at": now,
        "metadata": {},
        "status": "idle",
        "values": {},
    }


@thread_router.post("/threads/search")
async def search_threads(body: SearchRequest, identity: Identity) -> list[dict[str, Any]]:
    """The conversation sidebar: the caller's own threads, most recent first.

    Authed-only IN EFFECT, not by a 401 gate: a caller with no verified
    identity has no user_hash to match, so there is nothing to enumerate and
    the correct response is an empty list, not a 401 (unlike every other
    route in this module).
    """
    user_hash = _hash_user(identity.sub if identity else None)
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
    thread_id: str,
    body: HistoryRequest,
    raw_request: Request,
    identity: Identity,
    claim_key: ClaimKey = None,
) -> list[dict[str, Any]]:
    require_thread_access(thread_id, identity.sub if identity else None, claim_key)
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


@thread_router.post("/threads/{thread_id}/claim")
async def claim_thread_on_sign_in(
    thread_id: str, identity: Identity, claim_key: ClaimKey = None
) -> dict[str, str]:
    """Link a conversation started signed out to the caller who just signed in.

    Repeating this as the owner succeeds, which also retries a failed backfill.
    """
    require_thread_access(thread_id, require_identity(identity), claim_key, write=True)
    return {"thread_id": thread_id}


@thread_router.post("/threads/{thread_id}/handoff")
async def mint_handoff(thread_id: str, identity: Identity) -> dict[str, Any]:
    """Mint a single-use cross-surface handoff token for an authed-owned thread.

    Owners only: a handoff carries the thread into a URL, and an unclaimed
    thread has no owner for the exchange to bind to.
    """
    caller = require_identity(identity)
    require_thread_access(thread_id, caller, None)  # no key, so only the owner passes
    token = mint_handoff_token(thread_id, _hash_user(caller))  # type: ignore[arg-type]
    return {"token": token, "expires_in": DEFAULT_TTL_SECONDS}


@thread_router.post("/handoff/exchange")
async def exchange_handoff(body: ExchangeRequest, identity: Identity) -> dict[str, str]:
    """Redeem a single-use handoff token for a thread_id, identity-bound.

    The token is consumed atomically on exchange regardless of outcome; a
    mismatching caller burning someone else's token only harms an attacker.
    Returns ONLY thread_id — never message content (that comes from the
    separately-gated /history).
    """
    caller = require_identity(identity)
    result = exchange_handoff_token(body.token)
    if result is None:
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    thread_id, owner_hash = result
    if owner_hash != _hash_user(caller):
        raise HTTPException(status_code=404, detail="Invalid or expired token")
    return {"thread_id": thread_id}


@thread_router.post("/threads/{thread_id}/runs/stream", response_model=None)
async def thread_run_stream(
    thread_id: str,
    body: RunRequest,
    raw_request: Request,
    identity: Identity,
    claim_key: ClaimKey = None,
) -> StreamingResponse | JSONResponse:
    """Stream one agent turn on a thread as the LangGraph run protocol SSE.

    Ownership is atomic on the first run: ``require_thread_access`` writes the
    ownership row transactionally (UNIQUE(thread_id)), so one caller wins a new
    thread and an existing thread must pass the check. All status-code-bearing
    checks run before StreamingResponse is returned: the token's 401/503 in
    ``request_identity``, and 404/409 HERE. A status raised inside the generator
    would land after Starlette flushed 200.
    """
    # 1. Anon-accept: caller may be None; ownership below still fails closed.
    caller = identity.sub if identity else None

    # 2. Check access. For a new thread this also writes the ownership row.
    require_thread_access(thread_id, caller, claim_key, write=True, create=True)

    # Turnstile gate — ported from routes.py's /query (see src/turnstile.py).
    # Authed callers skip immediately inside check_turnstile. The thread_id is
    # the key the guard tracks free-query/verification state under, as
    # session_id is on /query.
    # Unlike /query, there is no "session_id required" 400 to mirror: thread_id
    # always exists (it's in the URL path), so the guard always has a key to
    # track. Placed AFTER the ownership gate and BEFORE acquire_thread_run so a
    # challenged request never acquires the run lock.
    turnstile_token = body.turnstile_token or (body.context or {}).get("turnstile_token")
    turnstile_response = await check_turnstile(caller, thread_id, turnstile_token)
    if turnstile_response is not None:
        return turnstile_response

    query = _last_user_text(body.input.messages)

    # Client-supplied UI hint, not server state: LangGraph SDK convention puts
    # per-run values under config.configurable; context is the newer top-level
    # alternative. Absent on both → None (accepted context-loss, per spec).
    cfg = (body.config or {}).get("configurable", {}) if body.config else {}
    resource_context = cfg.get("resource_context") or (body.context or {}).get("resource_context")

    # 3. + 4. Acquire the per-thread run lock in the handler so a busy thread is a
    #    real 409 status, not an event flushed after a 200.
    run_id = str(uuid.uuid4())
    acquire_thread_run(thread_id, run_id)
    # Same guard and key as /query, where session_id == thread_id, so a handed-off
    # conversation keeps its count.
    charge = charge_query(caller, thread_id)

    async def _gen() -> AsyncGenerator[str, None]:
        start_time = time.time()
        reset_turn_capture()
        final_state: dict[str, Any] = {}
        success = False

        async def _report(*, state: dict[str, Any], ok: bool) -> None:
            await _write_turn_report(
                final_state=state,
                thread_id=thread_id,
                run_id=run_id,
                query=query,
                start_time=start_time,
                caller=caller,
                success=ok,
            )

        async with held_run(thread_id, run_id):
            try:
                async with run_with_timeout(settings.AGENT_TURN_TIMEOUT_S):
                    async for event in _stream_run_events(
                        thread_id=thread_id,
                        run_id=run_id,
                        query=query,
                        context=AgentContext.from_identity(identity),
                        resource_context=resource_context,
                        checkpointer=getattr(raw_request.app.state, "checkpointer", None),
                        final_state=final_state,
                    ):
                        yield event
                    success = True
            except asyncio.CancelledError:
                # Real cancel (client disconnect / cancel_run). Emit, then propagate
                # so the task truly cancels; held_run's finally still releases. No
                # turn-report write here — mirrors /query, which also re-raises on
                # cancel without writing a failed-turn row.
                yield format_sse_event("error", {"error": "cancelled"})
                raise
            except TimeoutError:
                # asyncio.timeout expiry (a TimeoutError in 3.11, NOT CancelledError).
                # Emit and return; do not synthesize a CancelledError.
                charge.refund()
                yield format_sse_event("error", {"error": "timeout"})
                await _report(state=final_state, ok=False)
                return
            except Exception as exc:
                logger.exception("Run stream failed: %s", exc)
                charge.refund()
                yield format_sse_event("error", {"error": "agent_error"})
                await _report(state=final_state, ok=False)
                return

        # Turn report write lives OUTSIDE both run_with_timeout and held_run — a
        # slow reporter write must not trip a spurious TimeoutError into an
        # already-successful stream, as on /query.
        if success:
            await _report(state=final_state, ok=True)

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
        charge.refund()
        raise


@thread_router.post("/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_thread_run(
    thread_id: str, run_id: str, identity: Identity, claim_key: ClaimKey = None
) -> dict[str, str]:
    require_thread_access(thread_id, identity.sub if identity else None, claim_key)
    if not cancel_run(thread_id, run_id):  # no such in-flight run ON THIS THREAD
        raise HTTPException(status_code=404, detail="Run not found")
    return {"status": "cancelled"}
