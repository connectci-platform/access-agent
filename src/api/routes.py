"""FastAPI routes for the ACCESS Documentation Agent."""

import asyncio
import logging
import secrets
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import StreamingResponse

from ..agent.context import AgentContext
from ..agent.graph import stream_agent
from ..agent.profile import UserProfile
from ..agent.turn_capture import get_turn_capture, reset_turn_capture
from ..config import settings
from ..tools import ToolRegistry, get_catalog_aggregator
from ..turnstile import NOT_CHARGED, QueryCharge, charge_query, check_turnstile
from ..usage_logger import get_usage_logger
from .answer_tokens import is_answer_token
from .identity import ClaimKey, Identity
from .sse import format_sse_event as _format_sse_event
from .thread_access import require_thread_access
from .thread_runs import acquire_thread_run, held_run, release_thread_run, run_with_timeout

logger = logging.getLogger(__name__)

router = APIRouter()

# Check if checkpointing is enabled
USE_CHECKPOINTING = bool(settings.DATABASE_URL)

# Headers the redteam (PyRIT) harness sends so its prompts are recorded as a
# redteam run instead of polluting real-traffic views. The harness hits the
# same /api/v1/query door as a real user, so the header is the only signal.
# See access-redteam's AccessAgentTarget.
REDTEAM_HEADER = "X-Redteam"
REDTEAM_RUN_ID_HEADER = "X-Redteam-Run-Id"
REDTEAM_SUITE_HEADER = "X-Redteam-Suite"


def redteam_report_context(headers: Headers) -> dict[str, Any]:
    """Map redteam request headers to turn_report tagging kwargs.

    Returns ``{}`` for ordinary traffic. When the harness's ``X-Redteam``
    header is present, returns the ``origin`` / ``battery_id`` /
    ``battery_run_id`` kwargs so the prompt is tagged as a redteam run the
    reporting dashboard can group and keep out of real-traffic views. The
    grouping ids are best-effort; ``origin`` is the load-bearing signal.
    """
    if headers.get(REDTEAM_HEADER) is None:
        return {}

    # Header values are client-supplied; the columns are VARCHAR(64). Truncate
    # so an oversized grouping id can't raise on insert and drop the whole turn
    # report — the id is best-effort, origin is the load-bearing signal.
    def _clip(value: str | None) -> str | None:
        return value[:64] if value is not None else None

    return {
        "origin": "redteam",
        "battery_id": _clip(headers.get(REDTEAM_SUITE_HEADER)),
        "battery_run_id": _clip(headers.get(REDTEAM_RUN_ID_HEADER)),
    }


# Global registry - built from aggregated catalog, plus the catalog object it
# was built from. The aggregator swaps in a new catalog object on every build
# (a refresh or a background recovery), so identity tells us when to rebuild.
_registry: ToolRegistry | None = None
_registry_catalog: dict[str, Any] | None = None


async def get_registry() -> ToolRegistry:
    """Get the tool registry for the aggregator's current catalog.

    Rebuilt whenever the aggregator has swapped in a new catalog since the
    last build, so tools a refresh or the background recovery brought back
    reach the next request (#240, D8-3020).

    Guarded: a catalog that caught every MCP server down (a network blip —
    per-server failures don't fail the fetch) is not adopted over a registry
    we already have, or an unauthenticated refresh timed during a blip would
    strip the agent's tools until the next refresh or restart. /health still
    goes degraded, so the blip is visible.
    """
    global _registry, _registry_catalog
    catalog = await get_catalog_aggregator().fetch_catalog()
    if catalog is _registry_catalog and _registry is not None:
        return _registry
    if _registry is not None and catalog.get("servers_available", 0) <= 0:
        logger.warning("Catalog has no available servers; keeping the agent's current registry")
        return _registry
    _registry = ToolRegistry(catalog=catalog)
    _registry_catalog = catalog
    logger.info(f"Built tool registry with {_registry.tool_count} tools from aggregated catalog")
    return _registry


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="The user's question")
    session_id: str | None = Field(None, description="Session ID for conversation tracking")
    question_id: str | None = Field(None, description="Unique question ID")
    turnstile_token: str | None = Field(None, description="Cloudflare Turnstile response token")
    resource_context: str | None = Field(
        None, description="RP slug for resource-scoped queries (e.g. 'delta')"
    )
    profile: UserProfile | None = Field(
        None,
        description="Optional user profile hint (allocated resources); "
        "steers retrieval, authorizes nothing",
    )


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    success: bool
    response: str
    session_id: str
    question_id: str
    tools_used: list[str]
    confidence: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _domain_is_final(result: Any) -> bool:
    """Check whether a domain agent response is final or mid-conversation.

    Uses the domain_completed flag from the domain agent node, which checks
    whether any MCP tools were called during the ReAct loop. If tools were
    called, the agent took action (created a ticket, posted an announcement)
    and the response is final. If no tools were called, the agent is still
    gathering info (asking for email, clarifying the issue).

    Falls back to True if domain_completed is not set (e.g., non-domain responses).
    """
    return result.get("domain_completed") is not False


async def _check_capability_discovery(
    query: str,
    authenticated: bool,
    session_id: str,
    question_id: str,
    resource_context: str | None = None,
) -> QueryResponse | None:
    """Return a direct response for capability discovery queries.

    The frontend sends category labels ("Get help", "Explore resources") and
    "Show my options" as messages.  These are answered from the capability
    registry — no LLM or RAG call needed.

    When ``resource_context`` is set, uses RP-scoped capabilities.
    """
    from ..agent.domains.capabilities import get_capability_registry

    registry = get_capability_registry()
    # Strip lock emoji prefix that the frontend adds to auth-required buttons
    normalized = query.strip().removeprefix("🔒").strip().lower()

    # Get categories — scoped or general
    if resource_context:
        scoped = await registry.get_by_category_scoped(resource_context, authenticated)
        categories = scoped["categories"] if scoped else registry.get_by_category(authenticated)
    else:
        categories = registry.get_by_category(authenticated)

    # "Show my options" → list capabilities with example queries from descriptions
    if normalized in ("show my options", "what can you do", "what can you help with"):
        lines = ["Here are some things you can try:\n"]
        for cat in categories:
            # Skip "general" — typing is the default
            if cat["id"] == "general":
                continue
            lines.append(f"**{cat['label']}**")
            for cap in cat["capabilities"]:
                example = cap.get("example_query") or cap["description"]
                locked = " 🔒 (login required)" if cap.get("locked") else ""
                lines.append(f'- *"{example}"*{locked}')
            lines.append("")
        if not authenticated:
            lines.append("*Some features require logging in. Log in to unlock all capabilities.*")
        lines.append("Just type a question like one of these, or ask anything else!")
        answer = "\n".join(lines)
        return QueryResponse(
            success=True,
            response=answer,
            session_id=session_id,
            question_id=question_id,
            tools_used=[],
            confidence="high",
            metadata={
                "agent": "capability-discovery",
                "capability_id": "ask_question",
                "is_final_response": True,
                "rating_target": None,
                "question_id": question_id,
            },
        )

    return None


def _should_stream_token(msg: Any, metadata: dict[str, Any]) -> bool:
    """Whether a message chunk should reach the browser as a token event."""
    return is_answer_token(msg, metadata)


async def _write_query_turn_report(
    *,
    final_state: dict[str, Any],
    request: QueryRequest,
    session_id: str,
    question_id: str,
    duration_ms: float,
    acting_user: str | None,
    success: bool,
    report_context: dict[str, Any] | None,
) -> None:
    """Turn report (denormalized read model for the reporting dashboard).

    Off the response path and never raises into the stream. ``final_state`` may
    be partial when the turn failed or timed out; ``_assemble_turn_report``
    tolerates missing keys, and the judge is skipped.
    """
    try:
        from ..agent.domains.capabilities import get_capability_registry as _cap_reg
        from ..services.resource_matcher import resources_for_turn
        from ..turn_reporter import get_turn_reporter

        reporter = get_turn_reporter()
        prior_turns = await asyncio.to_thread(reporter.count_turns_for_session, session_id)
        # None = count unknown (DB error): write NULL, not a wrong "1".
        turn_index = prior_turns + 1 if prior_turns is not None else None
        resources = await resources_for_turn(
            request.query, str(final_state.get("final_answer") or "")
        )
        await asyncio.to_thread(
            reporter.log_turn_report,
            final_state=final_state,
            session_id=session_id,
            turn_index=turn_index,
            question_id=question_id,
            query_text=request.query,
            duration_ms=duration_ms,
            acting_user=acting_user,
            success=success,
            capabilities=_cap_reg().infer_capability_ids(final_state.get("tool_results", [])),
            resources=resources,
            turn_capture=get_turn_capture(),
            judge=None,
            **(report_context or {}),
        )
    except Exception:
        logger.exception("Turn report write failed")


async def _stream_events(  # noqa: PLR0912, PLR0915
    request: QueryRequest,
    context: AgentContext,
    session_id: str,
    question_id: str,
    include_trace: bool,
    raw_request: Request,
    report_context: dict[str, Any] | None = None,
    run_id: str | None = None,
    charge: QueryCharge = NOT_CHARGED,
) -> AsyncGenerator[str, None]:
    """Translate LangGraph stream chunks into SSE events.

    Yields SSE-formatted strings for status updates, LLM tokens,
    and a final done event with response metadata.

    The stream-consumption loop runs under the shared per-thread run lock
    (``held_run``) and turn bound (``run_with_timeout``) so the widget and the
    fullscreen ``runs/stream`` endpoint can't both resume the same thread at
    once. ``run_id`` is the lock token acquired in the handler; when absent (a
    direct unit-test call), ``held_run`` still registers/releases a fresh token
    harmlessly.

    ``context`` is the run's identity, built from the verified token only.
    """
    acting_user = context.acting_user
    start_time = time.time()
    final_state: dict[str, Any] = {}
    reset_turn_capture()
    lock_run_id = run_id or str(uuid.uuid4())
    timed_out = False
    # The widget concatenates every token and then ignores done.response, so
    # what is streamed has to read as the whole answer.
    streamed = ""
    last_chunk_id: str | None = None

    async def _report(*, success: bool, duration_ms: float | None = None) -> None:
        await _write_query_turn_report(
            final_state=final_state,
            request=request,
            session_id=session_id,
            question_id=question_id,
            duration_ms=duration_ms
            if duration_ms is not None
            else (time.time() - start_time) * 1000,
            acting_user=acting_user,
            success=success,
            report_context=report_context,
        )

    try:
        async with held_run(session_id, lock_run_id):
            try:
                # get_registry() hits MCP servers and can fail on a cold/refreshing
                # catalog cache. Keep it INSIDE held_run so a failure hits held_run's
                # finally and releases the lock the handler acquired — otherwise the
                # session_id is permanently 409-locked until restart.
                registry = await get_registry()
                async with run_with_timeout(settings.AGENT_TURN_TIMEOUT_S):
                    async for stream_type, chunk in stream_agent(
                        query=request.query,
                        session_id=session_id,
                        question_id=question_id,
                        tool_catalog=registry.catalog,
                        context=context,
                        resource_context=request.resource_context,
                        profile=request.profile,
                        checkpointer=getattr(raw_request.app.state, "checkpointer", None),
                        use_checkpointing=USE_CHECKPOINTING,
                        db_uri=settings.DATABASE_URL if USE_CHECKPOINTING else None,
                    ):
                        if stream_type == "custom":
                            # Status messages from nodes via get_stream_writer()
                            if isinstance(chunk, dict) and chunk.get("type") == "status":
                                yield _format_sse_event("status", {"message": chunk["message"]})

                        elif stream_type == "messages":
                            # LLM token chunks — tuple of (message, metadata)
                            msg, metadata = chunk
                            # Tokens come from the model nested in tool_calling_loop.
                            # LangGraph's messages stream emits both AIMessageChunk
                            # (tokens) and AIMessage (complete messages added to state).
                            # We only want the chunks to avoid duplicating the response.
                            if _should_stream_token(msg, metadata):
                                # Each model round has its own message id; a chunk
                                # without one continues the previous round.
                                chunk_id = getattr(msg, "id", None)
                                if streamed and chunk_id and chunk_id != last_chunk_id:
                                    streamed += "\n\n"
                                    yield _format_sse_event("token", {"content": "\n\n"})
                                if chunk_id:
                                    last_chunk_id = chunk_id
                                streamed += msg.content
                                yield _format_sse_event("token", {"content": msg.content})

                        elif stream_type == "updates" and isinstance(chunk, dict):
                            # State updates after each node — collect for final metadata
                            for node_output in chunk.values():
                                if isinstance(node_output, dict):
                                    final_state.update(node_output)
            except TimeoutError:
                # Turn bound expired (TimeoutError in 3.11, not CancelledError).
                # Emit the widget's error+done and return; do not synthesize a
                # CancelledError. held_run's finally releases the lock.
                charge.refund()
                yield _format_sse_event("error", {"message": "Query timed out", "code": "timeout"})
                yield _format_sse_event("done", {"success": False, "error": "timeout"})
                timed_out = True

        # Reported after held_run, like the success path, so a slow report
        # write doesn't keep the thread locked.
        if timed_out:
            await _report(success=False)
            return

        # Build metadata from final state (mirrors non-streaming QueryResponse fields)
        final_answer = final_state.get("final_answer") or "No answer generated"
        # An apology written after the model stopped was never streamed, and the
        # widget ignores done.response once any token arrived.
        if (
            final_state.get("answer_unavailable")
            and streamed
            and not streamed.endswith(final_answer)
        ):
            yield _format_sse_event("token", {"content": f"\n\n{final_answer}"})
        tools_used = final_state.get("tools_used", [])
        # Invocations, not distinct names; older state lacks the field.
        tool_call_count = final_state.get("tool_call_count")
        if tool_call_count is None:
            tool_call_count = len(tools_used)
        duration_ms = (time.time() - start_time) * 1000

        query_classification = final_state.get("query_classification")
        query_type = query_classification.query_type if query_classification else None
        confidence = None
        query_analysis = final_state.get("query_analysis")
        if query_analysis:
            confidence = query_analysis.confidence

        # Resolve capability_id
        from ..agent.domains.capabilities import get_capability_registry

        cap_registry = get_capability_registry()
        capability_id = None
        cap_category = None
        if query_classification and query_classification.capability_id:
            capability_id = query_classification.capability_id
        else:
            domain = query_classification.domain if query_classification else None
            capability_id = cap_registry.infer_capability_id(domain, tools_used)
        cap = cap_registry.get_by_id(capability_id) if capability_id else None
        if cap:
            cap_category = cap.category

        is_final = _domain_is_final(final_state)
        rating_target: str | None
        if not is_final:
            rating_target = None
        elif tools_used == ["search_access_documents"]:
            rating_target = "uky_rag"
        else:
            rating_target = "agent"

        # Build classification info
        classification_info: dict[str, Any] | None = None
        if query_classification:
            classification_info = {
                "query_type": query_classification.query_type,
                "confidence": query_classification.confidence,
                "domain": query_classification.domain,
                "reason": query_classification.reason[:200] if query_classification.reason else "",
            }

        done_data: dict[str, Any] = {
            "success": True,
            "response": final_answer,
            "session_id": session_id,
            "question_id": question_id,
            "confidence": confidence,
            "metadata": {
                "agent": "access-documentation-langgraph",
                "tool_count": tool_call_count,
                "tools_used": tools_used,
                "checkpointing_enabled": USE_CHECKPOINTING,
                "duration_ms": duration_ms,
                "classification": classification_info,
                "capability_id": capability_id,
                "is_final_response": is_final,
                "rating_target": rating_target,
                "question_id": question_id,
                **({"node_trace": final_state.get("node_trace", [])} if include_trace else {}),
            },
        }
        refusals = final_state.get("tool_refusals") or []
        if refusals:
            # The caller asked for something they may not do here. The turn
            # still completes with the model's reply; this annotates it. A
            # write_scope caller is already signed in, so no sign-in is offered.
            reason = refusals[0]["reason"]
            refusal = {"reason": reason}
            if reason != "write_scope":
                refusal["login_url"] = settings.LOGIN_URL
            yield _format_sse_event("requires_auth", refusal)
            done_data["metadata"]["refusal"] = refusal
        yield _format_sse_event("done", done_data)

        # The answer is delivered. Nothing below may fail the turn: a failure
        # here would otherwise reach the handler below, which reports the turn
        # as failed and sends a second done.
        try:
            usage_logger = get_usage_logger()
            await asyncio.to_thread(
                usage_logger.log_query,
                query_text=request.query,
                session_id=session_id,
                question_id=question_id,
                query_type=query_type,
                confidence=confidence,
                tools_used=tools_used,
                tool_call_count=tool_call_count,
                duration_ms=duration_ms,
                response_length=len(final_answer),
                acting_user=acting_user,
                success=True,
                capability_id=capability_id,
                category=cap_category,
            )
        except Exception:
            logger.exception("Usage logging failed")

        await _report(success=True, duration_ms=duration_ms)

    except asyncio.CancelledError:
        # Client disconnect / cancel. held_run's finally already released the
        # per-thread lock; propagate so the task truly cancels (never swallow).
        raise
    except Exception as e:
        logger.exception(f"Stream failed: {e}")
        charge.refund()
        # A success=False row lets the dashboard tell a failed query apart from
        # one that never happened.
        await _report(success=False)
        yield _format_sse_event(
            "error", {"message": "Failed to process query", "code": "agent_error"}
        )
        # Always yield done so clients can finalize the stream
        yield _format_sse_event("done", {"success": False, "error": str(e)})


@router.post("/query", response_model=None)
async def query_agent(
    request: QueryRequest,
    raw_request: Request,
    identity: Identity,
    claim_key: ClaimKey = None,
    include_trace: bool = Query(False, description="Include node_trace in response metadata"),
) -> QueryResponse | JSONResponse | StreamingResponse:
    """Execute a query against the ACCESS Documentation Agent.

    User identity comes only from the Bearer agent token (``request_identity``).
    There is no body-supplied identity fallback: a caller without a token is
    always anonymous.

    Args:
        request: The query request with question and optional IDs.
        raw_request: The raw FastAPI request (app state and redteam headers).

    Returns:
        For capability discovery: QueryResponse (JSON) with instant answer.
        For agent queries: StreamingResponse (SSE) with status, token, and done events.
    """
    # No body-supplied identity fallback: without a token, the caller is
    # anonymous. A bad token was already refused with a 401.
    user = identity.sub if identity else None
    acting_user: str | None = user

    # Generate IDs if not provided.
    # Anonymous users MUST provide session_id when Turnstile is enabled —
    # otherwise each request gets a unique ID and the free-query counter
    # never accumulates.
    timestamp = int(time.time() * 1000)
    if not request.session_id and not acting_user and settings.turnstile_enabled:
        raise HTTPException(
            status_code=400,
            detail="session_id is required for anonymous queries",
        )
    # A server-minted session_id is the credential for an anon /query session
    # (see _query_may_continue in thread_access.py). It MUST be unguessable: a
    # `sess_{timestamp}` fallback let anyone who knew the approximate request
    # time reconstruct another user's anon thread id and read/continue their
    # conversation. token_urlsafe(24) is 192 bits of entropy.
    session_id = request.session_id or f"sess_{secrets.token_urlsafe(24)}"
    question_id = request.question_id or f"q_{timestamp}"  # not a security id

    logger.info(
        f"Processing query: {request.query[:50]}... "
        f"(session={session_id}, acting_user={acting_user or 'anonymous'})"
    )

    # Turnstile gate — anonymous users may need to verify they're human.
    # Authenticated users (valid agent token) skip entirely.
    turnstile_response = await check_turnstile(acting_user, session_id, request.turnstile_token)
    if turnstile_response is not None:
        return turnstile_response

    # ── Capability discovery short-circuit ────────────────────────────
    # Category labels and "Show my options" come from the dynamic buttons.
    # Answer them directly from the registry — no LLM/RAG call needed.
    # Deliberately bypasses usage logging and Turnstile free-query counting
    # since discovery is navigation with no LLM/MCP cost.
    discovery_response = await _check_capability_discovery(
        request.query,
        acting_user is not None,
        session_id,
        question_id,
        resource_context=request.resource_context,
    )
    if discovery_response is not None:
        return discovery_response

    # Redteam (PyRIT) prompts come through this same door as real users; the
    # X-Redteam header is the only signal that lets us tag them instead of
    # polluting real-traffic views.
    report_context = redteam_report_context(raw_request.headers)

    # The same check as the thread routes, plus /query's exception for the
    # widget and non-browser callers. It creates a new thread's ownership row, so
    # every thread is owned from turn one. `user` and `acting_user` are the same
    # token-verified identity; there is no body-supplied identity.
    require_thread_access(session_id, user, claim_key, write=True, create=True, via_query=True)

    # Share the per-thread run lock with the runs/stream endpoint (they collide on
    # session_id == thread_id and both do the read-modify-write resume). Acquire in
    # the handler so a busy thread is a real 409, not an event flushed after 200.
    # A 409 on the widget is new (rare same-user cross-surface race), acceptable.
    run_id = str(uuid.uuid4())
    acquire_thread_run(session_id, run_id)
    charge = charge_query(acting_user, session_id)

    # The lock is released by held_run's finally inside _stream_events, but ONLY
    # once that generator is actually started. An async generator that is never
    # iterated never runs its finally, so any exception on the synchronous path
    # between acquire and returning a started response would leak the lock and
    # 409-wedge the thread until restart. Guard that gap: release on any exception
    # before hand-off. release_thread_run is idempotent, so held_run's own release
    # on the normal path is unaffected.
    try:
        # Agent queries stream via SSE
        return StreamingResponse(
            _stream_events(
                request,
                AgentContext.from_identity(identity),
                session_id,
                question_id,
                include_trace,
                raw_request,
                report_context,
                run_id,
                charge,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )
    except BaseException:
        release_thread_run(session_id, run_id)
        charge.refund()
        raise


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint with tool catalog status."""
    result: dict[str, Any] = {
        "status": "healthy",
        "agent": "access-documentation-langgraph",
        "version": "0.1.0",
        "checkpointing_enabled": USE_CHECKPOINTING,
    }

    # Include catalog status if available
    aggregator = get_catalog_aggregator()
    catalog = aggregator.catalog
    if catalog:
        total = catalog.get("total_servers", 0)
        available = catalog.get("servers_available", 0)
        result["tools"] = {
            "total": catalog.get("total_tools", 0),
            "servers_total": total,
            "servers_available": available,
        }
        if available < total:
            result["status"] = "degraded"
            result["tools"]["unavailable_servers"] = [
                s.get("server", s.get("name", "unknown"))
                for s in catalog.get("servers", [])
                if s.get("status") != "available"
            ]

    return result


@router.get("/capabilities")
async def get_capabilities(
    identity: Identity,
    resource_context: str | None = Query(
        None, description="RP slug for resource-scoped capabilities"
    ),
) -> dict[str, Any]:
    """Return available capabilities grouped by category.

    Fast, in-memory lookup — no external calls.  Anonymous users see all
    capabilities but auth-required ones are marked ``locked: true``.

    When ``resource_context`` is provided, returns RP-scoped capabilities
    with suggested questions for that resource's documented sections.
    """
    from ..agent.domains.capabilities import get_capability_registry

    # Check auth to decide locked vs unlocked
    authenticated = identity is not None

    registry = get_capability_registry()

    # RP-scoped response
    if resource_context:
        scoped = await registry.get_by_category_scoped(resource_context, authenticated)
        if scoped is not None:
            return scoped
        # Unknown slug — fall through to standard response

    return {
        "categories": registry.get_by_category(authenticated),
        "is_authenticated": authenticated,
    }


@router.get("/whoami")
async def whoami(identity: Identity) -> JSONResponse:
    """Reflect the caller's own ACCESS identity from their Bearer agent token.

    For debugging and the post-deploy check that a minted token is accepted.
    Clients decide whether the user is signed in from their own mint result,
    not from this endpoint.

    Scope: this reflects ONLY the identity in the caller's own token. There
    is no parameter to look up another user. A caller without a token gets
    ``authenticated: false``, and a bad token gets a 401.

    An identity response must never be cached by any intermediary (a shared
    cache serving one caller's cached body to another would leak identity
    across users) — ``Cache-Control: private, no-store`` plus
    ``Vary: Authorization`` so even a cache that ignores no-store keys on the
    token.
    """
    user = identity.sub if identity else None
    return JSONResponse(
        content={"authenticated": user is not None, "user": user},
        headers={"Cache-Control": "private, no-store", "Vary": "Authorization"},
    )


class RatingRequest(BaseModel):
    """Request model for the rating endpoint."""

    query_id: str = Field(..., description="The question_id from the original query")
    rating: str = Field(..., description="'helpful' or 'not_helpful'")
    feedback: str | None = Field(None, description="Optional free-text feedback")
    session_id: str | None = Field(None, description="Session ID for anonymous ownership binding")


@router.post("/rating")
async def submit_rating(request: RatingRequest, identity: Identity) -> dict[str, Any]:
    """Submit a rating for an agent response.

    Anti-spoofing: the caller must be the turn's signed-in owner or send its
    session_id, or gets the same 404 as an unknown query. One rating per query.
    24h window.
    """
    if request.rating not in ("helpful", "not_helpful"):
        raise HTTPException(status_code=400, detail="rating must be 'helpful' or 'not_helpful'")

    acting_user = identity.sub if identity else None

    usage_logger = get_usage_logger()
    result = await asyncio.to_thread(
        usage_logger.log_rating,
        question_id=request.query_id,
        rating=request.rating,
        feedback=request.feedback,
        acting_user=acting_user,
        session_id=request.session_id,
    )

    if result == "ok":
        # Mirror the rating onto turn_reports (read model). Best-effort;
        # update_rating swallows its own failures.
        from ..turn_reporter import get_turn_reporter

        await asyncio.to_thread(
            get_turn_reporter().update_rating,
            question_id=request.query_id,
            rating=request.rating,
            feedback=request.feedback,
        )
        return {"success": True}
    if result == "not_found":
        raise HTTPException(status_code=404, detail="query_id not found")
    if result == "already_rated":
        raise HTTPException(status_code=409, detail="query already rated")
    if result == "expired":
        raise HTTPException(status_code=410, detail="rating window expired (24h)")
    raise HTTPException(status_code=500, detail="rating failed")


@router.get("/tools")
async def list_tools() -> dict[str, Any]:
    """List available MCP tools."""
    try:
        registry = await get_registry()
        return {
            "tool_count": registry.tool_count,
            "tools": [
                {
                    "name": tool.name,
                    "server": tool.server,
                    "description": tool.description[:100],
                }
                for tool in registry.tools.values()
            ],
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load tools: {e!s}",
        ) from e


@router.get("/catalog")
async def get_catalog(refresh: bool = False) -> dict[str, Any]:
    """Get the aggregated MCP tool catalog.

    Fetches tools from all configured MCP servers and returns
    a unified catalog. Results are cached until refresh=true.

    Args:
        refresh: Force refresh the catalog from MCP servers.

    Returns:
        Aggregated catalog with all tools and metadata.
    """
    try:
        aggregator = get_catalog_aggregator()
        return await aggregator.fetch_catalog(force_refresh=refresh)
    except Exception as e:
        logger.exception(f"Failed to fetch catalog: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch catalog: {e!s}",
        ) from e


@router.post("/catalog/refresh")
async def refresh_catalog() -> dict[str, Any]:
    """Force refresh the MCP tool catalog.

    Fetches fresh tool definitions from all MCP servers.

    Returns:
        Refreshed catalog with metadata.
    """
    try:
        aggregator = get_catalog_aggregator()
        catalog = await aggregator.fetch_catalog(force_refresh=True)
        return {
            "success": True,
            "message": f"Catalog refreshed with {catalog['total_tools']} tools",
            "generated_at": catalog["generated_at"],
            "servers_available": catalog["servers_available"],
            "total_servers": catalog["total_servers"],
            "total_tools": catalog["total_tools"],
        }
    except Exception as e:
        logger.exception(f"Failed to refresh catalog: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh catalog: {e!s}",
        ) from e
