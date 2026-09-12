"""FastAPI routes for the ACCESS Documentation Agent."""

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessageChunk
from pydantic import BaseModel, Field
from starlette.datastructures import Headers
from starlette.responses import StreamingResponse

from ..agent.graph import stream_agent
from ..agent.turn_capture import get_turn_capture, reset_turn_capture
from ..auth import get_acting_user_from_cookie
from ..config import settings
from ..llm import is_empty_answer
from ..tools import ToolRegistry, get_catalog_aggregator
from ..turnstile import get_turnstile_guard, verify_turnstile_token
from ..usage_logger import get_usage_logger

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


# Global registry - built from aggregated catalog
_registry: ToolRegistry | None = None


def _maybe_invalidate_registry(catalog: dict[str, Any]) -> None:
    """Drop the cached ToolRegistry so the next query rebuilds it.

    Called by the catalog-refresh paths: without this, a refresh only
    updated the public /catalog payload while the agent kept serving tools
    (and tool descriptions) from the registry built at startup (#240).

    Guarded: a refresh that caught every MCP server down (a network blip —
    per-server failures don't fail the fetch) must not be adopted into the
    agent, or an unauthenticated refresh call timed during a blip would
    strip the agent's tools until the next refresh or restart. The public
    payload still updates (and /health goes degraded, so the blip is
    visible); the agent keeps its last-good registry.
    """
    if catalog.get("servers_available", 0) <= 0:
        logger.warning(
            "Catalog refresh found no available servers; keeping the agent's current registry"
        )
        return
    global _registry
    _registry = None


async def get_registry() -> ToolRegistry:
    """Get or create the tool registry from aggregated catalog.

    Uses the CatalogAggregator to fetch tools from MCP servers.
    The catalog is fetched at startup and cached.
    """
    global _registry
    if _registry is None:
        # Get catalog from aggregator (fetched at startup, or fetch now if needed)
        aggregator = get_catalog_aggregator()
        catalog = await aggregator.fetch_catalog()

        # Build registry from aggregated catalog
        _registry = ToolRegistry(catalog=catalog)
        logger.info(
            f"Built tool registry with {_registry.tool_count} tools from aggregated catalog"
        )
    return _registry


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="The user's question")
    session_id: str | None = Field(None, description="Session ID for conversation tracking")
    question_id: str | None = Field(None, description="Unique question ID")
    acting_user: str | None = Field(None, description="Transition fallback: acting user from body")
    turnstile_token: str | None = Field(None, description="Cloudflare Turnstile response token")
    resource_context: str | None = Field(
        None, description="RP slug for resource-scoped queries (e.g. 'delta')"
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


def _turnstile_challenge_response() -> JSONResponse:
    """Build the JSON response that tells the frontend to show the Turnstile widget."""
    return JSONResponse(
        content={
            "requires_turnstile": True,
            "site_key": settings.TURNSTILE_SITE_KEY,
        }
    )


async def _check_turnstile(
    acting_user: str | None,
    session_id: str,
    token: str | None,
) -> JSONResponse | None:
    """Check Turnstile for anonymous sessions. Returns a challenge response or None to proceed."""
    if acting_user:
        return None

    guard = get_turnstile_guard()
    if not guard.requires_challenge(session_id):
        return None

    # Session needs verification — check if a token was provided
    if token and await verify_turnstile_token(token):
        guard.mark_verified(session_id)
        return None

    return _turnstile_challenge_response()


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


def _format_sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _should_stream_token(msg: Any, metadata: dict[str, Any]) -> bool:
    """Whether a LangGraph message chunk should reach the browser as a token.

    Extracted from _stream_events so it can be tested without standing up the
    registry, reporter and Turnstile machinery that generator also drives.

    Four conditions, all load-bearing:

    * AIMessageChunk only — the messages stream also carries complete AIMessage
      objects added to state, and streaming those duplicates the whole answer.
    * tool_calling_loop only — other nodes' chatter is not the user's answer.
    * Non-empty — nothing to render.
    * Not the empty-answer sentinel — the LLM client emits that in place of a
      response that stripped to nothing, so the assistant turn stays non-empty
      for vLLM. It is wire-protocol only, and qa-bot-core concatenates every
      token event it receives (qa-flow.tsx), so streaming it would paste the
      marker into the visible answer. tool_calling_loop substitutes real prose
      when it builds final_answer, which the done event carries.
    """
    return (
        isinstance(msg, AIMessageChunk)
        and metadata.get("langgraph_node") == "tool_calling_loop"
        and bool(msg.content)
        and not is_empty_answer(msg.content if isinstance(msg.content, str) else None)
    )


async def _stream_events(  # noqa: PLR0912, PLR0915
    request: QueryRequest,
    acting_user: str | None,
    session_id: str,
    question_id: str,
    include_trace: bool,
    report_context: dict[str, Any] | None = None,
) -> AsyncGenerator[str, None]:
    """Translate LangGraph stream chunks into SSE events.

    Yields SSE-formatted strings for status updates, LLM tokens,
    and a final done event with response metadata.
    """
    start_time = time.time()
    final_state: dict[str, Any] = {}
    reset_turn_capture()

    try:
        registry = await get_registry()

        async for stream_type, chunk in stream_agent(
            query=request.query,
            session_id=session_id,
            question_id=question_id,
            tool_catalog=registry.catalog,
            acting_user=acting_user,
            resource_context=request.resource_context,
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
                # Only stream incremental tokens from the tool_calling_loop node.
                # LangGraph's messages stream emits both AIMessageChunk (tokens)
                # and AIMessage (complete messages added to state). We only want
                # the chunks to avoid duplicating the full response.
                if _should_stream_token(msg, metadata):
                    yield _format_sse_event("token", {"content": msg.content})

            elif stream_type == "updates" and isinstance(chunk, dict):
                # State updates after each node — collect for final metadata
                for node_output in chunk.values():
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

        # Build metadata from final state (mirrors non-streaming QueryResponse fields)
        final_answer = final_state.get("final_answer") or "No answer generated"
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
        yield _format_sse_event("done", done_data)

        # Log usage asynchronously (same as non-streaming path)
        usage_logger = get_usage_logger()
        try:
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

        # Turn report (denormalized read model for the reporting dashboard).
        # Off the response path, swallow failures — never affects the answer.
        from ..agent.domains.capabilities import get_capability_registry as _cap_reg
        from ..services.resource_matcher import resources_for_turn
        from ..turn_reporter import get_turn_reporter

        try:
            reporter = get_turn_reporter()
            prior_turns = await asyncio.to_thread(reporter.count_turns_for_session, session_id)
            # None = count unknown (DB error): write NULL, not a wrong "1".
            turn_index = prior_turns + 1 if prior_turns is not None else None
            resources = await resources_for_turn(request.query, final_answer or "")
            await asyncio.to_thread(
                reporter.log_turn_report,
                final_state=final_state,
                session_id=session_id,
                turn_index=turn_index,
                question_id=question_id,
                query_text=request.query,
                duration_ms=duration_ms,
                acting_user=acting_user,
                success=True,
                capabilities=_cap_reg().infer_capability_ids(final_state.get("tool_results", [])),
                resources=resources,
                turn_capture=get_turn_capture(),
                judge=None,
                **(report_context or {}),
            )
        except Exception:
            logger.exception("Turn report write failed")

        # Track query for Turnstile free-query counting
        if not acting_user:
            get_turnstile_guard().record_query(session_id)

    except Exception as e:
        logger.exception(f"Stream failed: {e}")
        # Write a minimal success=False turn report so the dashboard can tell a
        # failed query apart from one that never happened. Off the response path,
        # swallow failures — never affects the error the user sees. final_state
        # may be partial (the failure can land mid-stream); _assemble_turn_report
        # tolerates missing keys, and the judge is skipped (there's no answer).
        try:
            from ..agent.domains.capabilities import get_capability_registry as _cap_reg
            from ..services.resource_matcher import resources_for_turn
            from ..turn_reporter import get_turn_reporter

            reporter = get_turn_reporter()
            prior_turns = await asyncio.to_thread(reporter.count_turns_for_session, session_id)
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
                duration_ms=(time.time() - start_time) * 1000,
                acting_user=acting_user,
                success=False,
                capabilities=_cap_reg().infer_capability_ids(final_state.get("tool_results", [])),
                resources=resources,
                turn_capture=get_turn_capture(),
                judge=None,
                **(report_context or {}),
            )
        except Exception:
            logger.exception("Failed-turn report write failed")
        yield _format_sse_event(
            "error", {"message": "Failed to process query", "code": "agent_error"}
        )
        # Always yield done so clients can finalize the stream
        yield _format_sse_event("done", {"success": False, "error": str(e)})


@router.post("/query", response_model=None)
async def query_agent(
    request: QueryRequest,
    raw_request: Request,
    include_trace: bool = Query(False, description="Include node_trace in response metadata"),
) -> QueryResponse | JSONResponse | StreamingResponse:
    """Execute a query against the ACCESS Documentation Agent.

    User identity is resolved from the ``SESSaccess_auth`` JWT cookie set by
    ACCESS sites (Drupal, Django, etc.).  During transition, the ``acting_user`` body field is accepted
    as a fallback when ``ALLOW_BODY_ACTING_USER`` is enabled.

    Args:
        request: The query request with question and optional IDs.
        raw_request: The raw FastAPI request (for cookie access).

    Returns:
        For capability discovery: QueryResponse (JSON) with instant answer.
        For agent queries: StreamingResponse (SSE) with status, token, and done events.
    """
    # Resolve acting user from JWT cookie (preferred) or body fallback.
    # Body fallback uses the already-parsed QueryRequest to avoid
    # double-consuming the ASGI body stream.
    acting_user: str | None = None
    user, cookie_present = get_acting_user_from_cookie(raw_request)
    if user:
        acting_user = user
    elif cookie_present:
        # Cookie was present but invalid/expired — do NOT fall through
        # to body fallback (prevents downgrade attacks).
        acting_user = None
    elif settings.ALLOW_BODY_ACTING_USER and request.acting_user:
        # No cookie sent; use body fallback during transition period.
        acting_user = request.acting_user.strip() or None

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
    session_id = request.session_id or f"sess_{timestamp}"
    question_id = request.question_id or f"q_{timestamp}"

    logger.info(
        f"Processing query: {request.query[:50]}... "
        f"(session={session_id}, acting_user={acting_user or 'anonymous'})"
    )

    # Turnstile gate — anonymous users may need to verify they're human.
    # Authenticated users (JWT cookie or body fallback) skip entirely.
    turnstile_response = await _check_turnstile(acting_user, session_id, request.turnstile_token)
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

    # Agent queries stream via SSE
    return StreamingResponse(
        _stream_events(
            request, acting_user, session_id, question_id, include_trace, report_context
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


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
    raw_request: Request,
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
    user, _ = get_acting_user_from_cookie(raw_request)
    authenticated = user is not None

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


class RatingRequest(BaseModel):
    """Request model for the rating endpoint."""

    query_id: str = Field(..., description="The question_id from the original query")
    rating: str = Field(..., description="'helpful' or 'not_helpful'")
    feedback: str | None = Field(None, description="Optional free-text feedback")
    session_id: str | None = Field(None, description="Session ID for anonymous ownership binding")


@router.post("/rating")
async def submit_rating(request: RatingRequest, raw_request: Request) -> dict[str, Any]:
    """Submit a rating for an agent response.

    Anti-spoofing: authenticated users must own the query (user_hash match);
    anonymous users must match session_id. One rating per query. 24h window.
    """
    if request.rating not in ("helpful", "not_helpful"):
        raise HTTPException(status_code=400, detail="rating must be 'helpful' or 'not_helpful'")

    acting_user, _ = get_acting_user_from_cookie(raw_request)

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
    if result == "forbidden":
        raise HTTPException(status_code=403, detail="not authorized to rate this query")
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
        catalog = await aggregator.fetch_catalog(force_refresh=refresh)
        if refresh:
            _maybe_invalidate_registry(catalog)
        return catalog
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
        _maybe_invalidate_registry(catalog)
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
