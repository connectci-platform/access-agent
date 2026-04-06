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
from starlette.responses import StreamingResponse

from ..agent.graph import stream_agent
from ..auth import get_acting_user_from_cookie
from ..config import settings
from ..tools import ToolRegistry, get_catalog_aggregator
from ..turnstile import get_turnstile_guard, verify_turnstile_token
from ..usage_logger import get_usage_logger

logger = logging.getLogger(__name__)

router = APIRouter()

# Check if checkpointing is enabled
USE_CHECKPOINTING = bool(settings.DATABASE_URL)

# Global registry - built from aggregated catalog
_registry: ToolRegistry | None = None


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


def _check_capability_discovery(
    query: str,
    authenticated: bool,
    session_id: str,
    question_id: str,
) -> QueryResponse | None:
    """Return a direct response for capability discovery queries.

    The frontend sends category labels ("Get help", "Explore resources") and
    "Show my options" as messages.  These are answered from the capability
    registry — no LLM or RAG call needed.
    """
    from ..agent.domains.capabilities import get_capability_registry

    registry = get_capability_registry()
    # Strip lock emoji prefix that the frontend adds to auth-required buttons
    normalized = query.strip().removeprefix("🔒").strip().lower()

    # Build lookup: category label → category object
    categories = registry.get_by_category(authenticated)
    cat_by_label: dict[str, dict[str, Any]] = {}
    for cat in categories:
        cat_by_label[cat["label"].lower()] = cat

    # "Show my options" → list all categories
    if normalized in ("show my options", "what can you do", "what can you help with"):
        lines = ["Here's what I can help you with:\n"]
        for cat in categories:
            lines.append(f"**{cat['label']}**")
            for cap in cat["capabilities"]:
                locked = " 🔒 (login required)" if cap.get("locked") else ""
                lines.append(f"- {cap['label']}: {cap['description']}{locked}")
            lines.append("")
        if not authenticated:
            lines.append("*Some features require logging in. Log in to unlock all capabilities.*")
        lines.append(
            'You can also just type a question — try including keywords like '
            '"system status", "events", "software", or "allocations".'
        )
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

    # Category label match → list capabilities in that category
    if normalized in cat_by_label:
        cat = cat_by_label[normalized]
        caps = cat["capabilities"]

        if len(caps) == 1:
            cap = caps[0]
            if cap.get("locked"):
                answer = (
                    f"**{cap['label']}** requires logging in. "
                    f"{cap['description']}. Please log in to use this feature."
                )
            else:
                answer = (
                    f"I can help you with that! {cap['description']}. What would you like to know?"
                )
        else:
            lines = [f"Here's what I can help with for **{cat['label']}**:\n"]
            for cap in caps:
                locked = " 🔒 (login required)" if cap.get("locked") else ""
                lines.append(f"- **{cap['label']}**: {cap['description']}{locked}")

            # Per-category example prompts
            examples = {
                "explore": (
                    '\nTry asking something like *"Are there any outages right now?"* '
                    'or *"What software is on Delta?"*'
                ),
                "support": (
                    '\nTry something like *"I need help logging in"* '
                    'or *"I want to report a security issue"*'
                ),
            }
            lines.append(examples.get(cat["id"], "\nJust type your question!"))
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

    # Capability label match → direct prompt for that capability
    cap_by_label: dict[str, dict[str, Any]] = {}
    for cat in categories:
        for cap in cat["capabilities"]:
            cap_by_label[cap["label"].lower()] = cap

    # Example queries for each capability — shown in the shortcircuit response
    # to teach users how to phrase their questions.
    SAMPLE_QUERIES: dict[str, str] = {
        "check_allocations": "check allocations for project TG-CIS123456",
        "search_software": "is Python available on Anvil?",
        "check_system_status": "is Delta down right now?",
        "browse_events": "upcoming workshops this month",
        "browse_affinity_groups": "affinity groups related to machine learning",
        "check_usage": "show my usage on Delta last month",
        "search_nsf_awards": "NSF awards for computational biology",
        "search_announcements": "recent announcements about Expanse",
        "open_ticket": "I need help with my allocation request",
        "report_login_problem": "I can't log in to Anvil",
        "report_security": "I found a vulnerability on the support portal",
        "manage_announcements": "show my draft announcements",
    }

    if normalized in cap_by_label:
        cap = cap_by_label[normalized]
        if cap.get("locked"):
            answer = (
                f"**{cap['label']}** requires logging in. "
                f"{cap['description']}. Please log in to use this feature."
            )
        else:
            sample = SAMPLE_QUERIES.get(cap["id"])
            hint = f'\n\nTry typing something like *"{sample}"*' if sample else ""
            answer = (
                f"I can help with that! {cap['description']}.{hint}"
            )
        return QueryResponse(
            success=True,
            response=answer,
            session_id=session_id,
            question_id=question_id,
            tools_used=[],
            confidence="high",
            metadata={
                "agent": "capability-discovery",
                "capability_id": cap["id"],
                "is_final_response": True,
                "rating_target": None,
                "question_id": question_id,
            },
        )

    return None


def _format_sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_events(  # noqa: PLR0912, PLR0915
    request: QueryRequest,
    acting_user: str | None,
    session_id: str,
    question_id: str,
    include_trace: bool,
) -> AsyncGenerator[str, None]:
    """Translate LangGraph stream chunks into SSE events.

    Yields SSE-formatted strings for status updates, LLM tokens,
    and a final done event with response metadata.
    """
    start_time = time.time()
    final_state: dict[str, Any] = {}

    try:
        registry = await get_registry()

        async for stream_type, chunk in stream_agent(
            query=request.query,
            session_id=session_id,
            question_id=question_id,
            tool_catalog=registry.catalog,
            acting_user=acting_user,
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
                # Only stream incremental tokens from the synthesize node.
                # LangGraph's messages stream emits both AIMessageChunk (tokens)
                # and AIMessage (complete messages added to state). We only want
                # the chunks to avoid duplicating the full response.
                if (
                    isinstance(msg, AIMessageChunk)
                    and metadata.get("langgraph_node") == "synthesize"
                    and msg.content
                ):
                    yield _format_sse_event("token", {"content": msg.content})

            elif stream_type == "updates" and isinstance(chunk, dict):
                # State updates after each node — collect for final metadata
                for node_output in chunk.values():
                    if isinstance(node_output, dict):
                        final_state.update(node_output)

        # Build metadata from final state (mirrors non-streaming QueryResponse fields)
        final_answer = final_state.get("final_answer") or "No answer generated"
        tools_used = final_state.get("tools_used", [])
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
        elif "uky_rag_retrieval" in tools_used:
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
                "tool_count": len(tools_used),
                "tools_used": tools_used,
                "execution_strategy": final_state.get("execution_strategy", "parallel"),
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
                duration_ms=duration_ms,
                response_length=len(final_answer),
                acting_user=acting_user,
                success=True,
                capability_id=capability_id,
                category=cap_category,
            )
        except Exception:
            logger.exception("Usage logging failed")

        # Track query for Turnstile free-query counting
        if not acting_user:
            get_turnstile_guard().record_query(session_id)

    except Exception as e:
        logger.exception(f"Stream failed: {e}")
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
    discovery_response = _check_capability_discovery(
        request.query,
        acting_user is not None,
        session_id,
        question_id,
    )
    if discovery_response is not None:
        return discovery_response

    # Agent queries stream via SSE
    return StreamingResponse(
        _stream_events(request, acting_user, session_id, question_id, include_trace),
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
async def get_capabilities(raw_request: Request) -> dict[str, Any]:
    """Return available capabilities grouped by category.

    Fast, in-memory lookup — no external calls.  Anonymous users see all
    capabilities but auth-required ones are marked ``locked: true``.
    """
    from ..agent.domains.capabilities import get_capability_registry

    # Check auth to decide locked vs unlocked
    user, _ = get_acting_user_from_cookie(raw_request)
    authenticated = user is not None

    registry = get_capability_registry()
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
