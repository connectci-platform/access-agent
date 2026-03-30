"""FastAPI routes for the ACCESS Documentation Agent."""

import asyncio
import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..agent.graph import run_agent
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


@router.post("/query", response_model=None)
async def query_agent(
    request: QueryRequest,
    raw_request: Request,
    include_trace: bool = Query(False, description="Include node_trace in response metadata"),
) -> QueryResponse | JSONResponse:
    """Execute a query against the ACCESS Documentation Agent.

    User identity is resolved from the ``SESSaccess_auth`` JWT cookie set by
    ACCESS sites (Drupal, Django, etc.).  During transition, the ``acting_user`` body field is accepted
    as a fallback when ``ALLOW_BODY_ACTING_USER`` is enabled.

    Args:
        request: The query request with question and optional IDs.
        raw_request: The raw FastAPI request (for cookie access).

    Returns:
        QueryResponse with answer and metadata.

    Raises:
        HTTPException: If query execution fails.
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

    # Generate IDs if not provided
    timestamp = int(time.time() * 1000)
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

    start_time = time.time()

    try:
        # Get tool catalog
        registry = await get_registry()

        # Run the agent (with checkpointing if DATABASE_URL is set)
        final_state = await run_agent(
            query=request.query,
            session_id=session_id,
            question_id=question_id,
            tool_catalog=registry.catalog,
            acting_user=acting_user,
            use_checkpointing=USE_CHECKPOINTING,
            db_uri=settings.DATABASE_URL if USE_CHECKPOINTING else None,
        )

        # Extract results
        final_answer = final_state.get("final_answer") or "No answer generated"
        tools_used = final_state.get("tools_used", [])
        query_analysis = final_state.get("query_analysis")

        confidence = None
        query_type = None
        if query_analysis:
            confidence = query_analysis.confidence
        query_classification = final_state.get("query_classification")
        if query_classification:
            query_type = query_classification.query_type

        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000

        # Resolve capability_id — use classifier output, or infer from tools
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

        # Log usage for reporting (user ID is hashed, no PII stored).
        # Run in a thread to avoid blocking the async event loop with
        # synchronous SQLAlchemy calls.
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

        # Build classification summary for response
        classification_info = None
        if query_classification:
            classification_info = {
                "query_type": query_classification.query_type,
                "confidence": query_classification.confidence,
                "domain": query_classification.domain,
                "reason": query_classification.reason[:200] if query_classification.reason else "",
            }

        return QueryResponse(
            success=True,
            response=final_answer,
            session_id=session_id,
            question_id=question_id,
            tools_used=tools_used,
            confidence=confidence,
            metadata={
                "agent": "access-documentation-langgraph",
                "tool_count": len(tools_used),
                "execution_strategy": final_state.get("execution_strategy", "unknown"),
                "checkpointing_enabled": USE_CHECKPOINTING,
                "duration_ms": duration_ms,
                "classification": classification_info,
                **({"node_trace": final_state.get("node_trace", [])} if include_trace else {}),
            },
        )

    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query execution failed: {e!s}",
        ) from e


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
                s["name"] for s in catalog.get("servers", []) if s.get("status") != "available"
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


@router.post("/rating")
async def submit_rating(request: RatingRequest) -> dict[str, Any]:
    """Submit a rating for an agent response.

    Attaches the rating to the existing usage log entry identified by
    query_id. Anonymous ratings are accepted (support capabilities are
    available to anonymous users) but the query_id must exist.
    """
    if request.rating not in ("helpful", "not_helpful"):
        raise HTTPException(status_code=400, detail="rating must be 'helpful' or 'not_helpful'")

    usage_logger = get_usage_logger()
    found = await asyncio.to_thread(
        usage_logger.log_rating,
        question_id=request.query_id,
        rating=request.rating,
        feedback=request.feedback,
    )

    if not found:
        raise HTTPException(status_code=404, detail="query_id not found")

    return {"success": True}


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
