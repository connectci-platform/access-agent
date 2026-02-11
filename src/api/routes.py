"""FastAPI routes for the ACCESS Documentation Agent."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..agent.graph import run_agent
from ..auth import get_acting_user_from_cookie
from ..config import settings
from ..tools import ToolRegistry, get_catalog_aggregator
from ..usage_logger import get_usage_logger
from ..vault import get_jwt_secret

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


class QueryResponse(BaseModel):
    """Response model for query endpoint."""

    success: bool
    response: str
    session_id: str
    question_id: str
    tools_used: list[str]
    confidence: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.post("/query", response_model=QueryResponse)
async def query_agent(
    request: QueryRequest,
    raw_request: Request,
) -> QueryResponse:
    """Execute a query against the ACCESS Documentation Agent.

    User identity is resolved from the ``access_auth`` JWT cookie set by
    Drupal.  During transition, the ``acting_user`` body field is accepted
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
    try:
        jwt_secret = get_jwt_secret(
            vault_addr=settings.VAULT_ADDR,
            vault_token=settings.VAULT_TOKEN,
            vault_secret_path=settings.VAULT_SECRET_PATH,
            env_fallback=settings.JWT_SECRET,
        )
        user, cookie_present = get_acting_user_from_cookie(
            raw_request,
            jwt_secret=jwt_secret,
        )
        if user:
            acting_user = user
        elif cookie_present:
            # Cookie was present but invalid/expired — do NOT fall through
            # to body fallback (prevents downgrade attacks).
            acting_user = None
        elif settings.ALLOW_BODY_ACTING_USER and request.acting_user:
            # No cookie sent; use body fallback during transition period.
            acting_user = request.acting_user.strip() or None
    except RuntimeError:
        # No JWT secret configured — treat all users as anonymous
        logger.info("JWT secret not configured; treating all requests as anonymous")

    # Generate IDs if not provided
    timestamp = int(time.time() * 1000)
    session_id = request.session_id or f"sess_{timestamp}"
    question_id = request.question_id or f"q_{timestamp}"

    logger.info(
        f"Processing query: {request.query[:50]}... "
        f"(session={session_id}, acting_user={acting_user or 'anonymous'})"
    )

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

        # Log usage for reporting (user ID is hashed, no PII stored)
        usage_logger = get_usage_logger()
        usage_logger.log_query(
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
        )

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
    """Health check endpoint."""
    return {
        "status": "healthy",
        "agent": "access-documentation-langgraph",
        "version": "0.1.0",
        "checkpointing_enabled": USE_CHECKPOINTING,
    }


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
