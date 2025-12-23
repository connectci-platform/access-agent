"""FastAPI routes for the ACCESS Documentation Agent."""

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..agent.graph import run_agent
from ..config import settings
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

router = APIRouter()

# Check if checkpointing is enabled
USE_CHECKPOINTING = bool(settings.DATABASE_URL)

# Global registry - loaded on first request
_registry: ToolRegistry | None = None


async def get_registry() -> ToolRegistry:
    """Get or create the tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        await _registry.load()
        logger.info(f"Loaded tool registry with {_registry.tool_count} tools")
    return _registry


class QueryRequest(BaseModel):
    """Request model for query endpoint."""

    query: str = Field(..., description="The user's question")
    session_id: str | None = Field(None, description="Session ID for conversation tracking")
    question_id: str | None = Field(None, description="Unique question ID")


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
async def query_agent(request: QueryRequest) -> QueryResponse:
    """Execute a query against the ACCESS Documentation Agent.

    Args:
        request: The query request with question and optional IDs.

    Returns:
        QueryResponse with answer and metadata.

    Raises:
        HTTPException: If query execution fails.
    """
    # Generate IDs if not provided
    timestamp = int(time.time() * 1000)
    session_id = request.session_id or f"sess_{timestamp}"
    question_id = request.question_id or f"q_{timestamp}"

    logger.info(f"Processing query: {request.query[:50]}... (session={session_id})")

    try:
        # Get tool catalog
        registry = await get_registry()

        # Run the agent (with checkpointing if DATABASE_URL is set)
        final_state = await run_agent(
            query=request.query,
            session_id=session_id,
            question_id=question_id,
            tool_catalog=registry.catalog,
            use_checkpointing=USE_CHECKPOINTING,
            db_uri=settings.DATABASE_URL if USE_CHECKPOINTING else None,
        )

        # Extract results
        final_answer = final_state.get("final_answer") or "No answer generated"
        tools_used = final_state.get("tools_used", [])
        query_analysis = final_state.get("query_analysis")

        confidence = None
        if query_analysis:
            confidence = query_analysis.confidence

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
