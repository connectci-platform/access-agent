"""FastAPI application entry point for ACCESS Documentation Agent."""

import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ACCESS Documentation Agent",
    description="LangGraph-powered documentation agent for ACCESS-CI",
    version="0.1.0",
)

# Add CORS middleware
# In production, restrict to specific origins; in dev, allow all
if settings.ENVIRONMENT == "production" and settings.ALLOWED_ORIGINS:
    allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
else:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the agent on startup."""
    logger.info("Starting ACCESS Documentation Agent")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"API: http://{settings.API_HOST}:{settings.API_PORT}")

    # Fetch tool catalog from MCP servers
    from .tools import get_catalog_aggregator

    logger.info("Fetching tool catalog from MCP servers...")
    try:
        aggregator = get_catalog_aggregator()
        catalog = await aggregator.fetch_catalog()
        logger.info(
            f"Catalog loaded: {catalog['total_tools']} tools from "
            f"{catalog['servers_available']}/{catalog['total_servers']} servers"
        )
    except Exception as e:
        logger.warning(f"Failed to fetch catalog at startup: {e}")
        logger.warning("Will retry on first request")

    # Log QA Service configuration (RAG retrieval happens via HTTP on-demand)
    logger.info(f"QA Service URL: {settings.QA_SERVICE_URL}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    from .tools.mcp_client import close_shared_client

    logger.info("Shutting down ACCESS Documentation Agent")
    await close_shared_client()


def main() -> None:
    """Run the application."""
    uvicorn.run(
        "src.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    main()
