"""FastAPI application entry point for ACCESS Documentation Agent."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .agent.graph import create_pooled_checkpointer
from .agent_token import AgentTokenVerifier, check_agent_token_config
from .api import router
from .api.thread_routes import thread_router
from .config import settings
from .telemetry import init_telemetry, shutdown_telemetry

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown lifecycle for the FastAPI app."""
    # --- Startup ---
    # First, so a production deploy without the agent-token settings fails
    # here instead of answering every signed-in request with a 503.
    check_agent_token_config(settings)
    init_telemetry(app=app, service_name="access-agent")

    logger.info("Starting ACCESS Documentation Agent")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"API: http://{settings.API_HOST}:{settings.API_PORT}")

    app.state.agent_token_verifier = AgentTokenVerifier.from_settings(settings)

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

    if settings.UKY_RAG_ENABLED:
        logger.info(f"UKY RAG enabled: general={settings.UKY_RAG_GENERAL_URL}")
        logger.info(f"UKY RAG enabled: xdmod={settings.UKY_RAG_XDMOD_URL}")
    else:
        logger.info("UKY RAG disabled (UKY_RAG_ENABLED=false)")

    # Lifespan-scoped checkpointer: built and setup() once here, held open for
    # the app's lifetime, and injected into request handlers via app.state.
    # Replaces the old per-request `async with create_async_checkpointer(...)`
    # path, which opened a fresh connection and ran setup() on every request.
    app.state.checkpointer = None
    if settings.DATABASE_URL:
        cm = create_pooled_checkpointer(settings.DATABASE_URL)
        checkpointer = await cm.__aenter__()
        await checkpointer.setup()
        app.state.checkpointer = checkpointer
        app.state._checkpointer_cm = cm  # noqa: SLF001  # own state attr, not a third-party private
        logger.info("Pooled checkpointer initialized")

    yield

    # --- Shutdown ---
    from .services.uky_client import get_uky_client
    from .tools.mcp_client import close_shared_client

    logger.info("Shutting down ACCESS Documentation Agent")
    if getattr(app.state, "_checkpointer_cm", None) is not None:
        await app.state._checkpointer_cm.__aexit__(None, None, None)  # noqa: SLF001
    await app.state.agent_token_verifier.aclose()
    await close_shared_client()
    await get_uky_client().close()
    shutdown_telemetry()


# Create FastAPI app
app = FastAPI(
    title="ACCESS Documentation Agent",
    description="LangGraph-powered documentation agent for ACCESS-CI",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware.
# Production: allow all access-ci.org subdomains and any accessmatch Pantheon
# environment (dev, test, live, md-* multidevs). accessmatch is the only
# Drupal-multisite tenant that loads the chatbot today; other tenants are
# deliberately excluded. Setting ALLOWED_ORIGINS to a non-empty value
# overrides both the explicit list and the regex with the env-provided list.
allowed_origin_regex: str | None = None
if settings.ENVIRONMENT == "production":
    if settings.ALLOWED_ORIGINS:
        allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
    else:
        allowed_origins = []
        allowed_origin_regex = (
            r"https://"
            r"(?:"
            r"(?:[a-z0-9-]+\.)?access-ci\.org"
            r"|"
            r"[a-z0-9-]+-accessmatch\.pantheonsite\.io"
            r")"
        )
elif settings.ALLOWED_ORIGINS:
    allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
else:
    allowed_origins = [
        "https://accessmatch.ddev.site",
        "http://accessmatch.ddev.site",
        "https://chat.accessmatch.ddev.site",
        "http://localhost:5173",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    # Identity is the Bearer token, so no caller needs its cookies sent here.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")
app.include_router(thread_router, prefix="/api/v1")


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
