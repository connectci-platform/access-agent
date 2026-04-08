"""FastAPI application entry point for ACCESS Documentation Agent."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import router
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
    init_telemetry(app=app, service_name="access-agent")

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

    # Configure trusted JWKS issuers for JWT cookie authentication.
    from .auth import configure_trusted_issuers

    trusted_issuers: dict[str, str] = {}
    if settings.TRUSTED_JWKS_URLS:
        for raw_entry in settings.TRUSTED_JWKS_URLS.split(","):
            entry = raw_entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                logger.warning(
                    "Ignoring malformed TRUSTED_JWKS_URLS entry (missing '='): %s", entry
                )
                continue
            issuer, jwks_url = entry.split("=", 1)
            jwks_url = jwks_url.strip()
            if (
                settings.ENVIRONMENT == "production"
                and jwks_url
                and not jwks_url.startswith("https://")
            ):
                logger.warning(
                    "JWKS URL for issuer %s is not HTTPS — this is unsafe in production: %s",
                    issuer.strip(),
                    jwks_url,
                )
            trusted_issuers[issuer.strip()] = jwks_url
    configure_trusted_issuers(trusted_issuers, environment=settings.ENVIRONMENT)

    logger.info(f"QA Service URL: {settings.QA_SERVICE_URL}")
    if settings.UKY_RAG_ENABLED:
        logger.info(f"UKY RAG enabled: general={settings.UKY_RAG_GENERAL_URL}")
        logger.info(f"UKY RAG enabled: xdmod={settings.UKY_RAG_XDMOD_URL}")
    else:
        logger.info("UKY RAG disabled (UKY_RAG_ENABLED=false)")

    yield

    # --- Shutdown ---
    from .services.uky_client import get_uky_client
    from .tools.mcp_client import close_shared_client

    logger.info("Shutting down ACCESS Documentation Agent")
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

# Add CORS middleware
# In production, restrict to specific origins; in dev, allow all
if settings.ENVIRONMENT == "production":
    if settings.ALLOWED_ORIGINS:
        allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
    else:
        allowed_origins = [
            "https://support.access-ci.org",
            "https://allocations.access-ci.org",
            "https://operations.access-ci.org",
            "https://metrics.access-ci.org",
            "https://access-ci.org",
        ]
elif settings.ALLOWED_ORIGINS:
    allowed_origins = [o.strip() for o in settings.ALLOWED_ORIGINS.split(",")]
else:
    allowed_origins = [
        "https://accessmatch.ddev.site",
        "http://accessmatch.ddev.site",
        "http://localhost:5173",
        "http://localhost:3000",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1")


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
