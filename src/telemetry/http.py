"""Instrumented HTTP client factory.

Provides HTTPX clients with OpenTelemetry trace context propagation.
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)


def is_telemetry_enabled() -> bool:
    """Check if OpenTelemetry is enabled."""
    return os.getenv("OTEL_ENABLED", "true").lower() != "false"


def create_async_client(
    timeout: float = 30.0,
    connect_timeout: float = 5.0,
    max_connections: int = 100,
    max_keepalive_connections: int = 20,
) -> httpx.AsyncClient:
    """Create an HTTPX AsyncClient with OpenTelemetry instrumentation.

    When telemetry is enabled, the client uses AsyncOpenTelemetryTransport
    which automatically injects trace context headers (traceparent, tracestate)
    into outgoing requests. This enables distributed tracing across services.

    Args:
        timeout: Total request timeout in seconds.
        connect_timeout: Connection timeout in seconds.
        max_connections: Maximum number of connections in the pool.
        max_keepalive_connections: Maximum keepalive connections.

    Returns:
        An AsyncClient configured with connection pooling and optional
        OpenTelemetry instrumentation.
    """
    timeout_config = httpx.Timeout(timeout, connect=connect_timeout)
    limits = httpx.Limits(
        max_connections=max_connections,
        max_keepalive_connections=max_keepalive_connections,
    )

    if is_telemetry_enabled():
        try:
            from opentelemetry.instrumentation.httpx import AsyncOpenTelemetryTransport

            transport = AsyncOpenTelemetryTransport(httpx.AsyncHTTPTransport())
            logger.debug("Created instrumented AsyncClient with trace context propagation")
            return httpx.AsyncClient(
                transport=transport,
                timeout=timeout_config,
                limits=limits,
            )
        except ImportError:
            logger.warning(
                "OpenTelemetry HTTPX instrumentation not available, "
                "trace context will not be propagated"
            )

    logger.debug("Created AsyncClient without instrumentation")
    return httpx.AsyncClient(timeout=timeout_config, limits=limits)
