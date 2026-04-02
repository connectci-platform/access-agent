"""OpenTelemetry setup and configuration.

Initializes the OpenTelemetry SDK with OTLP export to Honeycomb.
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

# Load .env file if it exists (needed for OTLP credentials)
_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(_env_file)

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

try:
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
except ImportError:
    LangchainInstrumentor = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)

_tracer_provider: TracerProvider | None = None


def init_telemetry(
    app: "FastAPI | None" = None,
    service_name: str = "access-agent",
) -> None:
    """Initialize OpenTelemetry with OTLP export to Honeycomb.

    Configures tracing to export to Honeycomb (or console for local dev).
    Auto-instruments FastAPI and HTTPX if app is provided.

    Args:
        app: FastAPI application to instrument (optional).
        service_name: Name of this service in traces.

    Environment Variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint URL (e.g., https://api.honeycomb.io)
        OTEL_EXPORTER_OTLP_HEADERS: Headers for auth (e.g., "x-honeycomb-team=xxx")
        OTEL_SERVICE_NAME: Override service name (optional)
        OTEL_ENABLED: Set to "false" to disable telemetry
        HONEYCOMB_DATASET: Dataset name (default: "access-ci")
    """
    global _tracer_provider

    # Check if telemetry is disabled
    if os.getenv("OTEL_ENABLED", "true").lower() == "false":
        logger.info("OpenTelemetry disabled via OTEL_ENABLED=false")
        return

    # Get configuration from environment
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    otlp_headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
    service_name = os.getenv("OTEL_SERVICE_NAME", service_name)

    # Create resource with service info
    # Use HONEYCOMB_DATASET as service.name (determines dataset in Honeycomb)
    # Use original service_name as service.component for filtering
    dataset = os.getenv("HONEYCOMB_DATASET", "access-ci")
    resource = Resource.create(
        {
            "service.name": dataset,
            "service.component": service_name,
            "service.version": "0.1.0",
            "deployment.environment": os.getenv("ENVIRONMENT", "local"),
        }
    )

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Add span processor based on config
    if otlp_endpoint:
        # Production: export to OTLP endpoint (Honeycomb)
        headers = _parse_headers(otlp_headers) if otlp_headers else {}

        exporter = OTLPSpanExporter(
            endpoint=f"{otlp_endpoint}/v1/traces",
            headers=headers,
        )
        _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        logger.info(f"OpenTelemetry initialized: exporting to {otlp_endpoint} (dataset: {dataset})")
    else:
        # Local dev: export to console
        _tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        logger.info("OpenTelemetry initialized: exporting to console (no OTLP endpoint configured)")

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Auto-instrument FastAPI
    if app is not None:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI auto-instrumentation enabled")

    # Note: HTTPX instrumentation is handled via explicit AsyncOpenTelemetryTransport
    # in telemetry/http.py, which is more reliable than HTTPXClientInstrumentor().instrument()

    # Auto-instrument LangChain (for LLM calls with token usage)
    if LangchainInstrumentor is not None:
        LangchainInstrumentor().instrument()
        logger.info("LangChain auto-instrumentation enabled")
    else:
        logger.warning("LangChain instrumentation unavailable — skipping")


def shutdown_telemetry() -> None:
    """Shutdown telemetry and flush pending spans."""
    global _tracer_provider

    if _tracer_provider is not None:
        _tracer_provider.shutdown()
        logger.info("OpenTelemetry shutdown complete")
        _tracer_provider = None


def get_tracer(name: str = "access-agent") -> trace.Tracer:
    """Get a tracer instance for creating spans.

    Args:
        name: Name of the tracer (typically module name).

    Returns:
        Tracer instance.
    """
    return trace.get_tracer(name)


def _parse_headers(headers_str: str) -> dict[str, str]:
    """Parse OTLP headers from comma-separated key=value string.

    Format: "key1=value1,key2=value2"

    Args:
        headers_str: Header string from environment.

    Returns:
        Dictionary of headers.
    """
    headers = {}
    for pair in headers_str.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            headers[key.strip()] = value.strip()
    return headers
