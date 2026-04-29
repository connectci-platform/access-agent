"""Smoke tests for src.main lifespan and OTel auto-instrumentation.

These tests exist because the test suite previously didn't exercise
`src.main`'s lifespan — telemetry init, MCP catalog fetch, and JWKS
configuration only ran at container startup. As a result, the
wrapt 2.x / opentelemetry-instrumentation-langchain incompatibility
(prod crash-loop 2026-04-22 → 2026-04-29) was green in CI but red on
every deploy.

Anything in `lifespan` that can fail synchronously at startup belongs
under this file.
"""

import os

import pytest


def test_init_telemetry_runs_cleanly():
    """`init_telemetry` must complete without raising.

    Directly exercises the `LangchainInstrumentor().instrument()` call
    that broke against wrapt 2.x (the instrumentor passes `module=` to
    `wrap_function_wrapper`, which wrapt 2.x removed).
    """
    # Force console exporter so the test makes no network calls.
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)

    from src.telemetry import init_telemetry, shutdown_telemetry

    init_telemetry(app=None, service_name="test-smoke")
    shutdown_telemetry()


@pytest.mark.asyncio
async def test_app_lifespan_starts_and_stops_cleanly():
    """The full FastAPI lifespan context must enter and exit without raising.

    Catches: telemetry init failures, JWKS configuration crashes, shutdown
    bugs. The MCP catalog fetch is wrapped in try/except inside the lifespan
    itself, so a missing MCP server in the test env is fine.
    """
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)

    from src.main import app, lifespan

    async with lifespan(app):
        pass
