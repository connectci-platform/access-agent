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

    init_telemetry(service_name="test-smoke")
    shutdown_telemetry()


@pytest.mark.asyncio
async def test_app_lifespan_starts_and_stops_cleanly():
    """The full FastAPI lifespan context must enter and exit without raising.

    Catches: telemetry init failures, agent-token configuration crashes,
    shutdown bugs. The MCP catalog fetch is wrapped in try/except inside the
    lifespan itself, so a missing MCP server in the test env is fine.

    Request tests install their own verifier because ASGITransport skips the
    lifespan, so this is what checks that the app creates one.
    """
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)

    from src.agent_token import AgentTokenVerifier
    from src.main import app, lifespan

    async with lifespan(app):
        assert isinstance(app.state.agent_token_verifier, AgentTokenVerifier)


@pytest.mark.asyncio
async def test_lifespan_warms_the_resource_groups_and_stops_the_refresh(monkeypatch):
    """The group vocabulary loads in the background at startup, so the first
    signed-in request doesn't find it cold; shutdown cancels a fetch still
    running."""
    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)

    import src.main as main_mod

    calls: list[str] = []

    class _Cache:
        def refresh_in_background(self):
            calls.append("start")

        async def stop_background_refresh(self):
            calls.append("stop")

    monkeypatch.setattr(main_mod, "get_rp_cache", _Cache)

    async with main_mod.lifespan(main_mod.app):
        assert calls == ["start"]
    assert calls == ["start", "stop"]


@pytest.mark.asyncio
async def test_shutdown_waits_for_reports_of_stopped_turns():
    """A turn report written after a cancel finishes before the app stops (D8-3047)."""
    import asyncio

    os.environ.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    from src.api.thread_routes import _detached_reports
    from src.main import app, lifespan

    done: list[str] = []

    async def _report():
        await asyncio.sleep(0.01)
        done.append("reported")

    async with lifespan(app):
        task = asyncio.ensure_future(_report())
        _detached_reports.add(task)
        task.add_done_callback(_detached_reports.discard)
    assert done == ["reported"]
