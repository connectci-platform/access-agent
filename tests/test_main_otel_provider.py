"""The app's lifespan must leave our own tracer provider installed.

FastAPI >= 0.142 auto-configures OpenTelemetry on startup when an OTLP endpoint
is set, and installs its own global providers if nobody beat it to it. Our
provider is installed inside the lifespan, so unless auto-configuration is
turned off the framework wins and our resource (service.name = dataset,
service.component, deployment.environment) never reaches Honeycomb.
"""

import logging

import pytest
from fastapi.testclient import TestClient
from opentelemetry import _logs, metrics, trace
from opentelemetry._logs import _internal as logs_internal
from opentelemetry.metrics import _internal as metrics_internal
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind
from opentelemetry.util._once import Once

from src.config import settings

# (module, once-attr, provider-attr) for each signal. These are private OTel
# attributes with no public reset API; each signal has its own set-once flag.
_GLOBALS = [
    (trace, "_TRACER_PROVIDER_SET_ONCE", "_TRACER_PROVIDER"),
    (metrics_internal, "_METER_PROVIDER_SET_ONCE", "_METER_PROVIDER"),
    (logs_internal, "_LOGGER_PROVIDER_SET_ONCE", "_LOGGER_PROVIDER"),
]


@pytest.fixture(autouse=True)
def _reset_catalog_aggregator():
    """The real lifespan warms the module-level CatalogAggregator singleton
    against unreachable MCP servers; don't leak it into other test files."""
    yield
    import src.tools.registry as _reg

    _reg._aggregator = None


@pytest.fixture
def otel_globals(monkeypatch):
    """Run against pristine OTel globals and FastAPI telemetry bookkeeping.

    Resets before as well as after because test_main_lifespan_smoke leaves a
    shut-down provider installed. FastAPI's runtime registers an atexit
    shutdown over `_owned`, so stale entries would export to a dead port.
    """
    from fastapi.telemetry import _runtime

    saved = [
        (mod, once, prov, getattr(mod, once), getattr(mod, prov)) for mod, once, prov in _GLOBALS
    ]
    saved_owned = list(_runtime._owned)
    saved_configured = list(_runtime._configured)

    def reset():
        for mod, once, prov in _GLOBALS:
            setattr(mod, once, Once())
            setattr(mod, prov, None)
        _runtime._owned.clear()
        _runtime._configured.clear()

    reset()
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_ENABLED", "true")
    # conftest loads a dev .env; OTEL_TRACES_EXPORTER=console would make
    # FastAPI skip auto-configuration and turn the regression test falsely green.
    for name in (
        "OTEL_SDK_DISABLED",
        "OTEL_TRACES_EXPORTER",
        "OTEL_METRICS_EXPORTER",
        "OTEL_LOGS_EXPORTER",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_SERVICE_NAME",
        "HONEYCOMB_DATASET",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(settings, "DATABASE_URL", "", raising=False)
    yield
    for mod, once, prov, once_val, prov_val in saved:
        setattr(mod, once, once_val)
        setattr(mod, prov, prov_val)
    _runtime._owned[:] = saved_owned
    _runtime._configured[:] = saved_configured


def test_lifespan_keeps_our_tracer_provider(otel_globals, monkeypatch, caplog):
    import src.main
    from src.telemetry import setup

    exporter = InMemorySpanExporter()
    monkeypatch.setattr("src.telemetry.setup.OTLPSpanExporter", lambda **kw: exporter)
    caplog.set_level(logging.WARNING)

    with TestClient(src.main.app) as client:
        assert trace.get_tracer_provider() is setup._tracer_provider
        assert setup._tracer_provider.resource.attributes["service.name"] == "access-ci"
        assert "Overriding of current TracerProvider" not in caplog.text
        # LangchainInstrumentor only calls get_meter/get_logger, so these stay
        # unconfigured unless FastAPI's auto-configure installed them.
        assert isinstance(metrics.get_meter_provider(), metrics_internal._ProxyMeterProvider)
        assert isinstance(_logs.get_logger_provider(), logs_internal.ProxyLoggerProvider)

        # The router is mounted under /api/v1; a bare /health would 404.
        assert client.get("/api/v1/health").status_code == 200

        setup._tracer_provider.force_flush()
        server_spans = [
            s
            for s in exporter.get_finished_spans()
            if s.kind == SpanKind.SERVER and s.instrumentation_scope.name == "fastapi"
        ]
        # FastAPI also emits INTERNAL operation spans, so filter before counting.
        assert len(server_spans) == 1


def test_foreign_tracer_provider_is_logged_at_error(otel_globals, caplog):
    from src.telemetry import init_telemetry, shutdown_telemetry

    class ForeignProvider(TracerProvider):
        pass

    trace.set_tracer_provider(ForeignProvider())
    try:
        with caplog.at_level(logging.ERROR):
            init_telemetry(service_name="test")
    finally:
        shutdown_telemetry()

    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("ForeignProvider" in r.getMessage() for r in errors)


def test_unhandled_exception_is_recorded_on_server_span(otel_globals, monkeypatch):
    import src.main
    from src.telemetry import setup

    exporter = InMemorySpanExporter()
    monkeypatch.setattr("src.telemetry.setup.OTLPSpanExporter", lambda **kw: exporter)

    async def boom():
        raise ValueError("kaboom")

    app = src.main.app
    app.add_api_route("/_boom", boom)
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            assert client.get("/_boom").status_code == 500
            setup._tracer_provider.force_flush()
    finally:
        app.router.routes.pop()

    server_span = next(
        s
        for s in exporter.get_finished_spans()
        if s.kind == SpanKind.SERVER and s.instrumentation_scope.name == "fastapi"
    )
    events = [e for e in server_span.events if e.name == "exception"]
    assert len(events) == 1
    assert events[0].attributes["exception.type"] == "ValueError"
    # The message and traceback can carry request content, so the export
    # filter keeps only the type; the full traceback is in the server log.
    assert "exception.message" not in events[0].attributes
    assert "exception.stacktrace" not in events[0].attributes
