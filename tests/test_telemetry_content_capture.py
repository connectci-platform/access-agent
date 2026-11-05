"""LangChain spans exported to Honeycomb must not carry conversation text.

openllmetry's LangChain instrumentation records prompts and responses unless
TRACELOOP_TRACE_CONTENT is "false", and it reads the variable on every span.
The conversation is stored on-prem in turn_reports, so Honeycomb gets
operational telemetry only.
"""

import os

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from opentelemetry.instrumentation.langchain.callback_handler import TraceloopCallbackHandler
from opentelemetry.metrics import NoOpMeter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.agent.context import AgentContext
from src.config import settings
from src.telemetry.content_filter import (
    CONTENT_ATTRIBUTES,
    ContentFilteringSpanExporter,
    is_content_attribute,
)
from tests.test_main_otel_provider import otel_globals  # noqa: F401

QUESTION = "what is my allocation balance on anvil"
ANSWER = "you have 1234 SUs left"


def _exported_text() -> str:
    """Run one chat-model call through openllmetry's span handler and return
    every exported attribute value, joined.

    The handler is built directly rather than through LangchainInstrumentor:
    the instrumentor is a process-wide singleton whose handler stays bound to
    whichever provider was current when another test first instrumented it.
    """
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    meter = NoOpMeter("test")
    handler = TraceloopCallbackHandler(
        provider.get_tracer("test"),
        meter.create_histogram("duration"),
        meter.create_histogram("tokens"),
    )
    model = GenericFakeChatModel(messages=iter([AIMessage(content=ANSWER)]))
    model.invoke(QUESTION, config={"callbacks": [handler]})
    spans = exporter.get_finished_spans()
    assert spans, "the LangChain call produced no spans"
    return " ".join(str(v) for s in spans for v in (s.attributes or {}).values())


@pytest.mark.parametrize("capture", [False, True])
def test_init_telemetry_sets_content_capture_from_settings(
    otel_globals,  # noqa: F811
    monkeypatch,
    capture,
):
    from src.telemetry import init_telemetry, shutdown_telemetry

    monkeypatch.setattr("src.telemetry.setup.OTLPSpanExporter", lambda **kw: InMemorySpanExporter())
    monkeypatch.setattr(settings, "OTEL_CAPTURE_CONTENT", capture)
    # A stray .env value must not win over the setting.
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false" if capture else "true")
    try:
        init_telemetry(service_name="test")
    finally:
        shutdown_telemetry()
    assert os.environ["TRACELOOP_TRACE_CONTENT"] == ("true" if capture else "false")


def test_openllmetry_honors_content_flag_off(monkeypatch):
    # Library tripwire: our own wiring of the flag is covered by
    # test_init_telemetry_sets_content_capture_from_settings. This catches an
    # openllmetry upgrade that stops honoring TRACELOOP_TRACE_CONTENT.
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    exported = _exported_text()
    assert QUESTION not in exported
    assert ANSWER not in exported


def test_openllmetry_honors_content_flag_on(monkeypatch):
    # Control: proves the test above would see the text if it were exported.
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "true")
    exported = _exported_text()
    assert QUESTION in exported
    assert ANSWER in exported


def _span_with(attributes, event_attributes=None):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer("test").start_as_current_span("s", attributes=attributes) as span:
        if event_attributes:
            span.add_event("e", attributes=event_attributes)
    return exporter.get_finished_spans()[0]


def test_exporter_drops_content_and_keeps_metadata():
    span = _span_with(
        {
            "gen_ai.system_instructions": "profile: anvil allocation",
            "gen_ai.prompt.0.content": QUESTION,
            "gen_ai.tool.call.result": ANSWER,
            "gen_ai.usage.input_tokens": 12,
            "gen_ai.tool.name": "search_projects",
        },
        event_attributes={"gen_ai.output.messages": ANSWER, "kept": "yes"},
    )
    inner = InMemorySpanExporter()
    ContentFilteringSpanExporter(inner).export([span])
    (out,) = inner.get_finished_spans()

    assert dict(out.attributes) == {
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.tool.name": "search_projects",
    }
    assert dict(out.events[0].attributes) == {"kept": "yes"}
    assert (out.context, out.parent, out.name, out.start_time, out.end_time) == (
        span.context,
        span.parent,
        span.name,
        span.start_time,
        span.end_time,
    )


def test_exporter_passes_content_free_spans_through_unchanged():
    span = _span_with({"gen_ai.usage.input_tokens": 12})
    inner = InMemorySpanExporter()
    ContentFilteringSpanExporter(inner).export([span])
    assert inner.get_finished_spans()[0] is span


@pytest.mark.parametrize("capture", [False, True])
def test_init_telemetry_filters_content_unless_capture_is_on(
    otel_globals,  # noqa: F811
    monkeypatch,
    capture,
):
    from opentelemetry import trace

    from src.telemetry import init_telemetry, setup, shutdown_telemetry

    exporter = InMemorySpanExporter()
    monkeypatch.setattr("src.telemetry.setup.OTLPSpanExporter", lambda **kw: exporter)
    monkeypatch.setattr(settings, "OTEL_CAPTURE_CONTENT", capture)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    try:
        init_telemetry(service_name="test")
        with trace.get_tracer("test").start_as_current_span(
            "s", attributes={"gen_ai.system_instructions": "profile: anvil allocation"}
        ):
            pass
        setup._tracer_provider.force_flush()
        (span,) = exporter.get_finished_spans()
    finally:
        shutdown_telemetry()
    assert ("gen_ai.system_instructions" in span.attributes) is capture


def _root_span_attributes(monkeypatch, stream: bool) -> dict:
    import asyncio

    import src.agent.graph as graph_mod

    class _Graph:
        async def ainvoke(self, state, config, context=None):
            return {}

        async def astream(self, state, config=None, stream_mode=None, context=None):
            return
            yield  # makes this an (empty) async generator

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(graph_mod, "create_agent_graph", _Graph)
    monkeypatch.setattr(graph_mod, "get_tracer", lambda name="t": provider.get_tracer(name))
    kwargs = {
        "query": QUESTION,
        "session_id": "s",
        "question_id": "qid",
        "tool_catalog": {},
        "context": AgentContext.for_eval("jdoe@access-ci.org", allow_writes=False),
    }

    async def _run():
        if stream:
            async for _ in graph_mod.stream_agent(**kwargs):
                pass
        else:
            await graph_mod.run_agent(**kwargs)

    asyncio.run(_run())
    (root,) = [s for s in exporter.get_finished_spans() if s.parent is None]
    return dict(root.attributes)


@pytest.mark.parametrize("stream", [False, True])
def test_root_span_carries_no_question_or_identity(monkeypatch, stream):
    attrs = _root_span_attributes(monkeypatch, stream)
    exported = " ".join(str(v) for v in attrs.values())
    assert QUESTION not in exported
    assert "jdoe" not in exported
    assert attrs["agent.query_length"] == len(QUESTION)


@pytest.mark.parametrize(
    "key", [*sorted(CONTENT_ATTRIBUTES), "gen_ai.prompt.0.content", "gen_ai.completion.0.content"]
)
def test_known_content_keys_are_filtered(key):
    assert is_content_attribute(key)


@pytest.mark.parametrize(
    "key", ["gen_ai.usage.prompt_tokens", "gen_ai.tool.name", "gen_ai.request.model"]
)
def test_metadata_keys_are_kept(key):
    assert not is_content_attribute(key)


def test_exporter_delegates_flush_and_shutdown():
    calls = []

    class _Inner(InMemorySpanExporter):
        def force_flush(self, timeout_millis=30000):
            calls.append(("flush", timeout_millis))
            return False

        def shutdown(self):
            calls.append(("shutdown",))

    exporter = ContentFilteringSpanExporter(_Inner())
    assert exporter.force_flush(123) is False
    exporter.shutdown()
    assert calls == [("flush", 123), ("shutdown",)]


def test_console_exporter_is_filtered(otel_globals, monkeypatch):  # noqa: F811
    from opentelemetry import trace

    from src.telemetry import init_telemetry, setup, shutdown_telemetry

    exporter = InMemorySpanExporter()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    monkeypatch.setattr("src.telemetry.setup.ConsoleSpanExporter", lambda: exporter)
    monkeypatch.setattr(settings, "OTEL_CAPTURE_CONTENT", False)
    monkeypatch.setenv("TRACELOOP_TRACE_CONTENT", "false")
    try:
        init_telemetry(service_name="test")
        with trace.get_tracer("test").start_as_current_span(
            "s", attributes={"gen_ai.system_instructions": "profile: anvil allocation"}
        ):
            pass
        setup._tracer_provider.force_flush()
        (span,) = exporter.get_finished_spans()
    finally:
        shutdown_telemetry()
    assert "gen_ai.system_instructions" not in span.attributes


def _failed_span(scope_name: str):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer(scope_name)
    with pytest.raises(ValueError), tracer.start_as_current_span("s"):
        raise ValueError(f"1 validation error: input_value='{QUESTION}'")
    (span,) = exporter.get_finished_spans()
    return span


def _filtered(span):
    inner = InMemorySpanExporter()
    ContentFilteringSpanExporter(inner).export([span])
    return inner.get_finished_spans()[0]


def test_library_error_text_is_dropped_but_failure_stays_visible():
    import opentelemetry.instrumentation.langchain as lc
    from opentelemetry.trace import StatusCode

    # The scope name openllmetry's tracer actually uses (get_tracer(__name__)).
    span = _failed_span(lc.__name__)
    assert QUESTION in span.status.description  # precondition: the SDK put it there

    out = _filtered(span)
    assert out.status.status_code is StatusCode.ERROR
    assert not out.status.description
    (event,) = out.events
    assert event.name == "exception"
    assert dict(event.attributes) == {"exception.type": "ValueError", "exception.escaped": "False"}


def test_our_spans_keep_status_text_but_lose_exception_text():
    from opentelemetry.trace import StatusCode

    span = _failed_span("access-agent")
    out = _filtered(span)
    # Our status descriptions are written to be content-free, so they stay.
    assert out.status == span.status
    assert out.status.status_code is StatusCode.ERROR
    (event,) = out.events
    assert dict(event.attributes) == {"exception.type": "ValueError", "exception.escaped": "False"}


def test_instrumentor_spans_use_the_filtered_scope():
    """LIBRARY_SCOPE must match the scope the real LangchainInstrumentor
    tracer produces. Runs in a subprocess because the instrumentor is a
    process-wide singleton."""
    import subprocess
    import sys

    from src.telemetry.content_filter import LIBRARY_SCOPE

    script = """
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from opentelemetry.instrumentation.langchain import LangchainInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
LangchainInstrumentor().instrument(tracer_provider=provider)
GenericFakeChatModel(messages=iter([AIMessage(content="a")])).invoke("q")
print("\\n".join(sorted({s.instrumentation_scope.name for s in exporter.get_finished_spans()})))
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    ).stdout.split()
    assert out, "the instrumentor produced no spans"
    assert all(name.startswith(LIBRARY_SCOPE) for name in out), out
