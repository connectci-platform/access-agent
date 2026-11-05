"""Strip conversation content from spans before they leave the process.

openllmetry honors TRACELOOP_TRACE_CONTENT on its chat-model spans but not
everywhere: its create_agent wrapper writes the system prompt, which carries
the user's profile, unconditionally. Filtering at export covers every code
path, including ones a library upgrade adds.
"""

from collections.abc import Sequence
from typing import Any

from opentelemetry.sdk.trace import Event, ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.trace import Status
from opentelemetry.util.types import Attributes

CONTENT_ATTRIBUTES = frozenset(
    {
        "gen_ai.system_instructions",
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
        "gen_ai.task.input",
        "gen_ai.task.output",
        "traceloop.entity.input",
        "traceloop.entity.output",
        "traceloop.prompt.template_variables",
    }
)
# Older openllmetry releases index messages: gen_ai.prompt.0.content, ...
CONTENT_PREFIXES = ("gen_ai.prompt.", "gen_ai.completion.")

# An exception's message and traceback can render the arguments or text that
# caused it (a tool-argument validation error shows the model's arguments), so
# exception events keep only their type, on every span. openllmetry also puts
# str(error) in the status description, whatever TRACELOOP_TRACE_CONTENT says;
# our own status descriptions carry only status codes and exception types.
LIBRARY_SCOPE = "opentelemetry.instrumentation.langchain"
EXCEPTION_TEXT = frozenset({"exception.message", "exception.stacktrace"})


def is_content_attribute(key: str) -> bool:
    return key in CONTENT_ATTRIBUTES or key.startswith(CONTENT_PREFIXES)


def _without_content(attributes: Attributes) -> dict[str, Any]:
    return {k: v for k, v in (attributes or {}).items() if not is_content_attribute(k)}


def _is_library_span(span: ReadableSpan) -> bool:
    scope = span.instrumentation_scope
    return scope is not None and scope.name.startswith(LIBRARY_SCOPE)


def _event_without_content(event: Event) -> Event:
    attributes = _without_content(event.attributes)
    if event.name == "exception":
        attributes = {k: v for k, v in attributes.items() if k not in EXCEPTION_TEXT}
    return Event(event.name, attributes, event.timestamp)


def _has_content(span: ReadableSpan, library: bool) -> bool:
    if library and span.status.description:
        return True
    if any(e.name == "exception" and EXCEPTION_TEXT & set(e.attributes or {}) for e in span.events):
        return True
    keys = list(span.attributes or {})
    for event in span.events:
        keys.extend(event.attributes or {})
    return any(is_content_attribute(k) for k in keys)


def _strip(span: ReadableSpan) -> ReadableSpan:
    library = _is_library_span(span)
    if not _has_content(span, library):
        return span
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_without_content(span.attributes),
        events=[_event_without_content(e) for e in span.events],
        links=span.links,
        kind=span.kind,
        status=Status(span.status.status_code) if library else span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class ContentFilteringSpanExporter(SpanExporter):
    """Wraps an exporter and drops conversation-content attributes."""

    def __init__(self, inner: SpanExporter) -> None:
        self._inner = inner

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        return self._inner.export([_strip(s) for s in spans])

    def shutdown(self) -> None:
        self._inner.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._inner.force_flush(timeout_millis)
