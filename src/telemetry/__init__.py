"""OpenTelemetry instrumentation for ACCESS Agent.

This module provides distributed tracing and observability for the agent,
exporting traces to Honeycomb (or any OTLP-compatible backend).

LLM calls are automatically traced via opentelemetry-instrumentation-langchain.
"""

from .http import create_async_client
from .setup import get_tracer, init_telemetry, shutdown_telemetry
from .spans import add_span_event, set_span_attribute, trace_agent_node, trace_mcp_call

__all__ = [
    "add_span_event",
    "create_async_client",
    "get_tracer",
    "init_telemetry",
    "set_span_attribute",
    "shutdown_telemetry",
    "trace_agent_node",
    "trace_mcp_call",
]
