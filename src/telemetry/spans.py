"""Span creation utilities for agent tracing.

Provides decorators and context managers for creating spans
with standardized attributes.
"""

import functools
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from typing import Any, ParamSpec, TypeVar, cast

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .setup import get_tracer

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def trace_agent_node(
    node_name: str,
    *,
    extract_attributes: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """Decorator to trace an agent node function.

    Creates a span for the node execution with standardized attributes.

    Args:
        node_name: Name of the node (e.g., "plan", "execute", "synthesize").
        extract_attributes: Optional function to extract span attributes from state.

    Example:
        @trace_agent_node("plan", extract_attributes=lambda s: {"query": s.get("query")})
        async def plan_node(state: AgentState) -> dict:
            ...
    """

    def decorator(func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            tracer = get_tracer("access-agent.nodes")

            # Extract state from first positional arg (convention for node functions)
            state: dict[str, Any] = cast(
                "dict[str, Any]", args[0] if args else kwargs.get("state", {})
            )

            # Build attributes
            attributes: dict[str, Any] = {
                "agent.node": node_name,
            }

            # Add request_id if present
            if "request_id" in state:
                attributes["request_id"] = state["request_id"]

            # Add session_id if present
            if "session_id" in state:
                attributes["session_id"] = state["session_id"]

            # Extract custom attributes
            if extract_attributes and isinstance(state, dict):
                try:
                    custom_attrs = extract_attributes(state)
                    attributes.update(custom_attrs)
                except Exception as e:
                    logger.debug(f"Failed to extract attributes: {e}")

            with tracer.start_as_current_span(
                f"agent.{node_name}",
                attributes=attributes,
            ) as span:
                try:
                    result = await func(*args, **kwargs)

                    # Add result attributes for key nodes
                    if node_name == "plan" and isinstance(result, dict):
                        planned_tools = result.get("planned_tools", [])
                        span.set_attribute("agent.tools_planned", len(planned_tools))
                        if planned_tools:
                            tool_names = [
                                t.tool_name for t in planned_tools if hasattr(t, "tool_name")
                            ]
                            span.set_attribute("agent.tool_names", json.dumps(tool_names))

                            # Log the full plan for debugging
                            plan_details = [
                                {"tool": t.tool_name, "server": t.server, "args": t.arguments}
                                for t in planned_tools
                                if hasattr(t, "tool_name")
                            ]
                            span.add_event(
                                "plan_details",
                                {"plan": json.dumps(plan_details, default=str)},
                            )

                    elif node_name == "execute" and isinstance(result, dict):
                        tools_used = result.get("tools_used", [])
                        span.set_attribute("agent.tools_executed", len(tools_used))

                    elif node_name == "synthesize" and isinstance(result, dict):
                        answer = result.get("answer", "")
                        span.set_attribute("agent.answer_length", len(answer) if answer else 0)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


@contextmanager
def trace_mcp_call(
    server: str,
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> Any:
    """Context manager to trace an MCP tool call.

    Creates a span with tool call details and captures the result.

    Args:
        server: MCP server name.
        tool_name: Name of the tool being called.
        arguments: Tool arguments (sensitive values should be redacted).

    Example:
        with trace_mcp_call("system-status", "get_infrastructure_news", {"time": "current"}) as span:
            result = await client.call_tool(...)
            span.set_attribute("mcp.result_count", len(result.get("items", [])))
    """
    tracer = get_tracer("access-agent.mcp")

    # Redact potentially sensitive arguments
    safe_args = _redact_sensitive(arguments) if arguments else {}

    attributes = {
        "mcp.server": server,
        "mcp.tool": tool_name,
        "mcp.arguments": json.dumps(safe_args, default=str),
    }

    with tracer.start_as_current_span(
        f"mcp.call_tool.{tool_name}",
        attributes=attributes,
    ) as span:
        try:
            yield span
            span.set_status(Status(StatusCode.OK))
        except Exception as e:
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            raise


def _redact_sensitive(data: dict[str, Any]) -> dict[str, Any]:
    """Redact potentially sensitive values from arguments.

    Redacts values for keys that might contain PII or secrets.

    Args:
        data: Dictionary to redact.

    Returns:
        Dictionary with sensitive values replaced.
    """
    sensitive_keys = {
        "password",
        "token",
        "secret",
        "key",
        "auth",
        "credential",
        "user_id",
        "email",
    }
    redacted: dict[str, Any] = {}

    for key, value in data.items():
        key_lower = key.lower()
        if any(s in key_lower for s in sensitive_keys):
            redacted[key] = "[REDACTED]"
        elif isinstance(value, dict):
            redacted[key] = _redact_sensitive(value)
        else:
            redacted[key] = value

    return redacted


def add_span_event(name: str, attributes: dict[str, Any] | None = None) -> None:
    """Add an event to the current span.

    Useful for logging significant occurrences within a span.

    Args:
        name: Event name.
        attributes: Event attributes.
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.add_event(name, attributes=attributes or {})


def set_span_attribute(key: str, value: Any) -> None:
    """Set an attribute on the current span.

    Args:
        key: Attribute key.
        value: Attribute value.
    """
    span = trace.get_current_span()
    if span.is_recording():
        span.set_attribute(key, value)
