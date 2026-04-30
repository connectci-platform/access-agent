"""Execute node — DEPRECATED: superseded by tool_calling_loop_node (Phase 3).

This node runs only when USE_TOOL_CALLING_LOOP=false. The new path in
src/agent/nodes/tool_calling_loop.py folds planning, execution, evaluation,
and recovery into a single LLM-driven loop. See
docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md §Phase 3.

Retained for rollback safety until the feature-flag cutover is complete.

--- Original docstring below ---

Execute node - parallel tool execution.

This node executes the planned MCP tool calls, handling both parallel
and sequential execution strategies with dependency resolution.
"""

import asyncio
import json
import logging
import re
import time
from typing import Any

from langgraph.config import get_stream_writer

from ...telemetry import get_tracer
from ...telemetry.spans import add_span_event
from ...tools import MCPClient
from ..state import AgentState, ToolCall, ToolResult

logger = logging.getLogger(__name__)


async def execute_node(state: AgentState) -> dict[str, Any]:
    """Execute planned tool calls.

    This node:
    1. Checks if tools are needed (skip if not)
    2. Executes tools in parallel or sequential based on strategy
    3. Resolves parameter dependencies between steps
    4. Collects and returns results

    Args:
        state: Current agent state with planned_tools.

    Returns:
        Dict with tool_results and tools_used.
    """
    tracer = get_tracer("access-agent.nodes")

    with tracer.start_as_current_span("agent.execute") as span:
        query_analysis = state.get("query_analysis")
        planned_tools = state.get("planned_tools", [])

        span.set_attribute("agent.node", "execute")
        span.set_attribute("agent.tools_planned", len(planned_tools))

        # Skip execution if no tools needed
        if query_analysis and not query_analysis.requires_tools:
            logger.info("No tools needed, skipping execution")
            span.set_attribute("agent.skipped", True)
            span.set_attribute("agent.skip_reason", "no_tools_needed")
            return {
                "tool_results": [],
                "tools_used": [],
                "node_trace": [{"node": "execute", "skipped": True, "reason": "no_tools_needed"}],
            }

        if not planned_tools:
            logger.info("No tools planned, skipping execution")
            span.set_attribute("agent.skipped", True)
            span.set_attribute("agent.skip_reason", "no_tools_planned")
            return {
                "tool_results": [],
                "tools_used": [],
                "node_trace": [{"node": "execute", "skipped": True, "reason": "no_tools_planned"}],
            }

        strategy = state.get("execution_strategy", "parallel")
        acting_user = state.get("acting_user")
        span.set_attribute("agent.strategy", strategy)

        # Emit per-tool status messages
        writer = get_stream_writer()
        tool_names = [t.tool_name for t in planned_tools]
        if len(tool_names) == 1:
            writer({"type": "status", "message": f"Querying {tool_names[0]}..."})
        else:
            writer({"type": "status", "message": f"Querying {len(tool_names)} tools..."})

        logger.info(f"Executing {len(planned_tools)} tools with strategy: {strategy}")

        # Create MCP client
        mcp_client = MCPClient()

        # Execute based on strategy
        if strategy == "parallel":
            results = await _execute_parallel(mcp_client, planned_tools, acting_user)
        elif strategy == "sequential":
            results = await _execute_sequential(mcp_client, planned_tools, acting_user)
        else:  # mixed
            results = await _execute_mixed(mcp_client, planned_tools, acting_user)

        # Extract successful tool names
        tools_used = [r.tool_name for r in results if r.success]

        # Log results to span
        span.set_attribute("agent.tools_executed", len(results))
        span.set_attribute("agent.tools_succeeded", len(tools_used))

        # Add detailed result info for each tool
        for result in results:
            event_attrs = {
                "tool": result.tool_name,
                "server": result.server,
                "success": result.success,
                "duration_ms": result.duration_ms,
            }
            if result.error:
                event_attrs["error"] = result.error
            if result.data:
                # Log a summary of the result data
                data_summary = _summarize_result(result.data)
                event_attrs["data_summary"] = data_summary
            add_span_event(f"tool_result.{result.tool_name}", event_attrs)

        failed = [r.tool_name for r in results if not r.success]
        logger.info(f"Execution complete: {len(tools_used)}/{len(results)} tools succeeded")

        return {
            "tool_results": results,
            "tools_used": tools_used,
            "node_trace": [
                {
                    "node": "execute",
                    "tools_called": [r.tool_name for r in results],
                    "succeeded": tools_used,
                    "failed": failed,
                }
            ],
        }


def _summarize_result(data: Any) -> str:
    """Create a brief summary of tool result data for tracing."""
    if isinstance(data, dict):
        # For dict results, show key counts
        summary_parts = []
        for key, value in data.items():
            if isinstance(value, list):
                summary_parts.append(f"{key}={len(value)} items")
            elif isinstance(value, int | float | str | bool):
                summary_parts.append(f"{key}={value}"[:50])
        return ", ".join(summary_parts[:5])
    if isinstance(data, list):
        return f"{len(data)} items"
    return str(type(data).__name__)


async def _execute_parallel(
    client: MCPClient,
    tools: list[ToolCall],
    acting_user: str | None = None,
) -> list[ToolResult]:
    """Execute all tools in parallel.

    Args:
        client: MCP client for making calls.
        tools: List of tool calls to execute.
        acting_user: ACCESS ID of user performing action.

    Returns:
        List of tool results.
    """
    tasks = [_execute_single_tool(client, tool, acting_user) for tool in tools]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convert exceptions to failed results
    processed: list[ToolResult] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            # Log full traceback for debugging unexpected errors
            logger.exception(f"Unexpected error executing {tools[i].tool_name}: {result}")
            processed.append(
                ToolResult(
                    step_id=tools[i].step_id,
                    tool_name=tools[i].tool_name,
                    server=tools[i].server,
                    success=False,
                    error=f"{type(result).__name__}: {result}",
                    duration_ms=0,
                    arguments=tools[i].arguments,
                )
            )
        else:
            # result is ToolResult at this point
            processed.append(result)

    return processed


async def _execute_sequential(
    client: MCPClient,
    tools: list[ToolCall],
    acting_user: str | None = None,
) -> list[ToolResult]:
    """Execute tools sequentially, resolving dependencies.

    Args:
        client: MCP client for making calls.
        tools: List of tool calls to execute.
        acting_user: ACCESS ID of user performing action.

    Returns:
        List of tool results.
    """
    results: list[ToolResult] = []
    results_by_step: dict[str, ToolResult] = {}

    for tool in tools:
        # Resolve parameter references from previous results
        resolved_args = _resolve_parameters(tool.arguments, results_by_step)
        resolved_tool = ToolCall(
            step_id=tool.step_id,
            tool_name=tool.tool_name,
            server=tool.server,
            arguments=resolved_args,
            depends_on=tool.depends_on,
        )

        result = await _execute_single_tool(client, resolved_tool, acting_user)
        results.append(result)
        results_by_step[tool.step_id] = result

        # Stop on failure in sequential mode
        if not result.success:
            logger.warning(f"Stopping sequential execution due to failure: {tool.tool_name}")
            break

    return results


async def _execute_mixed(
    client: MCPClient,
    tools: list[ToolCall],
    acting_user: str | None = None,
) -> list[ToolResult]:
    """Execute tools with mixed parallel/sequential based on dependencies.

    Groups tools by dependency level and executes each level in parallel.

    Args:
        client: MCP client for making calls.
        tools: List of tool calls to execute.
        acting_user: ACCESS ID of user performing action.

    Returns:
        List of tool results.
    """
    # Build dependency levels
    levels = _build_dependency_levels(tools)
    results: list[ToolResult] = []
    results_by_step: dict[str, ToolResult] = {}

    for level_tools in levels:
        # Resolve parameters for this level
        resolved_tools = []
        for tool in level_tools:
            resolved_args = _resolve_parameters(tool.arguments, results_by_step)
            resolved_tools.append(
                ToolCall(
                    step_id=tool.step_id,
                    tool_name=tool.tool_name,
                    server=tool.server,
                    arguments=resolved_args,
                    depends_on=tool.depends_on,
                )
            )

        # Execute level in parallel
        level_results = await _execute_parallel(client, resolved_tools, acting_user)
        results.extend(level_results)

        # Store results for next level
        for r in level_results:
            results_by_step[r.step_id] = r

    return results


def _build_dependency_levels(tools: list[ToolCall]) -> list[list[ToolCall]]:
    """Build execution levels based on dependencies.

    Tools with no dependencies go in level 0, tools depending on level 0
    go in level 1, etc.

    Args:
        tools: List of tool calls.

    Returns:
        List of levels, each containing tools that can run in parallel.
    """
    levels: list[list[ToolCall]] = []
    assigned: set[str] = set()

    while len(assigned) < len(tools):
        # Find tools whose dependencies are all assigned
        current_level = []
        for tool in tools:
            if tool.step_id in assigned:
                continue
            if all(dep in assigned for dep in tool.depends_on):
                current_level.append(tool)

        if not current_level:
            # Circular dependency or missing dependency - add remaining
            for tool in tools:
                if tool.step_id not in assigned:
                    current_level.append(tool)
            levels.append(current_level)
            break

        levels.append(current_level)
        for tool in current_level:
            assigned.add(tool.step_id)

    return levels


async def _execute_single_tool(
    client: MCPClient,
    tool: ToolCall,
    acting_user: str | None = None,
) -> ToolResult:
    """Execute a single MCP tool call.

    Args:
        client: MCP client for making calls.
        tool: Tool call to execute.
        acting_user: ACCESS ID of user performing action.

    Returns:
        ToolResult with success status and data or error.
    """
    tracer = get_tracer("access-agent.mcp")

    with tracer.start_as_current_span(f"mcp.call_tool.{tool.tool_name}") as span:
        span.set_attribute("mcp.server", tool.server)
        span.set_attribute("mcp.tool", tool.tool_name)
        span.set_attribute("mcp.arguments", json.dumps(tool.arguments, default=str))
        span.set_attribute("mcp.step_id", tool.step_id)

        logger.debug(f"Executing tool: {tool.tool_name} on {tool.server}")
        logger.debug(f"Arguments: {tool.arguments}")

        start_time = time.perf_counter()

        result = await client.call_tool(
            server=tool.server,
            tool_name=tool.tool_name,
            arguments=tool.arguments,
            acting_user=acting_user,
        )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Add result info to span
        span.set_attribute("mcp.success", result.success)
        span.set_attribute("mcp.duration_ms", result.duration_ms or duration_ms)

        if result.error:
            span.set_attribute("mcp.error", result.error)

        if result.data and isinstance(result.data, dict):
            # Log summary of result data
            for key in ["total", "total_outages", "total_items", "total_events", "count"]:
                if key in result.data:
                    span.set_attribute(f"mcp.result.{key}", result.data[key])

        return ToolResult(
            step_id=tool.step_id,
            tool_name=tool.tool_name,
            server=tool.server,
            success=result.success,
            data=result.data,
            error=result.error,
            duration_ms=result.duration_ms,
            arguments=tool.arguments,
        )


# DEPRECATED (Phase 3): used only by the legacy plan→execute path when
# USE_TOOL_CALLING_LOOP=false. The new tool_calling_loop does not need
# $step_N substitution — tool outputs flow via ToolMessage content and
# the LLM reads actual values on subsequent turns. Delete alongside
# the legacy path during the post-launch cleanup.
def _resolve_parameters(
    arguments: dict[str, Any],
    previous_results: dict[str, ToolResult],
) -> dict[str, Any]:
    """Resolve parameter references to previous step results.

    Handles references like "$step_1.data.resources[0].id" by extracting
    the value from the corresponding step's result.

    Args:
        arguments: Original arguments dict.
        previous_results: Results from previous steps.

    Returns:
        Arguments dict with references resolved to actual values.
    """
    resolved = {}

    for key, value in arguments.items():
        if isinstance(value, str) and value.startswith("$step"):
            resolved[key] = _resolve_reference(value, previous_results)
        elif isinstance(value, dict):
            resolved[key] = _resolve_parameters(value, previous_results)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_reference(v, previous_results)
                if isinstance(v, str) and v.startswith("$step")
                else v
                for v in value
            ]
        else:
            resolved[key] = value

    return resolved


# DEPRECATED (Phase 3): used only by the legacy plan→execute path when
# USE_TOOL_CALLING_LOOP=false. The new tool_calling_loop does not need
# $step_N substitution — tool outputs flow via ToolMessage content and
# the LLM reads actual values on subsequent turns. Delete alongside
# the legacy path during the post-launch cleanup.
def _resolve_reference(
    reference: str,
    previous_results: dict[str, ToolResult],
) -> Any:
    """Resolve a single parameter reference.

    Reference format: $step_1.data.path.to.value
    or: $step_1.data.array[0].field

    Args:
        reference: Reference string starting with $step.
        previous_results: Results from previous steps.

    Returns:
        The resolved value, or the original reference if not found.
    """
    # Parse reference: $step_1.data.resources[0].id
    match = re.match(r"\$(step_\d+)\.(.+)", reference)
    if not match:
        return reference

    step_id = match.group(1)
    path = match.group(2)

    if step_id not in previous_results:
        logger.warning(f"Reference to unknown step: {step_id}")
        return reference

    result = previous_results[step_id]
    if not result.success or result.data is None:
        logger.warning(f"Reference to failed step: {step_id}")
        return reference

    # Navigate the path
    current: object = result.data
    for part in _parse_path(path):
        if isinstance(part, int):
            # Array index
            if isinstance(current, list) and 0 <= part < len(current):
                current = current[part]
            else:
                return reference
        # Object key
        elif isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return reference

    return current


def _parse_path(path: str) -> list[str | int]:
    """Parse a path string into parts.

    Handles: "data.resources[0].id" -> ["data", "resources", 0, "id"]

    Args:
        path: Path string with dots and array indices.

    Returns:
        List of path parts (strings and integers).
    """
    parts: list[str | int] = []
    current = ""

    i = 0
    while i < len(path):
        char = path[i]

        if char == ".":
            if current:
                parts.append(current)
                current = ""
        elif char == "[":
            if current:
                parts.append(current)
                current = ""
            # Find closing bracket
            end = path.find("]", i)
            if end != -1:
                index_str = path[i + 1 : end]
                try:
                    parts.append(int(index_str))
                except ValueError:
                    parts.append(index_str)
                i = end
        else:
            current += char

        i += 1

    if current:
        parts.append(current)

    return parts
