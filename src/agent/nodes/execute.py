"""Execute node - parallel tool execution.

This node executes the planned MCP tool calls, handling both parallel
and sequential execution strategies with dependency resolution.
"""

import asyncio
import logging
import re
from typing import Any

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
    query_analysis = state.get("query_analysis")
    planned_tools = state.get("planned_tools", [])

    # Skip execution if no tools needed
    if query_analysis and not query_analysis.requires_tools:
        logger.info("No tools needed, skipping execution")
        return {
            "tool_results": [],
            "tools_used": [],
        }

    if not planned_tools:
        logger.info("No tools planned, skipping execution")
        return {
            "tool_results": [],
            "tools_used": [],
        }

    strategy = state.get("execution_strategy", "parallel")
    logger.info(f"Executing {len(planned_tools)} tools with strategy: {strategy}")

    # Create MCP client
    mcp_client = MCPClient()

    # Execute based on strategy
    if strategy == "parallel":
        results = await _execute_parallel(mcp_client, planned_tools)
    elif strategy == "sequential":
        results = await _execute_sequential(mcp_client, planned_tools)
    else:  # mixed
        results = await _execute_mixed(mcp_client, planned_tools)

    # Extract successful tool names
    tools_used = [r.tool_name for r in results if r.success]

    logger.info(f"Execution complete: {len(tools_used)}/{len(results)} tools succeeded")

    return {
        "tool_results": results,
        "tools_used": tools_used,
    }


async def _execute_parallel(
    client: MCPClient,
    tools: list[ToolCall],
) -> list[ToolResult]:
    """Execute all tools in parallel.

    Args:
        client: MCP client for making calls.
        tools: List of tool calls to execute.

    Returns:
        List of tool results.
    """
    tasks = [_execute_single_tool(client, tool) for tool in tools]
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
                )
            )
        else:
            # result is ToolResult at this point
            processed.append(result)

    return processed


async def _execute_sequential(
    client: MCPClient,
    tools: list[ToolCall],
) -> list[ToolResult]:
    """Execute tools sequentially, resolving dependencies.

    Args:
        client: MCP client for making calls.
        tools: List of tool calls to execute.

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

        result = await _execute_single_tool(client, resolved_tool)
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
) -> list[ToolResult]:
    """Execute tools with mixed parallel/sequential based on dependencies.

    Groups tools by dependency level and executes each level in parallel.

    Args:
        client: MCP client for making calls.
        tools: List of tool calls to execute.

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
        level_results = await _execute_parallel(client, resolved_tools)
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
) -> ToolResult:
    """Execute a single MCP tool call.

    Args:
        client: MCP client for making calls.
        tool: Tool call to execute.

    Returns:
        ToolResult with success status and data or error.
    """
    logger.debug(f"Executing tool: {tool.tool_name} on {tool.server}")
    logger.debug(f"Arguments: {tool.arguments}")

    result = await client.call_tool(
        server=tool.server,
        tool_name=tool.tool_name,
        arguments=tool.arguments,
    )

    return ToolResult(
        step_id=tool.step_id,
        tool_name=tool.tool_name,
        server=tool.server,
        success=result.success,
        data=result.data,
        error=result.error,
        duration_ms=result.duration_ms,
    )


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
    current = result.data
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
