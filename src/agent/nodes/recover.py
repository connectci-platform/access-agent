"""Recover node — DEPRECATED: superseded by tool_calling_loop_node (Phase 3).

This node runs only when USE_TOOL_CALLING_LOOP=false. The new path in
src/agent/nodes/tool_calling_loop.py folds planning, execution, evaluation,
and recovery into a single LLM-driven loop. See
docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md §Phase 3.

Retained for rollback safety until the feature-flag cutover is complete.

--- Original docstring below ---

Recover node - LLM-driven error recovery.

This node analyzes tool execution failures and decides on recovery strategy:
- fix_params: Retry with corrected parameters
- try_alternative: Use a different tool
- retry: Simple retry (for transient errors)
- fail: Give up and synthesize best-effort answer
"""

import logging
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ...config import settings
from ...llm import get_llm
from ..state import AgentState, ToolCall

logger = logging.getLogger(__name__)

RECOVERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You analyze tool execution errors and decide on recovery strategy.

Given a failed tool call, determine:
1. What caused the failure?
2. Is it recoverable?
3. What's the best recovery strategy?

Available strategies:
- "fix_params": The parameters were wrong - provide corrected parameters
- "try_alternative": Use a different tool that might have the data
- "retry": Transient error - simple retry might work
- "fail": Not recoverable - give up on this tool

Respond with JSON only:
{{
  "recoverable": true or false,
  "root_cause": "brief explanation of what went wrong",
  "strategy": "fix_params" | "try_alternative" | "retry" | "fail",
  "corrected_params": {{"param": "value"}} or null,
  "alternative_tool": "tool_name" or null,
  "confidence": "high" | "medium" | "low"
}}

AVAILABLE TOOLS FOR ALTERNATIVES:
{available_tools}""",
        ),
        (
            "human",
            """ORIGINAL QUERY: {query}

FAILED TOOL CALL:
- Tool: {tool_name}
- Server: {server}
- Parameters: {parameters}
- Error: {error}

ATTEMPT NUMBER: {attempt_number} of {max_attempts}

Analyze the failure and decide recovery strategy.""",
        ),
    ]
)


async def recover_node(state: AgentState) -> dict[str, Any]:
    """Analyze errors and plan recovery strategy.

    This node:
    1. Identifies failed tool calls
    2. Classifies the error type
    3. Uses LLM to decide recovery strategy
    4. Updates planned_tools for retry if recoverable

    Args:
        state: Current agent state with tool_results.

    Returns:
        Dict with updated planned_tools and retry_context.
    """
    tool_results = state.get("tool_results", [])
    planned_tools = state.get("planned_tools", [])
    retry_context = state.get("retry_context")
    query = state["query"]
    catalog = state["tool_catalog"]

    # Find failed results
    failed_results = [r for r in tool_results if not r.success]

    if not failed_results:
        logger.info("No failed tools to recover")
        return {"node_trace": [{"node": "recover", "action": "nothing_to_recover"}]}

    # Check retry limits
    if retry_context and retry_context.current_total_retries >= retry_context.max_retries_total:
        logger.warning("Max total retries exceeded - giving up")
        return {
            "planned_tools": [],
            "node_trace": [{"node": "recover", "action": "max_retries_exceeded"}],
        }

    # Analyze first failure (could extend to handle multiple)
    failed = failed_results[0]

    # Classify error for quick decisions
    error_type = _classify_error(failed.error or "")

    # Non-recoverable errors - fail fast
    if error_type in ("auth", "not_found"):
        logger.info(f"Non-recoverable error type: {error_type}")
        return {
            "planned_tools": [],
            "node_trace": [
                {"node": "recover", "action": "non_recoverable", "error_type": error_type}
            ],
        }

    # Build available tools list for alternatives
    available_tools = _get_available_tools(catalog, exclude=failed.tool_name)

    # Get LLM recovery decision
    llm = get_llm(temperature=0.1, max_tokens=settings.MAX_TOKENS_RECOVER)
    chain = RECOVERY_PROMPT | llm | JsonOutputParser()

    # Find the original tool call parameters
    original_call = next((t for t in planned_tools if t.tool_name == failed.tool_name), None)
    parameters = original_call.arguments if original_call else {}

    try:
        result = await chain.ainvoke(
            {
                "query": query,
                "tool_name": failed.tool_name,
                "server": failed.server,
                "parameters": str(parameters),
                "error": failed.error,
                "attempt_number": retry_context.current_total_retries + 1 if retry_context else 1,
                "max_attempts": retry_context.max_retries_total if retry_context else 5,
                "available_tools": available_tools,
            }
        )

        logger.info(
            f"Recovery decision: strategy={result.get('strategy')}, "
            f"recoverable={result.get('recoverable')}"
        )

        if not result.get("recoverable", False):
            return {
                "planned_tools": [],
                "node_trace": [{"node": "recover", "action": "llm_says_unrecoverable"}],
            }

        # Apply recovery strategy
        strategy = result.get("strategy", "fail")
        new_planned_tools = []

        if strategy == "fix_params" and result.get("corrected_params"):
            # Retry with corrected parameters
            new_planned_tools = [
                ToolCall(
                    step_id=f"retry_{failed.step_id}",
                    tool_name=failed.tool_name,
                    server=failed.server,
                    arguments=result["corrected_params"],
                )
            ]

        elif strategy == "try_alternative" and result.get("alternative_tool"):
            # Try a different tool
            alt_tool = result["alternative_tool"]
            alt_server = _get_server_for_tool(alt_tool, catalog)
            if alt_server:
                new_planned_tools = [
                    ToolCall(
                        step_id=f"alt_{failed.step_id}",
                        tool_name=alt_tool,
                        server=alt_server,
                        arguments=result.get("corrected_params", {}),
                    )
                ]

        elif strategy == "retry":
            # Simple retry with same parameters
            new_planned_tools = [
                ToolCall(
                    step_id=f"retry_{failed.step_id}",
                    tool_name=failed.tool_name,
                    server=failed.server,
                    arguments=parameters,
                )
            ]

        # Update retry context
        new_retry_context = None
        if retry_context:
            new_history_entry = {
                "tool": failed.tool_name,
                "error": failed.error,
                "strategy": strategy,
                "recoverable": result.get("recoverable", False),
            }
            new_retry_context = retry_context.model_copy(
                update={
                    "current_total_retries": retry_context.current_total_retries + 1,
                    "history": [*retry_context.history, new_history_entry],
                }
            )

        return {
            "planned_tools": new_planned_tools,
            "retry_context": new_retry_context,
            "tool_results": [],
            "node_trace": [
                {
                    "node": "recover",
                    "action": strategy,
                    "new_tools": [t.tool_name for t in new_planned_tools],
                }
            ],
        }

    except Exception as e:
        logger.error(f"Recovery planning failed: {e}")
        return {
            "planned_tools": [],
            "node_trace": [{"node": "recover", "action": "error", "error": str(e)[:100]}],
        }


def _classify_error(error: str) -> str:
    """Classify error for quick recovery decisions."""
    error_lower = error.lower()

    if "401" in error or "403" in error or "unauthorized" in error_lower:
        return "auth"
    # XDMoD "No Statistic found" or "Valid statistics for" = wrong parameter, recoverable
    if "no statistic found" in error_lower or "valid statistics for" in error_lower:
        return "parameter"
    if "404" in error or "not found" in error_lower:
        return "not_found"
    if "timeout" in error_lower or "etimedout" in error_lower:
        return "timeout"
    if "429" in error or "rate limit" in error_lower:
        return "rate_limit"
    if any(str(code) in error for code in range(500, 600)):
        return "server"
    if "400" in error or "validation" in error_lower or "parameter" in error_lower:
        return "parameter"
    return "unknown"


def _get_available_tools(catalog: dict[str, Any], exclude: str) -> str:
    """Get list of available tools for alternatives."""
    tools = []
    for tool in catalog.get("tools", []):
        if tool.get("name") != exclude:
            desc = tool.get("description", "")[:80]
            tools.append(f"- {tool['name']}: {desc}")

    return "\n".join(tools[:10])  # Limit to 10 for context


def _get_server_for_tool(tool_name: str, catalog: dict[str, Any]) -> str | None:
    """Get server name for a tool from catalog."""
    quick_lookup = catalog.get("quick_lookup", {})
    if tool_name in quick_lookup:
        server = quick_lookup[tool_name].get("server")
        return str(server) if server is not None else None
    return None
