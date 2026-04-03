"""Plan node - LLM-driven tool selection.

This node analyzes the user's query and selects which MCP tools to call,
mirroring the Query Planner logic from the n8n workflow.
"""

import json
import logging
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer

from ...llm import get_llm
from ...telemetry import get_tracer
from ...telemetry.spans import add_span_event
from ..state import AgentState, QueryAnalysis, ToolCall

logger = logging.getLogger(__name__)

# System prompt for tool selection - adapted from n8n workflow
PLANNING_SYSTEM_PROMPT = """You are a tool selection assistant for ACCESS-CI documentation queries.

Your job is to analyze the user's question and select which MCP tools to call to gather the information needed to answer it.

## CONVERSATION HISTORY

{conversation_history}

## AVAILABLE TOOLS

{tool_catalog}

## RESPONSE FORMAT

You MUST respond with valid JSON only, no markdown or explanation:

{{
  "query_analysis": {{
    "user_intent": "Brief summary of what the user wants to know",
    "entities_mentioned": ["list", "of", "key", "entities"],
    "requires_tools": true
  }},
  "execution_plan": {{
    "strategy": "parallel",
    "steps": [
      {{
        "step_id": "step_1",
        "tool_name": "exact_tool_name_from_catalog",
        "server": "server_name",
        "arguments": {{"param_name": "value"}},
        "depends_on": []
      }}
    ]
  }},
  "confidence": "high"
}}

## RULES

1. Return ONLY valid JSON - no markdown code blocks, no explanations
2. Use EXACT tool names from the catalog above
3. Extract parameter values from the user's question when possible
4. Use "parallel" strategy when tools are independent
5. Use "sequential" strategy with depends_on when one tool needs output from another
6. For general questions that don't need live data, set requires_tools: false
7. Prefer fewer, more specific tools over many broad tools
8. Maximum 4 tools per query unless absolutely necessary
9. Use CONVERSATION HISTORY to understand context - if user asks "which one has X", look at previous messages to understand what "one" refers to
10. Only include optional filter parameters when the user's question explicitly requires filtering
11. IMPORTANT: When looking for specific categories of software (e.g., "AI tools", "machine learning", "MPI libraries"), use search_software with a query filter rather than list_all_software. The search_software tool returns filtered results which are much more efficient.

## GUIDELINES

- For general questions that don't need live data (e.g., "How do I acknowledge ACCESS?"), set requires_tools: false
- Use conversation history to resolve references (e.g., "which one" refers to previously discussed items)
- Select tools based on their descriptions in the catalog above
- Prefer tools that return data directly over discovery/metadata tools. Only use discovery tools when the direct tool's description doesn't list the parameter value you need
- For questions about aggregate counts, totals, trends, or utilization (e.g., "how many projects", "total CPU hours", "active users"), prefer XDMoD get_chart_data over domain-specific search/list tools. XDMoD tracks aggregate metrics across all ACCESS resources.
"""


async def plan_node(state: AgentState) -> dict[str, Any]:
    """Analyze query and select tools to execute.

    This node:
    1. Builds a compact tool catalog for the LLM
    2. Includes conversation history for context
    3. Asks the LLM to analyze the query and select tools
    4. Parses and validates the tool selections

    Args:
        state: Current agent state with query, tool_catalog, and messages.

    Returns:
        Dict with query_analysis, planned_tools, and execution_strategy.
    """
    writer = get_stream_writer()
    writer({"type": "status", "message": "Planning tool calls..."})

    tracer = get_tracer("access-agent.nodes")

    with tracer.start_as_current_span("agent.plan") as span:
        query = state["query"]
        catalog = state["tool_catalog"]
        messages = state.get("messages", [])

        # Add query to span
        span.set_attribute("agent.query", query[:500] if query else "")
        span.set_attribute("agent.node", "plan")

        # Build compact tool catalog for prompt
        tool_catalog_text = _build_tool_catalog_text(catalog)

        # Build conversation history (exclude current query which is the last message)
        conversation_history = _build_conversation_history(messages[:-1] if messages else [])

        # Create the prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PLANNING_SYSTEM_PROMPT),
                ("human", "{query}"),
            ]
        )

        # Get LLM and create chain
        llm = get_llm(temperature=0.1, max_tokens=1500)
        chain = prompt | llm | JsonOutputParser()

        try:
            result = await chain.ainvoke(
                {
                    "tool_catalog": tool_catalog_text,
                    "conversation_history": conversation_history,
                    "query": query,
                }
            )

            # Log the raw LLM response for debugging
            add_span_event("llm_response", {"raw_result": json.dumps(result, default=str)[:2000]})

            # Parse query analysis
            analysis_data = result.get("query_analysis", {})
            query_analysis = QueryAnalysis(
                user_intent=analysis_data.get("user_intent", "Unknown intent"),
                entities_mentioned=analysis_data.get("entities_mentioned", []),
                requires_tools=analysis_data.get("requires_tools", True),
                confidence=result.get("confidence", "medium"),
            )

            # Parse execution plan
            exec_plan = result.get("execution_plan", {})
            strategy = exec_plan.get("strategy", "parallel")

            # Parse tool calls
            planned_tools: list[ToolCall] = []
            for step in exec_plan.get("steps", []):
                tool_call = ToolCall(
                    step_id=step.get("step_id", f"step_{len(planned_tools) + 1}"),
                    tool_name=step.get("tool_name", ""),
                    server=step.get("server", ""),
                    arguments=step.get("arguments", {}),
                    depends_on=step.get("depends_on", []),
                )

                # Validate tool exists in catalog
                if _validate_tool(tool_call.tool_name, catalog):
                    # Fill in server if not provided
                    if not tool_call.server:
                        tool_call.server = _get_server_for_tool(tool_call.tool_name, catalog)
                    planned_tools.append(tool_call)
                else:
                    logger.warning(f"LLM selected unknown tool: {tool_call.tool_name}")

            # Add detailed plan to span for debugging
            span.set_attribute("agent.tools_planned", len(planned_tools))
            span.set_attribute("agent.strategy", strategy)
            span.set_attribute("agent.requires_tools", query_analysis.requires_tools)

            # Log each planned tool with its arguments (this is the key debugging info!)
            for tool in planned_tools:
                add_span_event(
                    f"planned_tool.{tool.tool_name}",
                    {
                        "server": tool.server,
                        "arguments": json.dumps(tool.arguments, default=str),
                        "step_id": tool.step_id,
                    },
                )

            logger.info(
                f"Planning complete: {len(planned_tools)} tools selected, "
                f"strategy={strategy}, requires_tools={query_analysis.requires_tools}"
            )

            return {
                "query_analysis": query_analysis,
                "planned_tools": planned_tools,
                "execution_strategy": strategy,
                "node_trace": [
                    {
                        "node": "plan",
                        "requires_tools": query_analysis.requires_tools,
                        "tool_count": len(planned_tools),
                        "tools": [t.tool_name for t in planned_tools],
                        "strategy": strategy,
                    }
                ],
            }

        except Exception as e:
            span.record_exception(e)
            logger.error(f"Planning failed: {e}")
            return {
                "query_analysis": QueryAnalysis(
                    user_intent="Error during planning",
                    requires_tools=False,
                    confidence="low",
                ),
                "planned_tools": [],
                "execution_strategy": "sequential",
                "node_trace": [
                    {
                        "node": "plan",
                        "error": str(e)[:200],
                        "requires_tools": False,
                        "tool_count": 0,
                        "tools": [],
                        "strategy": "sequential",
                    }
                ],
            }


def _build_conversation_history(messages: list[Any]) -> str:
    """Build a text representation of conversation history.

    Args:
        messages: List of previous messages (HumanMessage/AIMessage).

    Returns:
        Formatted conversation history string.
    """
    if not messages:
        return "(No previous conversation)"

    lines = []
    for msg in messages[-6:]:  # Last 6 messages (3 turns) for context
        role = "User" if msg.type == "human" else "Assistant"
        # Truncate long messages
        content = msg.content
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def _build_tool_catalog_text(catalog: dict[str, Any]) -> str:
    """Build a compact text representation of the tool catalog.

    Args:
        catalog: The full MCP tool catalog. Supports multiple formats:
            - {"servers": [{"server": "name", "tools": [...]}]}
            - {"tools": [...], "quick_lookup": {...}}
            - {"server_name": [tools...], ...}  (simple format)

    Returns:
        A formatted string for the LLM prompt.
    """
    lines = []

    # Handle different catalog formats
    if "servers" in catalog:
        for server_info in catalog["servers"]:
            server_name = server_info.get("server", "")
            for tool in server_info.get("tools", []):
                line = _format_tool_line(tool, server_name)
                lines.append(line)
    elif "tools" in catalog:
        quick_lookup = catalog.get("quick_lookup", {})
        for tool in catalog["tools"]:
            server_name = quick_lookup.get(tool["name"], {}).get("server", "")
            line = _format_tool_line(tool, server_name)
            lines.append(line)
    else:
        # Simple format: {server_name: [tools...]}
        for server_name, tools in catalog.items():
            if isinstance(tools, list):
                for tool in tools:
                    line = _format_tool_line(tool, server_name)
                    lines.append(line)

    return "\n".join(lines)


TOOL_CAVEATS: dict[str, str] = {
    "search_projects": (
        "NOTE: This is a PUBLIC catalog search across all ACCESS research projects. "
        "It has no user/owner parameter and CANNOT look up a specific user's projects "
        "or allocations. Do not use for 'my projects' or 'my allocations' queries."
    ),
}


def _format_tool_line(tool: dict[str, Any], server_name: str) -> str:
    """Format a single tool for the catalog text."""
    name = tool.get("name", "")
    desc = tool.get("description", "")[:500]

    caveat = TOOL_CAVEATS.get(name, "")
    if caveat:
        desc = f"{desc} {caveat}"

    # Build parameter string with descriptions, enum values, and defaults
    # Support both "parameters" (list format) and "inputSchema" (JSON Schema format)
    params = tool.get("parameters", [])
    param_strs = []

    if params:
        # List format: [{"name": "x", "type": "string", "required": true, "description": "...", "default": ...}]
        for p in params:
            param_strs.append(
                _format_param(
                    name=p.get("name", ""),
                    ptype=p.get("type", "string"),
                    required=p.get("required", False),
                    description=p.get("description", ""),
                    enum_vals=p.get("enum"),
                    default=p.get("default"),
                )
            )
    else:
        # JSON Schema format: {"inputSchema": {"properties": {...}, "required": [...]}}
        input_schema = tool.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = input_schema.get("required", [])

        for pname, pschema in properties.items():
            param_strs.append(
                _format_param(
                    name=pname,
                    ptype=pschema.get("type", "string"),
                    required=pname in required_params,
                    description=pschema.get("description", ""),
                    enum_vals=pschema.get("enum"),
                    default=pschema.get("default"),
                )
            )

    params_text = "\n    ".join(param_strs) if param_strs else "none"

    return f"- {name} (server: {server_name}): {desc}\n    Parameters:\n    {params_text}"


def _format_param(
    name: str,
    ptype: str,
    required: bool,
    description: str,
    enum_vals: list[str] | None,
    default: Any,
) -> str:
    """Format a single parameter with its metadata."""
    parts = [f"{name}"]

    # Type and required marker
    if required:
        parts.append(f"({ptype}, REQUIRED)")
    else:
        parts.append(f"({ptype}, optional)")

    # Enum values
    if enum_vals:
        enum_str = "|".join(str(v) for v in enum_vals)
        parts.append(f"[one of: {enum_str}]")

    # Default value
    if default is not None:
        parts.append(f"[default: {default}]")

    # Description - important for understanding when to use/omit
    if description:
        parts.append(f"- {description}")

    return " ".join(parts)


def _validate_tool(tool_name: str, catalog: dict[str, Any]) -> bool:
    """Check if a tool exists in the catalog."""
    # Check quick_lookup first
    if "quick_lookup" in catalog and tool_name in catalog["quick_lookup"]:
        return True

    # Check tools array
    if "tools" in catalog:
        return any(t.get("name") == tool_name for t in catalog["tools"])

    # Check servers format
    if "servers" in catalog:
        for server in catalog["servers"]:
            if any(t.get("name") == tool_name for t in server.get("tools", [])):
                return True

    # Check simple format: {server_name: [tools...]}
    for value in catalog.values():
        if isinstance(value, list) and any(t.get("name") == tool_name for t in value):
            return True

    return False


def _get_server_for_tool(tool_name: str, catalog: dict[str, Any]) -> str:
    """Get the server name for a tool from the catalog."""
    # Check quick_lookup
    if "quick_lookup" in catalog:
        lookup = catalog["quick_lookup"].get(tool_name, {})
        if "server" in lookup:
            return str(lookup["server"])

    # Check servers format
    if "servers" in catalog:
        for server in catalog["servers"]:
            if any(t.get("name") == tool_name for t in server.get("tools", [])):
                return str(server.get("server", ""))

    # Check simple format: {server_name: [tools...]}
    for server_name, tools in catalog.items():
        if isinstance(tools, list) and any(t.get("name") == tool_name for t in tools):
            return server_name

    return ""
