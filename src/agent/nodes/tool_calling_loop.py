"""Tool-calling loop node — the only node in the agent graph.

LangGraph's ``create_react_agent`` drives a turn-by-turn loop where the LLM
selects tools, sees results as ToolMessages, and continues until it emits a
non-tool-call response. The system prompt instructs the LLM to call
``search_access_documents`` for documentation-style questions; live data
flows from the MCP catalog tools.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

from ...config import settings
from ...llm import get_llm
from ...telemetry import get_tracer
from ..domains.capabilities import WRITE_MCP_TOOL_NAMES
from ..domains.tools import create_mcp_tools_from_catalog
from ..prompts.no_classify import build_system_prompt_no_classify
from ..state import ToolResult
from ..tools import search_access_documents

logger = logging.getLogger(__name__)


def _build_prompt_and_tools(
    *,
    tool_catalog: dict[str, Any],
    acting_user: str | None,
    resource_context: str | None,
) -> tuple[str, list[BaseTool]]:
    """Assemble the loop's system prompt and tool list.

    Tool list = MCP catalog (read-only-filtered) + ``search_access_documents``.
    Prompt is the no-classify variant (docs-as-tool framing, announcements +
    JSM choreography appended).
    """
    mcp_tools = _apply_read_only_filter(create_mcp_tools_from_catalog(tool_catalog, acting_user))
    prompt = build_system_prompt_no_classify(
        acting_user=acting_user,
        resource_context=resource_context,
    )
    return prompt, [*mcp_tools, search_access_documents]


def _apply_read_only_filter(tools: list[BaseTool]) -> list[BaseTool]:
    """Strip write-capable MCP tools when ``settings.READ_ONLY`` is True.

    See `docs/security/write-capability-audit.md`.
    """
    if not settings.READ_ONLY:
        return tools
    before = len(tools)
    filtered = [t for t in tools if t.name not in WRITE_MCP_TOOL_NAMES]
    removed = before - len(filtered)
    if removed:
        logger.info(
            "READ_ONLY=true active in tool_calling_loop — removed %d "
            "write-capable tool(s) from registry",
            removed,
        )
    return filtered


def _emit_status(message: str) -> None:
    """Emit a status event to the SSE stream, no-op outside runnable context.

    ``get_stream_writer()`` raises ``RuntimeError`` when called outside a
    LangGraph runnable (eg. direct invocation in tests). Wrapping here keeps
    the call sites compact and lets tests exercise the node directly.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return
    writer({"type": "status", "message": message})


async def tool_calling_loop_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the tool-calling loop.

    Consumes:
        state.messages: conversation history (HumanMessage / AIMessage / ToolMessage)
        state.query: current user query (already present in messages[-1])
        state.tool_catalog: dict describing MCP tools available to this request
        state.acting_user: optional ACCESS ID for personalized tool calls
        state.resource_context: optional RP slug for resource-scoped queries

    Produces:
        final_answer: LLM's final text response
        messages: full loop message history (caller messages + tool calls + result messages + final)
        tools_used: list of tool-name strings the loop actually invoked
        node_trace: telemetry entry for this node
    """
    tracer = get_tracer("access-agent.nodes")

    _emit_status("Processing query...")

    with tracer.start_as_current_span(
        "agent.tool_calling_loop",
        attributes={"agent.node": "tool_calling_loop"},
    ) as span:
        acting_user = state.get("acting_user")

        system_prompt, tools = _build_prompt_and_tools(
            tool_catalog=state.get("tool_catalog") or {},
            acting_user=acting_user,
            resource_context=state.get("resource_context"),
        )

        span.set_attribute("agent.tool_count", len(tools))
        span.set_attribute("agent.authenticated", bool(acting_user))

        llm = get_llm(max_tokens=settings.MAX_TOKENS_LOOP)
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        messages = list(state.get("messages", []))
        logger.info(
            "Running tool_calling_loop: %d tools, %d prior messages",
            len(tools),
            len(messages),
        )

        final_answer: str | None = None
        recursion_limit_hit = False

        # recursion_limit = 2 * max_tool_turns + 1; gives the LLM room for
        # roughly 10 tool turns before LangGraph hard-stops.
        recursion_limit = 25
        try:
            result = await agent.ainvoke(
                {"messages": messages},
                {"recursion_limit": recursion_limit},
            )
            result_messages = result.get("messages", [])
        except GraphRecursionError:
            # create_react_agent exhausted its recursion budget (e.g., the LLM
            # kept requesting tool calls and never emitted a final answer).
            # Return a user-facing apology rather than a 500 so the chatbot
            # can render something useful.
            recursion_limit_hit = True
            logger.warning(
                "tool_calling_loop hit GraphRecursionError: "
                "tool_count=%d, recursion_limit=%d, messages_so_far=%d",
                len(tools),
                recursion_limit,
                len(messages),
            )
            span.set_attribute("agent.recursion_limit_hit", True)
            result_messages = list(messages)
            final_answer = (
                "I wasn't able to complete an answer for this query within "
                "the allotted tool-turn budget. You can try rephrasing, or "
                "open a support ticket at "
                "https://support.access-ci.org/open-a-ticket."
            )

        if not recursion_limit_hit:
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    # AIMessage.content can be str or list[str|dict] (multimodal);
                    # the loop only emits string content for final answers.
                    content = msg.content
                    final_answer = content if isinstance(content, str) else str(content)
                    break

        tools_used: list[str] = []
        for msg in result_messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name and name not in tools_used:
                        tools_used.append(name)

        tool_result_count = sum(1 for m in result_messages if isinstance(m, ToolMessage))
        tool_results, orphan_count = _build_tool_results(result_messages, tools)

        answer_length = len(final_answer) if final_answer else 0
        span.set_attribute("agent.answer_length", answer_length)
        span.set_attribute("agent.tool_calls_made", len(tools_used))
        span.set_attribute("agent.tool_results_received", tool_result_count)
        if orphan_count > 0:
            span.set_attribute("agent.tool_results_orphaned", orphan_count)

        logger.info(
            "tool_calling_loop complete: %d messages, %d tool calls, "
            "%d tool results, answer_len=%d",
            len(result_messages),
            len(tools_used),
            tool_result_count,
            answer_length,
        )

        return {
            "final_answer": final_answer,
            "messages": result_messages,
            "tools_used": tools_used,
            "tool_results": tool_results,
            "node_trace": [
                {
                    "node": "tool_calling_loop",
                    "tool_count": len(tools),
                    "tool_calls_made": len(tools_used),
                    "tools_called": list(tools_used),
                    "tool_results": tool_result_count,
                    "answer_length": answer_length,
                }
            ],
        }


def _build_tool_results(
    result_messages: list[Any],
    tools: list[Any],
) -> tuple[list[ToolResult], int]:
    """Back-fill state.tool_results from a react-loop message thread.

    Downstream consumers (eval scorer, observability) read state.tool_results
    in a structured form. The react loop only emits ToolMessages, so we
    reconstruct ToolResult objects by pairing each ToolMessage with its
    originating AIMessage tool_call (by id) and parsing the JSON content
    MCPToolWrapper produced.

    Orphan ToolMessages (no matching tool_call id) are dropped — we never
    fabricate a ToolResult we can't anchor to an AIMessage call. The caller
    still receives the orphan count so telemetry can surface these rare
    cases (e.g., future LLM quirks where a ToolMessage appears without a
    matching call).

    duration_ms is left at 0 because the react loop doesn't track per-call
    timing; the eval scorer doesn't depend on it.

    Returns:
        (tool_results, orphan_count) — the reconstructed ToolResult list and
        the number of ToolMessages skipped because their tool_call_id had no
        matching AIMessage tool_call.
    """
    tool_server_lookup: dict[str, str] = {
        getattr(t, "name", ""): getattr(t, "tool_server", "") for t in tools
    }

    call_lookup: dict[str, tuple[str, dict[str, Any]]] = {}
    for msg in result_messages:
        if not (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)):
            continue
        for tc in msg.tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            if tc_id and tc_name:
                call_lookup[tc_id] = (tc_name, tc_args or {})

    results: list[ToolResult] = []
    orphan_count = 0
    for msg in result_messages:
        if not isinstance(msg, ToolMessage):
            continue
        tc_id = msg.tool_call_id
        if not tc_id or tc_id not in call_lookup:
            orphan_count += 1
            continue
        tool_name, tool_args = call_lookup[tc_id]
        server = tool_server_lookup.get(tool_name, "")
        results.append(_parse_tool_message(msg, tc_id, tool_name, server, tool_args))

    return results, orphan_count


def _parse_tool_message(
    msg: ToolMessage,
    tc_id: str,
    tool_name: str,
    server: str,
    tool_args: dict[str, Any],
) -> ToolResult:
    """Build a ToolResult from a single ToolMessage + its originating call metadata."""
    raw_content = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        parsed: Any = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        parsed = raw_content

    if isinstance(parsed, dict) and "error" in parsed:
        return ToolResult(
            step_id=tc_id,
            tool_name=tool_name,
            server=server,
            success=False,
            error=str(parsed["error"]),
            arguments=tool_args,
        )

    return ToolResult(
        step_id=tc_id,
        tool_name=tool_name,
        server=server,
        success=True,
        data=parsed,
        arguments=tool_args,
    )
