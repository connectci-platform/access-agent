"""Tool-calling loop node — single-node replacement for plan+execute+evaluate+recover.

Launched behind the USE_TOOL_CALLING_LOOP feature flag (launch Phase 3). When
the flag is True, tool-using queries route here instead of the legacy chain.

Approach: LangGraph's `create_react_agent` drives a turn-by-turn loop where the
LLM selects tools, sees results as ToolMessages, and continues until it emits
a non-tool-call response. Planning, execution, evaluation, and recovery all
happen inside that loop — no separate nodes needed.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt import create_react_agent

from ...llm import get_llm
from ...telemetry import get_tracer
from ..domains.tools import create_mcp_tools_from_catalog
from ..prompts.tool_calling_loop import build_system_prompt, format_rag_matches

logger = logging.getLogger(__name__)


async def tool_calling_loop_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the single-node tool-calling loop.

    Consumes:
        state.messages: conversation history (HumanMessage / AIMessage / ToolMessage)
        state.query: current user query (already present in messages[-1])
        state.tool_catalog: dict describing MCP tools available to this request
        state.acting_user: optional ACCESS ID for personalized tool calls
        state.rag_matches: optional list of RAGMatch objects from rag_answer
        state.query_classification: optional classifier output with .domain

    Produces:
        final_answer: LLM's final text response
        messages: full loop message history (caller messages + tool calls + result messages + final)
        tools_used: list of tool-name strings the loop actually invoked
        node_trace: telemetry entry for this node
    """
    tracer = get_tracer("access-agent.nodes")

    with tracer.start_as_current_span(
        "agent.tool_calling_loop",
        attributes={"agent.node": "tool_calling_loop"},
    ) as span:
        acting_user = state.get("acting_user")
        rag_matches = state.get("rag_matches") or []
        classification = state.get("query_classification")
        domain_hint = classification.domain if classification else None
        tool_catalog = state.get("tool_catalog") or {}

        rag_context = format_rag_matches(rag_matches) if rag_matches else None
        system_prompt = build_system_prompt(
            rag_context=rag_context,
            domain_hint=domain_hint,
            acting_user=acting_user,
        )

        tools = create_mcp_tools_from_catalog(tool_catalog, acting_user)

        span.set_attribute("agent.tool_count", len(tools))
        span.set_attribute("agent.has_rag_context", bool(rag_context))
        span.set_attribute("agent.authenticated", bool(acting_user))

        llm = get_llm()
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        messages = list(state.get("messages", []))
        logger.info(
            "Running tool_calling_loop: %d tools, %d prior messages, "
            "rag_context=%s, domain_hint=%s",
            len(tools),
            len(messages),
            bool(rag_context),
            domain_hint,
        )

        # recursion_limit = 2 * max_tool_turns + 1; gives the LLM room for
        # roughly 10 tool turns before LangGraph hard-stops.
        result = await agent.ainvoke(
            {"messages": messages},
            {"recursion_limit": 25},
        )

        result_messages = result.get("messages", [])

        final_answer = ""
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

        span.set_attribute("agent.answer_length", len(final_answer))
        span.set_attribute("agent.tool_calls_made", len(tools_used))
        span.set_attribute("agent.tool_results_received", tool_result_count)

        logger.info(
            "tool_calling_loop complete: %d messages, %d tool calls, "
            "%d tool results, answer_len=%d",
            len(result_messages),
            len(tools_used),
            tool_result_count,
            len(final_answer),
        )

        return {
            "final_answer": final_answer,
            "messages": result_messages,
            "tools_used": tools_used,
            "node_trace": [
                {
                    "node": "tool_calling_loop",
                    "tool_count": len(tools),
                    "tool_calls_made": len(tools_used),
                    "tool_results": tool_result_count,
                    "answer_length": len(final_answer),
                }
            ],
        }
