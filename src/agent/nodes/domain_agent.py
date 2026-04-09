"""Domain agent node — runs a react agent for domain-specific tasks.

Routes domain queries (announcements, JSM, etc.) to a specialized react agent
that has direct access to that domain's MCP tools and a domain-specific prompt.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.prebuilt import create_react_agent

from ...llm import get_llm
from ...telemetry import get_tracer
from ..domains.registry import get_domain_registry
from ..domains.tools import create_domain_tools
from ..state import AgentState

logger = logging.getLogger(__name__)


async def domain_agent_node(state: AgentState) -> dict[str, Any]:
    """Run a domain-specific react agent.

    1. Reads classification.domain to determine which domain
    2. Looks up domain config from registry
    3. Creates LangChain tool wrappers for that domain's MCP tools
    4. Runs a create_react_agent loop with the domain's system prompt
    5. Returns final answer and messages

    Args:
        state: Current agent state with query_classification.domain set.

    Returns:
        Dict with final_answer, messages, and tools_used.
    """
    tracer = get_tracer("access-agent.nodes")
    classification = state.get("query_classification")
    domain_name = classification.domain if classification else None

    writer = get_stream_writer()
    if domain_name:
        writer({"type": "status", "message": f"Starting {domain_name} workflow..."})

    if not domain_name:
        logger.error("domain_agent_node called without domain in classification")
        error_msg = "I'm sorry, I wasn't able to process that request. Please try again."
        return {
            "final_answer": error_msg,
            "messages": [AIMessage(content=error_msg)],
            "node_trace": [{"node": "domain_agent", "error": "no_domain"}],
        }

    with tracer.start_as_current_span(
        "agent.domain_agent",
        attributes={
            "agent.node": "domain_agent",
            "agent.domain": domain_name,
        },
    ) as span:
        # Defense-in-depth: route_after_rag should already have filtered
        # disabled domains. If routing is bypassed somehow, fail closed
        # with a user-visible message rather than an empty response so
        # the user gets something actionable.
        from ..domains.capabilities import get_capability_registry

        if not get_capability_registry().is_domain_enabled(domain_name):
            logger.warning(
                "domain_agent_node invoked for disabled domain '%s' "
                "(router should have prevented this)",
                domain_name,
            )
            user_msg = (
                "That workflow isn't available right now. "
                "You can open a support ticket at https://support.access-ci.org/open-a-ticket "
                "for direct assistance."
            )
            return {
                "final_answer": user_msg,
                "messages": [AIMessage(content=user_msg)],
                "node_trace": [
                    {
                        "node": "domain_agent",
                        "domain": domain_name,
                        "error": "domain_disabled",
                    }
                ],
            }

        # Look up domain config
        registry = get_domain_registry()
        config = registry.get(domain_name)

        if not config:
            logger.error(f"Unknown domain: {domain_name}")
            error_msg = "I'm sorry, I wasn't able to process that request. Please try again."
            return {
                "final_answer": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "node_trace": [
                    {"node": "domain_agent", "domain": domain_name, "error": "unknown_domain"}
                ],
            }

        # Create domain tools from catalog
        tool_catalog = state.get("tool_catalog", {})
        acting_user = state.get("acting_user")
        tools = create_domain_tools(config, tool_catalog, acting_user)

        if not tools:
            logger.warning(
                f"No tools available for domain {domain_name}, using UKY context if available"
            )
            # Fall back to UKY content from rag_answer (which now always runs first)
            rag_matches = state.get("rag_matches", [])
            if rag_matches:
                uky_answer = rag_matches[0].answer
                logger.info(f"Domain agent falling back to UKY answer ({len(uky_answer)} chars)")
                return {
                    "final_answer": uky_answer,
                    "messages": [AIMessage(content=uky_answer)],
                    "tools_used": ["uky_rag_retrieval"],
                    "node_trace": [
                        {
                            "node": "domain_agent",
                            "domain": domain_name,
                            "fallback": "uky_rag",
                            "answer_length": len(uky_answer),
                        }
                    ],
                }
            # No UKY content either — genuine dead end
            error_msg = (
                "I'm sorry, I don't have enough information to help with that right now. "
                "You can open a support ticket at https://support.access-ci.org/open-a-ticket "
                "for direct assistance."
            )
            return {
                "final_answer": error_msg,
                "messages": [AIMessage(content=error_msg)],
                "node_trace": [
                    {"node": "domain_agent", "domain": domain_name, "error": "no_tools_no_rag"}
                ],
            }

        span.set_attribute("agent.domain_tools", len(tools))

        # Format system prompt with acting user
        system_prompt = config.system_prompt.format(
            acting_user=acting_user or "anonymous (not logged in)",
        )

        # Create LLM for this domain
        llm = get_llm(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=2000,
        )

        # Create and run react agent
        react_graph = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        # Pass conversation messages to the react agent
        messages = list(state.get("messages", []))

        logger.info(
            f"Running domain agent '{domain_name}' with {len(tools)} tools, "
            f"{len(messages)} messages"
        )

        result = await react_graph.ainvoke(
            {"messages": messages},
            {"recursion_limit": config.max_iterations * 2 + 1},
        )

        # Extract the final AI message from the react agent's output
        react_messages = result.get("messages", [])
        final_message = ""
        if react_messages:
            # The last message from the react agent is the final answer
            last_msg = react_messages[-1]
            if hasattr(last_msg, "content"):
                final_message = last_msg.content

        # Detect whether the domain agent completed an action (called a tool)
        # or is still gathering info (only produced text, no tool calls).
        # If tools were called in this turn, the agent took action — the response
        # is final. If no tools were called, the agent is asking a clarifying
        # question — the response is not final.
        tool_was_called = any(isinstance(m, ToolMessage) for m in react_messages)

        span.set_attribute("agent.answer_length", len(final_message))
        span.set_attribute("agent.react_messages", len(react_messages))
        span.set_attribute("agent.domain_completed", tool_was_called)

        logger.info(
            f"Domain agent '{domain_name}' complete: "
            f"{len(react_messages)} messages, answer={len(final_message)} chars, "
            f"tool_called={tool_was_called}"
        )

        return {
            "final_answer": final_message,
            "messages": react_messages,
            "tools_used": [domain_name],
            "domain_completed": tool_was_called,
            "node_trace": [
                {
                    "node": "domain_agent",
                    "domain": domain_name,
                    "tool_count": len(tools),
                    "domain_completed": tool_was_called,
                }
            ],
        }
