"""Domain agent node — runs a react agent for domain-specific tasks.

Routes domain queries (announcements, JSM, etc.) to a specialized react agent
that has direct access to that domain's MCP tools and a domain-specific prompt.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
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

    if not domain_name:
        logger.error("domain_agent_node called without domain in classification")
        error_msg = "I'm sorry, I wasn't able to process that request. Please try again."
        return {
            "final_answer": error_msg,
            "messages": [AIMessage(content=error_msg)],
        }

    with tracer.start_as_current_span(
        "agent.domain_agent",
        attributes={
            "agent.node": "domain_agent",
            "agent.domain": domain_name,
        },
    ) as span:
        # Look up domain config
        registry = get_domain_registry()
        config = registry.get(domain_name)

        if not config:
            logger.error(f"Unknown domain: {domain_name}")
            error_msg = "I'm sorry, I wasn't able to process that request. Please try again."
            return {
                "final_answer": error_msg,
                "messages": [AIMessage(content=error_msg)],
            }

        # Create domain tools from catalog
        tool_catalog = state.get("tool_catalog", {})
        acting_user = state.get("acting_user")
        tools = create_domain_tools(config, tool_catalog, acting_user)

        if not tools:
            logger.warning(f"No tools available for domain {domain_name}")
            error_msg = (
                "I'm sorry, the tools needed for this task are currently unavailable. "
                "Please try again later."
            )
            return {
                "final_answer": error_msg,
                "messages": [AIMessage(content=error_msg)],
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

        span.set_attribute("agent.answer_length", len(final_message))
        span.set_attribute("agent.react_messages", len(react_messages))

        logger.info(
            f"Domain agent '{domain_name}' complete: "
            f"{len(react_messages)} messages, answer={len(final_message)} chars"
        )

        return {
            "final_answer": final_message,
            "messages": react_messages,
            "tools_used": [domain_name],
        }
