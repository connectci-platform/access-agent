"""Synthesize node - answer generation from tool results.

This node takes the tool execution results and generates a natural
language answer for the user.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from ...llm import get_llm
from ..state import AgentState, ToolResult

logger = logging.getLogger(__name__)

# System prompt for answer synthesis
SYNTHESIS_SYSTEM_PROMPT = """You are an ACCESS-CI documentation assistant. Your job is to answer user questions using data from ACCESS tools.

## GUIDELINES

1. Be concise and direct - answer the question first, then provide details
2. Use the tool results provided to give accurate, specific information
3. If tool results are empty or failed, acknowledge this honestly
4. Format data clearly - use bullet points, tables, or lists where appropriate
5. Include relevant links when available in the data
6. Don't make up information not present in the tool results
7. If you cannot answer due to missing data, suggest what the user might try

## TOOL RESULTS

{tool_results}

## ANSWER FORMAT

Respond naturally as a helpful documentation assistant. Do not mention "tool results" or internal system details - just answer the question as if you know this information."""


async def synthesize_node(state: AgentState) -> dict[str, Any]:
    """Generate a natural language answer from tool results.

    This node:
    1. Uses tool results directly
    2. Generates a helpful answer based on the results
    3. Handles cases with no tools or failed tools

    Args:
        state: Current agent state with query and tool_results.

    Returns:
        Dict with final_answer.
    """
    query = state["query"]
    tool_results = state.get("tool_results", [])
    query_analysis = state.get("query_analysis")

    # Handle no-tools-needed case
    if query_analysis and not query_analysis.requires_tools:
        return await _synthesize_without_tools(query, query_analysis)

    if not tool_results:
        logger.warning("No results to synthesize")
        return {
            "final_answer": (
                "I wasn't able to retrieve specific information for your question. "
                "Please try rephrasing your query or ask about a specific ACCESS resource."
            ),
        }

    results_text = _format_tool_results(tool_results)
    all_failed = all(not r.success for r in tool_results)

    if all_failed:
        logger.warning("All tools failed")
        failed_tools = ", ".join(r.tool_name for r in tool_results)
        return {
            "final_answer": (
                "I encountered issues retrieving data for your question. "
                f"The following tools were attempted but failed: {failed_tools}. "
                "Please try again later or contact support if this persists."
            ),
        }

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )

    # Get LLM and generate answer
    llm = get_llm(temperature=0.3, max_tokens=2000)

    try:
        response = await llm.ainvoke(
            prompt.format_messages(
                tool_results=results_text,
                query=query,
            )
        )
        answer = response.content

        logger.info(f"Generated answer: {len(answer)} characters")

        return {
            "final_answer": answer,
            # Add assistant response to messages for conversation memory
            "messages": [AIMessage(content=answer)],
        }

    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        error_answer = (
            "I encountered an error generating your answer. "
            "The tool data was retrieved successfully, but I couldn't format the response. "
            "Please try again."
        )
        return {
            "final_answer": error_answer,
            "messages": [AIMessage(content=error_answer)],
        }


async def _synthesize_without_tools(
    query: str,
    query_analysis: Any,
) -> dict[str, Any]:
    """Generate an answer for queries that don't need tools.

    Args:
        query: The user's query.
        query_analysis: The analysis from the plan node.

    Returns:
        Dict with final_answer.
    """
    # For no-tools queries, use general knowledge prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an ACCESS-CI documentation assistant. Answer the user's general question about ACCESS.

ACCESS (Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support) provides
researchers with allocations to high-performance computing resources.

Be helpful, accurate, and concise. If you're not certain about specific details,
suggest the user check the official ACCESS documentation at access-ci.org.""",
            ),
            ("human", "{query}"),
        ]
    )

    llm = get_llm(temperature=0.3, max_tokens=1500)

    try:
        response = await llm.ainvoke(prompt.format_messages(query=query))
        answer = response.content
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    except Exception as e:
        logger.error(f"No-tools synthesis failed: {e}")
        error_answer = (
            "I can help with that question. Please check the ACCESS documentation "
            "at https://access-ci.org for the most up-to-date information."
        )
        return {
            "final_answer": error_answer,
            "messages": [AIMessage(content=error_answer)],
        }


def _format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results for the synthesis prompt.

    Args:
        results: List of tool execution results.

    Returns:
        Formatted string for the LLM prompt.
    """
    sections = []

    for result in results:
        if result.success:
            # Format successful result
            data_str = _format_data(result.data)
            sections.append(f"### {result.tool_name}\nStatus: SUCCESS\nData:\n{data_str}")
        else:
            # Format failed result
            sections.append(f"### {result.tool_name}\nStatus: FAILED\nError: {result.error}")

    return "\n\n".join(sections)


def _format_data(data: Any) -> str:
    """Format data for the prompt.

    Args:
        data: The data to format.

    Returns:
        Formatted string.
    """
    import json

    if data is None:
        return "(no data)"

    if isinstance(data, str):
        return data

    try:
        return json.dumps(data, indent=2, default=str)
    except (TypeError, ValueError):
        return str(data)
