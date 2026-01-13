"""Synthesize node - answer generation from tool results and RAG matches.

This node takes the tool execution results and/or RAG matches and generates
a natural language answer for the user. For combined queries, it merges
verified Q&A knowledge with real-time tool data.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate

from ...llm import get_llm
from ..state import AgentState, RAGMatch, ToolResult

logger = logging.getLogger(__name__)

# System prompt for answer synthesis (tools only)
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

# System prompt for combined synthesis (RAG + tools)
COMBINED_SYNTHESIS_PROMPT = """You are an ACCESS-CI documentation assistant. Your job is to answer user questions by combining verified documentation knowledge with real-time data from ACCESS tools.

## GUIDELINES

1. Be concise and direct - answer the question first, then provide details
2. Use BOTH the verified knowledge AND tool results to give comprehensive answers
3. The "VERIFIED KNOWLEDGE" section contains accurate, human-verified information about ACCESS resources
4. The "REAL-TIME DATA" section contains current data retrieved from live systems
5. When information appears in both sources, prefer the real-time data for status/availability but use verified knowledge for specifications/capabilities
6. Format data clearly - use bullet points, tables, or lists where appropriate
7. Include relevant links when available
8. Don't make up information not present in either source

## VERIFIED KNOWLEDGE (from ACCESS documentation)

{rag_context}

## REAL-TIME DATA (from ACCESS tools)

{tool_results}

## ANSWER FORMAT

Respond naturally as a helpful documentation assistant. Seamlessly combine verified documentation with real-time data. Do not mention "verified knowledge", "tool results", or internal system details - just answer the question as if you know this information."""

# System prompt for RAG-only synthesis (when tools failed but RAG has data)
RAG_ONLY_SYNTHESIS_PROMPT = """You are an ACCESS-CI documentation assistant. Your job is to answer user questions using verified documentation knowledge.

## GUIDELINES

1. Be concise and direct - answer the question first, then provide details
2. Use the verified knowledge provided to give accurate information
3. This information comes from human-verified ACCESS documentation
4. Format data clearly - use bullet points, tables, or lists where appropriate
5. Include relevant links when available
6. Don't make up information not present in the verified knowledge
7. If the knowledge doesn't fully answer the question, acknowledge what's missing

## VERIFIED KNOWLEDGE (from ACCESS documentation)

{rag_context}

## ANSWER FORMAT

Respond naturally as a helpful documentation assistant. Do not mention "verified knowledge" or internal system details - just answer the question as if you know this information."""


async def synthesize_node(state: AgentState) -> dict[str, Any]:
    """Generate a natural language answer from tool results and/or RAG matches.

    This node:
    1. Checks for RAG matches (verified Q&A knowledge)
    2. Checks for tool results (real-time data)
    3. Uses appropriate synthesis strategy based on available data:
       - RAG + tools: Combined synthesis for comprehensive answers
       - Tools only: Standard tool synthesis
       - RAG only: Use verified knowledge when tools failed
    4. Handles cases with no data available

    Args:
        state: Current agent state with query, tool_results, and rag_matches.

    Returns:
        Dict with final_answer and messages.
    """
    query = state["query"]
    tool_results = state.get("tool_results", [])
    rag_matches = state.get("rag_matches", [])
    query_analysis = state.get("query_analysis")

    # Handle no-tools-needed case
    if query_analysis and not query_analysis.requires_tools:
        return await _synthesize_without_tools(query, query_analysis)

    # Determine what data we have
    has_rag = bool(rag_matches)
    has_tools = bool(tool_results)
    tools_succeeded = has_tools and any(r.success for r in tool_results)

    logger.info(
        f"Synthesis: has_rag={has_rag} ({len(rag_matches)} matches), "
        f"has_tools={has_tools}, tools_succeeded={tools_succeeded}"
    )

    # No data at all
    if not has_rag and not has_tools:
        logger.warning("No results to synthesize")
        return {
            "final_answer": (
                "I wasn't able to retrieve specific information for your question. "
                "Please try rephrasing your query or ask about a specific ACCESS resource."
            ),
        }

    # Format available data
    rag_context = _format_rag_matches(rag_matches) if has_rag else ""
    results_text = _format_tool_results(tool_results) if has_tools else ""

    # All tools failed but we have RAG data - use RAG only
    if has_rag and has_tools and not tools_succeeded:
        logger.info("Tools failed but RAG has data - using RAG-only synthesis")
        return await _synthesize_with_rag_only(query, rag_context)

    # All tools failed and no RAG data
    if has_tools and not tools_succeeded and not has_rag:
        logger.warning("All tools failed and no RAG data")
        failed_tools = ", ".join(r.tool_name for r in tool_results)
        return {
            "final_answer": (
                "I encountered issues retrieving data for your question. "
                f"The following tools were attempted but failed: {failed_tools}. "
                "Please try again later or contact support if this persists."
            ),
        }

    # Choose synthesis strategy
    if has_rag and tools_succeeded:
        # Combined synthesis: RAG + tools
        return await _synthesize_combined(query, rag_context, results_text)
    if tools_succeeded:
        # Tools only
        return await _synthesize_tools_only(query, results_text)
    if has_rag:
        # RAG only (tools not attempted or empty)
        return await _synthesize_with_rag_only(query, rag_context)

    # Fallback (shouldn't reach here)
    return {
        "final_answer": (
            "I wasn't able to generate a complete answer. " "Please try rephrasing your question."
        ),
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


async def _synthesize_combined(
    query: str,
    rag_context: str,
    tool_results: str,
) -> dict[str, Any]:
    """Synthesize answer from both RAG matches and tool results.

    Args:
        query: The user's query.
        rag_context: Formatted RAG matches.
        tool_results: Formatted tool results.

    Returns:
        Dict with final_answer and messages.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COMBINED_SYNTHESIS_PROMPT),
            ("human", "{query}"),
        ]
    )

    llm = get_llm(temperature=0.3, max_tokens=2000)

    try:
        response = await llm.ainvoke(
            prompt.format_messages(
                rag_context=rag_context,
                tool_results=tool_results,
                query=query,
            )
        )
        answer = response.content

        logger.info(f"Generated combined answer: {len(answer)} characters")

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    except Exception as e:
        logger.error(f"Combined synthesis failed: {e}")
        return {
            "final_answer": ("I encountered an error generating your answer. " "Please try again."),
            "messages": [AIMessage(content="Error generating response.")],
        }


async def _synthesize_tools_only(
    query: str,
    tool_results: str,
) -> dict[str, Any]:
    """Synthesize answer from tool results only.

    Args:
        query: The user's query.
        tool_results: Formatted tool results.

    Returns:
        Dict with final_answer and messages.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYNTHESIS_SYSTEM_PROMPT),
            ("human", "{query}"),
        ]
    )

    llm = get_llm(temperature=0.3, max_tokens=2000)

    try:
        response = await llm.ainvoke(
            prompt.format_messages(
                tool_results=tool_results,
                query=query,
            )
        )
        answer = response.content

        logger.info(f"Generated tools-only answer: {len(answer)} characters")

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    except Exception as e:
        logger.error(f"Tools-only synthesis failed: {e}")
        return {
            "final_answer": ("I encountered an error generating your answer. " "Please try again."),
            "messages": [AIMessage(content="Error generating response.")],
        }


async def _synthesize_with_rag_only(
    query: str,
    rag_context: str,
) -> dict[str, Any]:
    """Synthesize answer from RAG matches only (when tools failed).

    Args:
        query: The user's query.
        rag_context: Formatted RAG matches.

    Returns:
        Dict with final_answer and messages.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RAG_ONLY_SYNTHESIS_PROMPT),
            ("human", "{query}"),
        ]
    )

    llm = get_llm(temperature=0.3, max_tokens=2000)

    try:
        response = await llm.ainvoke(
            prompt.format_messages(
                rag_context=rag_context,
                query=query,
            )
        )
        answer = response.content

        logger.info(f"Generated RAG-only answer: {len(answer)} characters")

        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
        }

    except Exception as e:
        logger.error(f"RAG-only synthesis failed: {e}")
        return {
            "final_answer": ("I encountered an error generating your answer. " "Please try again."),
            "messages": [AIMessage(content="Error generating response.")],
        }


def _format_rag_matches(matches: list[RAGMatch]) -> str:
    """Format RAG matches for the synthesis prompt.

    Args:
        matches: List of RAG matches from the Q&A service.

    Returns:
        Formatted string for the LLM prompt.
    """
    if not matches:
        return "(no verified knowledge available)"

    sections = []
    for match in matches:
        section = (
            f"### Q: {match.question}\n"
            f"A: {match.answer}\n"
            f"Source: {match.domain}/{match.entity_id} "
            f"(similarity: {match.similarity_score:.2f})"
        )
        sections.append(section)

    return "\n\n".join(sections)


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
