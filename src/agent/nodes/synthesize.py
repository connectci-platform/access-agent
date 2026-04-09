"""Synthesize node - answer generation from tool results and RAG matches.

This node takes the tool execution results and/or RAG matches and generates
a natural language answer for the user. For combined queries, it merges
verified Q&A knowledge with real-time tool data.

If tool results exceed the configured token budget, they are condensed
using an intermediate LLM call to extract query-relevant information.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.config import get_stream_writer

from ...config import settings
from ...llm import get_llm
from ...telemetry import get_tracer
from ..state import AgentState, RAGMatch, ToolResult

logger = logging.getLogger(__name__)

# Lazy-loaded capability summaries for system prompts (auth vs anon)
_capabilities_text_auth: str | None = None
_capabilities_text_anon: str | None = None


def _get_capabilities_text(authenticated: bool = False) -> str:
    """Get the capability summary for injection into synthesis prompts.

    Built once per auth level on first call from the CapabilityRegistry.
    Authenticated users see all capabilities; anonymous users see only
    public ones (matching the filtering in get_system_prompt_section).
    """
    global _capabilities_text_auth, _capabilities_text_anon
    if authenticated:
        if _capabilities_text_auth is None:
            try:
                from ..domains.capabilities import get_capability_registry

                registry = get_capability_registry()
                _capabilities_text_auth = registry.get_system_prompt_section(authenticated=True)
            except Exception:
                logger.warning("Could not load capability summary for synthesis prompt")
                _capabilities_text_auth = ""
        return _capabilities_text_auth
    if _capabilities_text_anon is None:
        try:
            from ..domains.capabilities import get_capability_registry

            registry = get_capability_registry()
            _capabilities_text_anon = registry.get_system_prompt_section(authenticated=False)
        except Exception:
            logger.warning("Could not load capability summary for synthesis prompt")
            _capabilities_text_anon = ""
    return _capabilities_text_anon


# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4

# System prompt for answer synthesis (tools only)
SYNTHESIS_SYSTEM_PROMPT = """You are an ACCESS-CI documentation assistant.

## TOOL RESULTS

{tool_results}

## INSTRUCTIONS

- Use the tool results above to answer the user's question.
- Be concise and direct — answer the question first, then provide details.
- Format data clearly using bullet points, tables, or lists where appropriate.
- Do NOT add information from your own training data. Only use what is provided in the tool results above. If the results don't answer the question, say so honestly.
- NEVER generate specific dates, event titles, names, or numbers that do not appear verbatim in the tool results above. Before including any specific detail, verify it appears in the data above. If it doesn't, do not include it.
- If a tool returned no results, zero items, or an empty list, tell the user honestly — for example "I checked for upcoming webinars but none are currently scheduled" or "No current outages were found for Delta." Do NOT fill in details from your own knowledge when tools return empty.
- If a tool returned results for a different resource or category than the user asked about, clarify what was and wasn't found — for example "There are no current outages for Expanse. There is a planned outage for Anvil on [date from tool data]."
- Do not mention "tool results" or system internals.

URL PRESERVATION (MANDATORY):
- You MUST include every URL that appears in the tool results. Do not summarize, omit, or replace any URL.
- Before finalizing your answer, re-read the tool results and verify that every URL present appears in your answer.

- For issues needing human help: https://support.access-ci.org/help-ticket

{capabilities}"""

# System prompt for combined synthesis (RAG + tools)
COMBINED_SYNTHESIS_PROMPT = """You are an ACCESS-CI documentation assistant.

## VERIFIED KNOWLEDGE (from ACCESS documentation — authoritative)

{rag_context}

## REAL-TIME DATA (from live ACCESS APIs — current)

{tool_results}

## INSTRUCTIONS

- Combine verified knowledge with real-time data to produce the best possible answer.
- CRITICAL: Combine both sources intelligently. Neither is always right:
  - Prefer REAL-TIME DATA for: hardware specs, software versions, system status, availability, current events, affinity group listings, NSF award data, usage metrics, and any enumerated/listed items — this data is live and more current than documentation.
  - Prefer VERIFIED KNOWLEDGE for: procedures, policies, how-to guides, troubleshooting steps, and contact information — documentation is more reliable for these.
  - When real-time data CONTRADICTS verified knowledge (e.g., verified knowledge says "no information available" but real-time data has specific results), TRUST THE REAL-TIME DATA — it means the documentation was incomplete or out of date.
- If the verified knowledge starts with hedging language like "The provided documents do not contain..." — ignore that preamble and use the substantive content that follows.
- Be concise and direct — answer the question first, then provide details.
- Format data clearly using bullet points, tables, or lists where appropriate.

URL PRESERVATION (MANDATORY):
- You MUST include every URL that appears in the verified knowledge. Do not summarize, omit, or replace any URL.
- You MUST include every URL that appears in the real-time data.
- Before finalizing your answer, re-read the verified knowledge and real-time data sections and verify that every URL present in either section appears in your answer. If any URL is missing, add it.
- This includes support ticket links, documentation links, user guide links, and any other URLs.

- Do NOT add information from your own training data. Only use what is provided in the verified knowledge and real-time data sections above.
- NEVER generate specific dates, event titles, names, or numbers that do not appear verbatim in the verified knowledge or real-time data above. Before including any specific detail, verify it appears in one of those sections. If it doesn't, do not include it.
- If real-time data returned no results, zero items, or an empty list, that is the authoritative answer — say so honestly (e.g., "No upcoming webinars are currently scheduled"). Do NOT use stale information from verified knowledge to fill in what the real-time data says is empty. The real-time data is more current.
- If real-time data returned results for a different resource or category than the user asked about, clarify what was and wasn't found — for example "There are no current outages for Expanse. There is a planned outage for Anvil on [date from real-time data]."
- Do not mention "verified knowledge", "tool results", or system internals.
- For issues needing human help: https://support.access-ci.org/help-ticket

{capabilities}"""

# System prompt for RAG-only synthesis (when tools failed but RAG has data)
RAG_ONLY_SYNTHESIS_PROMPT = """You are an ACCESS-CI documentation assistant. Your job is to answer user questions using verified documentation knowledge.

## GUIDELINES

1. Be concise and direct — answer the question first, then provide details
2. Use the verified knowledge provided to give accurate information
3. This information comes from human-verified ACCESS documentation — it is authoritative
4. Format data clearly — use bullet points, tables, or lists where appropriate
5. Do NOT add information from your own training data. Only use what is provided below.
6. NEVER generate specific dates, event titles, names, or numbers that do not appear verbatim in the verified knowledge below. Before including any specific detail, verify it appears in the data below. If it doesn't, do not include it.
7. If the knowledge starts with hedging language like "The provided documents do not contain..." — ignore that preamble and present the substantive content that follows
8. If the knowledge doesn't fully answer the question, acknowledge what's missing and suggest the user visit access-ci.org or open a support ticket

URL PRESERVATION (MANDATORY):
8. You MUST include every URL that appears in the verified knowledge below. Do not summarize, omit, or replace any URL.
9. Before finalizing your answer, re-read the verified knowledge and verify that every URL present appears in your answer. If any URL is missing, add it.

## VERIFIED KNOWLEDGE (from ACCESS documentation)

{rag_context}

## ANSWER FORMAT

Respond naturally as a helpful documentation assistant. Do not mention "verified knowledge" or internal system details — just answer the question as if you know this information.

{capabilities}"""

# System prompt for condensing large tool results
CONDENSE_RESULTS_PROMPT = """You are a data extraction assistant. Your job is to extract information relevant to the user's question from large tool results.

## USER'S QUESTION

{query}

## TOOL RESULTS

{tool_results}

## TASK

Extract and summarize ONLY the information from the tool results that is relevant to answering the user's question. Be thorough - include all relevant details, versions, availability information, and any other specifics that would help answer the question.

Do NOT answer the question yourself. Just extract and organize the relevant data.

Output the relevant information in a clear, structured format."""


def _strip_hedge_preamble(answer: str) -> str:
    """Strip common UKY hedge preambles without LLM rewrite.

    UKY often starts with "The provided documents do not contain..." then
    gives useful content. This strips the hedge sentence(s) at the start,
    preserving everything after. Uses simple sentence splitting — no LLM call.

    Args:
        answer: The raw UKY answer.

    Returns:
        The answer with hedge preamble removed, or unchanged if no hedge found.
    """
    hedge_starts = [
        "the provided documents do not",
        "the provided documents does not",
        "the provided information does not",
        "the documents do not",
        "the information provided does not",
    ]

    lower = answer.lower()
    if not any(lower.startswith(h) for h in hedge_starts):
        return answer

    # Find the end of the first sentence (period followed by space or newline)
    # Strip the hedge sentence and any leading whitespace/newlines after it
    for i, char in enumerate(answer):
        if char == "." and i < len(answer) - 1 and answer[i + 1] in (" ", "\n"):
            rest = answer[i + 2 :].lstrip()
            if rest:
                # Capitalize first letter of remaining content
                return rest[0].upper() + rest[1:] if rest else answer
            break

    # If we couldn't find a clean split point, return unchanged
    return answer


def _estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a text string.

    Uses a simple character-based heuristic. For more accurate counts,
    consider using tiktoken, but this is sufficient for budget checks.

    Args:
        text: The text to estimate tokens for.

    Returns:
        Estimated token count.
    """
    return len(text) // CHARS_PER_TOKEN


async def _condense_tool_results(query: str, results_text: str) -> str:
    """Condense large tool results by extracting query-relevant information.

    Uses an LLM to extract only the information relevant to the user's
    question, reducing token count while preserving important details.

    Args:
        query: The user's original query.
        results_text: The formatted tool results text.

    Returns:
        Condensed results text with only relevant information.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONDENSE_RESULTS_PROMPT),
        ]
    )

    # Use a faster model for condensation if available, with higher token limit
    llm = get_llm(temperature=0, max_tokens=4000)

    try:
        response = await llm.ainvoke(
            prompt.format_messages(
                query=query,
                tool_results=results_text,
            )
        )
        condensed = str(response.content)

        original_tokens = _estimate_tokens(results_text)
        condensed_tokens = _estimate_tokens(condensed)
        logger.info(
            f"Condensed tool results: {original_tokens} -> {condensed_tokens} tokens "
            f"({100 * condensed_tokens // original_tokens}% of original)"
        )

        return condensed

    except Exception as e:
        logger.error(f"Failed to condense tool results: {e}")
        # Fall back to truncation as last resort
        max_chars = settings.SYNTHESIS_TOKEN_BUDGET * CHARS_PER_TOKEN
        return results_text[:max_chars] + "\n\n[Results truncated due to length]"


async def _maybe_condense_results(
    query: str,
    results_text: str,
    span: Any,
) -> str:
    """Check token budget and condense results if needed.

    Args:
        query: The user's query for context-aware condensation.
        results_text: The formatted tool results.
        span: The telemetry span for attributes.

    Returns:
        Original or condensed results text.
    """
    if not results_text:
        return results_text

    estimated_tokens = _estimate_tokens(results_text)
    span.set_attribute("synthesis.tool_results_tokens", estimated_tokens)

    if estimated_tokens > settings.SYNTHESIS_TOKEN_BUDGET:
        logger.warning(
            f"Tool results exceed token budget: {estimated_tokens} > "
            f"{settings.SYNTHESIS_TOKEN_BUDGET}. Condensing..."
        )
        span.set_attribute("synthesis.condensed", True)
        condensed = await _condense_tool_results(query, results_text)
        span.set_attribute("synthesis.condensed_tokens", _estimate_tokens(condensed))
        return condensed

    span.set_attribute("synthesis.condensed", False)
    return results_text


async def synthesize_node(state: AgentState) -> dict[str, Any]:  # noqa: PLR0912, PLR0915
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
    writer = get_stream_writer()
    writer({"type": "status", "message": "Generating answer..."})

    tracer = get_tracer("access-agent.nodes")
    query = state["query"]
    tool_results = state.get("tool_results", [])
    rag_matches = state.get("rag_matches", [])
    query_analysis = state.get("query_analysis")
    authenticated = state.get("acting_user") is not None

    with tracer.start_as_current_span(
        "agent.synthesize",
        attributes={
            "agent.node": "synthesize",
            "agent.query_length": len(query),
            "agent.rag_matches": len(rag_matches) if rag_matches else 0,
            "agent.tool_results": len(tool_results) if tool_results else 0,
        },
    ) as span:
        strategy = "unknown"
        result: dict[str, Any] = {}

        # Handle no-tools-needed case
        if query_analysis and not query_analysis.requires_tools:
            # If UKY provided content, serve it directly — no LLM rewrite.
            # This avoids synthesis dilution when tools have nothing to add.
            if rag_matches:
                strategy = "uky_direct_no_tools"
                span.set_attribute("synthesis.strategy", strategy)
                raw_answer = rag_matches[0].answer
                # Strip common hedge preambles without LLM rewrite
                answer = _strip_hedge_preamble(raw_answer)
                logger.info(
                    f"No tools needed, serving UKY answer directly "
                    f"({len(answer)} chars, stripped={len(raw_answer) != len(answer)})"
                )
                result = {
                    "final_answer": answer,
                    "messages": [AIMessage(content=answer)],
                    "tools_used": ["uky_rag_retrieval"],
                }
            else:
                strategy = "no_tools_needed"
                span.set_attribute("synthesis.strategy", strategy)
                result = await _synthesize_without_tools(query, query_analysis, rag_matches)
        else:
            # Determine what data we have
            has_rag = bool(rag_matches)
            has_tools = bool(tool_results)
            tools_succeeded = has_tools and any(r.success for r in tool_results)

            span.set_attribute("synthesis.has_rag", has_rag)
            span.set_attribute("synthesis.has_tools", has_tools)
            span.set_attribute("synthesis.tools_succeeded", tools_succeeded)

            logger.info(
                f"Synthesis: has_rag={has_rag} ({len(rag_matches)} matches), "
                f"has_tools={has_tools}, tools_succeeded={tools_succeeded}"
            )

            if not has_rag and not has_tools:
                strategy = "no_data"
                logger.warning("No results to synthesize")
                result = {
                    "final_answer": (
                        "I wasn't able to retrieve specific information for your question. "
                        "Please try rephrasing your query or ask about a specific ACCESS resource."
                    ),
                }
            else:
                # Format available data
                rag_context = _format_rag_matches(rag_matches) if has_rag else ""
                results_text = _format_tool_results(tool_results) if has_tools else ""
                results_text = await _maybe_condense_results(query, results_text, span)

                if has_rag and has_tools and not tools_succeeded:
                    strategy = "uky_direct_tools_failed"
                    raw_answer = rag_matches[0].answer
                    answer = _strip_hedge_preamble(raw_answer)
                    logger.info(f"Tools failed, serving UKY answer directly ({len(answer)} chars)")
                    result = {
                        "final_answer": answer,
                        "messages": [AIMessage(content=answer)],
                        "tools_used": ["uky_rag_retrieval"],
                    }
                elif has_tools and not tools_succeeded and not has_rag:
                    strategy = "all_failed"
                    failed_tools = ", ".join(r.tool_name for r in tool_results)
                    logger.warning(f"All tools failed and no RAG data: {failed_tools}")
                    result = {
                        "final_answer": (
                            "I wasn't able to retrieve specific information for your question. "
                            "Would you like me to help you open a support ticket? "
                            "You can also open one directly at "
                            "https://support.access-ci.org/help-ticket."
                        ),
                    }
                elif has_rag and tools_succeeded:
                    strategy = "combined"
                    result = await _synthesize_combined(
                        query, rag_context, results_text, authenticated=authenticated
                    )
                elif tools_succeeded:
                    strategy = "tools_only"
                    result = await _synthesize_tools_only(
                        query, results_text, authenticated=authenticated
                    )
                elif has_rag:
                    strategy = "uky_direct_only"
                    raw_answer = rag_matches[0].answer
                    answer = _strip_hedge_preamble(raw_answer)
                    logger.info(f"RAG only, serving UKY answer directly ({len(answer)} chars)")
                    result = {
                        "final_answer": answer,
                        "messages": [AIMessage(content=answer)],
                        "tools_used": ["uky_rag_retrieval"],
                    }
                else:
                    strategy = "fallback"
                    result = {
                        "final_answer": (
                            "I wasn't able to generate a complete answer. "
                            "Please try rephrasing your question."
                        ),
                    }

            span.set_attribute("synthesis.strategy", strategy)

        answer_len = len(result.get("final_answer", "") or "")
        if answer_len:
            span.set_attribute("synthesis.answer_length", answer_len)

        result["node_trace"] = [
            {
                "node": "synthesize",
                "strategy": strategy,
                "answer_length": answer_len,
            }
        ]
        return result


async def _synthesize_without_tools(
    query: str,
    query_analysis: Any,
    rag_matches: list[RAGMatch] | None = None,
) -> dict[str, Any]:
    """Generate an answer for queries that don't need tools.

    Args:
        query: The user's query.
        query_analysis: The analysis from the plan node.
        rag_matches: Optional RAG matches to fall back to if the LLM call fails.

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
        # Fall back to RAG content if available, rather than a generic deflection
        if rag_matches:
            answer = _strip_hedge_preamble(rag_matches[0].answer)
            logger.info(f"No-tools synthesis failed, falling back to RAG ({len(answer)} chars)")
            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
                "tools_used": ["uky_rag_retrieval"],
            }
        error_answer = (
            "I wasn't able to find specific information for your question. "
            "Would you like me to help you open a support ticket? "
            "You can also open one directly at https://support.access-ci.org/open-a-ticket."
        )
        return {
            "final_answer": error_answer,
            "messages": [AIMessage(content=error_answer)],
        }


async def _synthesize_combined(
    query: str,
    rag_context: str,
    tool_results: str,
    authenticated: bool = False,
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
                capabilities=_get_capabilities_text(authenticated),
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
            "final_answer": ("I encountered an error generating your answer. Please try again."),
            "messages": [AIMessage(content="Error generating response.")],
        }


async def _synthesize_tools_only(
    query: str,
    tool_results: str,
    authenticated: bool = False,
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
                capabilities=_get_capabilities_text(authenticated),
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
            "final_answer": ("I encountered an error generating your answer. Please try again."),
            "messages": [AIMessage(content="Error generating response.")],
        }


async def _synthesize_with_rag_only(
    query: str,
    rag_context: str,
    authenticated: bool = False,
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
                capabilities=_get_capabilities_text(authenticated),
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
            "final_answer": ("I encountered an error generating your answer. Please try again."),
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
