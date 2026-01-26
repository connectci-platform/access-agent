"""RAG answer node.

Retrieves verified answers from the Q&A service for static and combined queries.
This is the RAG-primary architecture where verified Q&A pairs are the first
source of truth for factual knowledge.

Flow for static queries:
1. Query the access-qa-service for semantic matches
2. If high confidence match found: return verified answer → END
3. If no match: fall through to tools

Flow for combined queries:
1. Query the access-qa-service for semantic matches
2. Store matches in state (rag_matches) for synthesis
3. Continue to tools for real-time data
4. Synthesize combines RAG knowledge + tool results
"""

import logging
import re

from langchain_core.messages import AIMessage

from ...config import settings
from ...services.qa_client import get_qa_client
from ...telemetry import get_tracer
from ..state import AgentState, RAGMatch

logger = logging.getLogger(__name__)

# Citation pattern: <<SRC:domain:entity_id>>
CITATION_PATTERN = re.compile(r"<<SRC:([^:>]+):([^>]+)>>")


def process_citations(text: str) -> str:
    """Convert <<SRC:...>> markers to readable citations.

    The Q&A pairs include citations like:
        <<SRC:compute-resources:delta.ncsa.access-ci.org>>

    This converts them to a more readable format.

    Args:
        text: Response text with citation markers.

    Returns:
        Text with processed citations.
    """

    def replace_citation(match: re.Match[str]) -> str:
        domain = match.group(1)
        entity_id = match.group(2)
        return f"[Source: {domain}/{entity_id}]"

    return CITATION_PATTERN.sub(replace_citation, text)


def _get_threshold_for_query_type(query_type: str) -> float:
    """Get the appropriate similarity threshold based on query type.

    Args:
        query_type: The classification (static, combined, dynamic).

    Returns:
        The similarity threshold to use.
    """
    if query_type == "static":
        return settings.RAG_THRESHOLD_STATIC
    if query_type == "combined":
        return settings.RAG_THRESHOLD_COMBINED
    return settings.RAG_THRESHOLD_FALLBACK


async def rag_answer_node(state: AgentState) -> dict[str, object]:
    """Retrieve answer from Q&A service via semantic search.

    For static queries: Returns verified answer if confident match found.
    For combined queries: Stores matches for synthesis with tool results.

    Args:
        state: Current agent state with query.

    Returns:
        State update with:
        - For static with match: final_answer, messages, tools_used, rag_matches, rag_used
        - For combined: rag_matches, rag_used (no final_answer - continues to tools)
        - For no match: query_classification updated
    """
    tracer = get_tracer("access-agent.nodes")
    query = state["query"]
    classification = state["query_classification"]
    query_type = classification.query_type if classification else "static"

    # Use expanded_query from classification (has pronouns/references resolved)
    search_query = (
        classification.expanded_query if classification and classification.expanded_query else query
    )

    with tracer.start_as_current_span(
        "agent.rag_answer",
        attributes={
            "agent.node": "rag_answer",
            "agent.query_type": query_type,
            "agent.query_length": len(query),
            "agent.query_expanded": search_query != query,
        },
    ) as span:
        logger.info(f"RAG lookup for {query_type} query: {search_query[:50]}...")

        # Check if QA service is configured
        client = get_qa_client()
        if not client.is_configured:
            logger.warning("QA service not configured, falling back to tools")
            span.set_attribute("rag.configured", False)
            return {"rag_matches": [], "rag_used": False}

        span.set_attribute("rag.configured", True)

        # Get appropriate threshold for query type
        threshold = _get_threshold_for_query_type(query_type)
        span.set_attribute("rag.threshold", threshold)

        try:
            # Search for matching Q&A pairs using expanded query
            matches = await client.search(
                query=search_query,
                limit=settings.RAG_TOP_K,
                threshold=threshold,
            )

            # Convert to RAGMatch objects for state
            rag_matches = [
                RAGMatch(
                    id=m.id,
                    question=m.question,
                    answer=m.answer,
                    domain=m.domain,
                    entity_id=m.entity_id,
                    similarity_score=m.similarity_score,
                    metadata=m.metadata,
                )
                for m in matches
            ]

            # Record match statistics
            span.set_attribute("rag.matches_found", len(rag_matches))
            if rag_matches:
                span.set_attribute("rag.best_score", rag_matches[0].similarity_score)

            if rag_matches:
                best_match = rag_matches[0]
                logger.info(
                    f"RAG found {len(rag_matches)} matches. Best: "
                    f"similarity={best_match.similarity_score:.3f}, "
                    f"question='{best_match.question[:50]}...'"
                )

                # For static queries with confident match, return final answer
                if query_type == "static" and best_match.similarity_score >= threshold:
                    answer = process_citations(best_match.answer)
                    span.set_attribute("rag.result", "direct_answer")
                    span.set_attribute("rag.answer_length", len(answer))
                    return {
                        "final_answer": answer,
                        "messages": [AIMessage(content=answer)],
                        "tools_used": ["rag_retrieval"],
                        "rag_matches": rag_matches,
                        "rag_used": True,
                    }

                # For combined queries, store matches and continue to tools
                if query_type == "combined":
                    logger.info(
                        f"Combined query: Storing {len(rag_matches)} RAG matches for synthesis"
                    )
                    span.set_attribute("rag.result", "matches_for_synthesis")
                    return {
                        "rag_matches": rag_matches,
                        "rag_used": True,
                    }

                # Static query but below threshold - fall through
                logger.info(
                    f"Best match below threshold: {best_match.similarity_score:.3f} < {threshold}"
                )
                span.set_attribute("rag.result", "below_threshold")

            else:
                span.set_attribute("rag.result", "no_matches")

            # No confident match found
            logger.info("No RAG match found above threshold")

            # Return empty matches but preserve the classification
            # The graph routing will handle fallback to tools
            return {
                "rag_matches": rag_matches,  # May have low-confidence matches
                "rag_used": len(rag_matches) > 0,
            }

        except Exception as e:
            logger.error(f"RAG lookup failed: {e}")
            span.set_attribute("rag.result", "error")
            span.set_attribute("rag.error", str(e)[:200])
            return {"rag_matches": [], "rag_used": False}
