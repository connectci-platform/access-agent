"""RAG answer node.

Retrieves answers from UKY RAG endpoints or the pgvector Q&A service.

Routing:
- If classification.rag_endpoint is set and UKY is enabled, query the
  appropriate UKY endpoint (general or xdmod).
- Falls back to pgvector Q&A service if UKY fails or isn't configured.

Flow for static queries:
1. Query UKY endpoint (or pgvector fallback) for an answer
2. If answer found: return as final_answer → END
3. If no answer: fall through to tools

Flow for combined queries:
1. Query UKY endpoint (or pgvector fallback)
2. Store answer as RAGMatch for synthesis with tool results
3. Continue to tools for real-time data
4. Synthesize combines RAG knowledge + tool results
"""

import logging
import re
from typing import Literal

from langchain_core.messages import AIMessage
from opentelemetry.trace import Span

from ...config import settings
from ...services.qa_client import get_qa_client
from ...services.uky_client import get_uky_client
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


async def _ask_uky(
    search_query: str,
    query_type: str,
    rag_endpoint: Literal["general", "xdmod"],
    session_id: str,
    question_id: str,
    span: Span,
) -> dict[str, object] | None:
    """Query a UKY RAG endpoint and return a state update.

    Args:
        search_query: The query to send.
        query_type: Classification type (static/combined).
        rag_endpoint: Which UKY endpoint ("general" or "xdmod").
        session_id: Session identifier.
        question_id: Question identifier.
        span: Active OpenTelemetry span.

    Returns:
        State update dict, or None if UKY should be skipped/failed.
    """
    client = get_uky_client()
    if not client.is_configured:
        logger.info("UKY RAG not configured, falling back to pgvector")
        return None

    try:
        uky_response = await client.ask(
            query=search_query,
            endpoint_type=rag_endpoint,
            session_id=session_id,
            question_id=question_id,
        )

        if not uky_response.response:
            logger.info(f"UKY RAG ({rag_endpoint}) returned empty response")
            return None

        answer = uky_response.response
        span.set_attribute("rag.source", f"uky_{rag_endpoint}")
        span.set_attribute("rag.answer_length", len(answer))
        span.set_attribute("uky_rag.duration_ms", uky_response.duration_ms)

        # Wrap answer as a RAGMatch for downstream compatibility
        rag_match = RAGMatch(
            id=f"uky-{rag_endpoint}",
            question=search_query,
            answer=answer,
            domain=f"uky-{rag_endpoint}",
            entity_id=f"uky-{rag_endpoint}",
            similarity_score=1.0,  # UKY returns a complete answer, not a similarity score
        )

        # For static queries, the UKY answer is the final answer
        if query_type == "static":
            span.set_attribute("rag.result", "uky_direct_answer")
            return {
                "final_answer": answer,
                "messages": [AIMessage(content=answer)],
                "tools_used": ["uky_rag_retrieval"],
                "rag_matches": [rag_match],
                "rag_used": True,
            }

        # For combined queries, store the answer for synthesis with tool results
        logger.info(f"Combined query: Storing UKY {rag_endpoint} answer for synthesis")
        span.set_attribute("rag.result", "uky_for_synthesis")
        return {
            "rag_matches": [rag_match],
            "rag_used": True,
        }

    except Exception as e:
        logger.warning(f"UKY RAG ({rag_endpoint}) failed, falling back to pgvector: {e}")
        span.set_attribute("uky_rag.error", str(e)[:200])
        return None


async def _search_pgvector(
    search_query: str,
    query_type: str,
    span: Span,
) -> dict[str, object]:
    """Search pgvector Q&A service (original logic, used as fallback).

    Args:
        search_query: The query to search.
        query_type: Classification type (static/combined).
        span: Active OpenTelemetry span.

    Returns:
        State update dict.
    """
    client = get_qa_client()
    if not client.is_configured:
        logger.warning("QA service not configured, falling back to tools")
        span.set_attribute("rag.configured", False)
        return {"rag_matches": [], "rag_used": False}

    span.set_attribute("rag.configured", True)
    span.set_attribute("rag.source", "pgvector")

    threshold = _get_threshold_for_query_type(query_type)
    span.set_attribute("rag.threshold", threshold)

    try:
        matches = await client.search(
            query=search_query,
            limit=settings.RAG_TOP_K,
            threshold=threshold,
        )

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

            if query_type == "combined":
                logger.info(f"Combined query: Storing {len(rag_matches)} RAG matches for synthesis")
                span.set_attribute("rag.result", "matches_for_synthesis")
                return {
                    "rag_matches": rag_matches,
                    "rag_used": True,
                }

            logger.info(
                f"Best match below threshold: {best_match.similarity_score:.3f} < {threshold}"
            )
            span.set_attribute("rag.result", "below_threshold")

        else:
            span.set_attribute("rag.result", "no_matches")

        logger.info("No RAG match found above threshold")
        return {
            "rag_matches": rag_matches,
            "rag_used": len(rag_matches) > 0,
        }

    except Exception as e:
        logger.error(f"RAG lookup failed: {e}")
        span.set_attribute("rag.result", "error")
        span.set_attribute("rag.error", str(e)[:200])
        return {"rag_matches": [], "rag_used": False}


async def rag_answer_node(state: AgentState) -> dict[str, object]:
    """Retrieve answer from UKY RAG endpoints or pgvector Q&A service.

    Routes to UKY endpoints when classification.rag_endpoint is set,
    falling back to pgvector if UKY fails or isn't configured.

    Args:
        state: Current agent state with query.

    Returns:
        State update with:
        - For static with match: final_answer, messages, tools_used, rag_matches, rag_used
        - For combined: rag_matches, rag_used (no final_answer - continues to tools)
        - For no match: rag_matches, rag_used=False
    """
    tracer = get_tracer("access-agent.nodes")
    query = state["query"]
    classification = state["query_classification"]
    query_type = classification.query_type if classification else "static"
    rag_endpoint = classification.rag_endpoint if classification else None

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
            "agent.rag_endpoint": rag_endpoint or "none",
        },
    ) as span:
        logger.info(
            f"RAG lookup for {query_type} query (endpoint={rag_endpoint}): {search_query[:50]}..."
        )

        # Try UKY endpoint first if configured
        if rag_endpoint:
            result = await _ask_uky(
                search_query=search_query,
                query_type=query_type,
                rag_endpoint=rag_endpoint,
                session_id=state.get("session_id", ""),
                question_id=state.get("question_id", ""),
                span=span,
            )
            if result is None:
                logger.info("Falling back to pgvector after UKY failure")
                result = await _search_pgvector(search_query, query_type, span)
        else:
            # pgvector primary (no rag_endpoint set)
            result = await _search_pgvector(search_query, query_type, span)

        rag_matches = result.get("rag_matches", [])
        best_score = rag_matches[0].similarity_score if rag_matches else None
        result["node_trace"] = [{
            "node": "rag_answer",
            "source": "uky" if result.get("tools_used") == ["uky_rag_retrieval"] else "pgvector",
            "match_count": len(rag_matches),
            "best_score": best_score,
            "rag_used": result.get("rag_used", False),
            "has_final_answer": "final_answer" in result,
        }]
        return result
