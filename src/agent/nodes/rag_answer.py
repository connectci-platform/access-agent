"""RAG answer node.

Retrieves answers from UKY document RAG endpoints.

Routing:
- If classification.rag_endpoint is set, query the appropriate UKY
  endpoint (general or xdmod).
- If UKY fails or isn't configured, return empty result (routes to plan).

Flow for static queries:
1. Query UKY endpoint for an answer
2. If answer found: return as final_answer → END
3. If no answer: fall through to tools

Flow for combined queries:
1. Query UKY endpoint
2. Store answer as RAGMatch for synthesis with tool results
3. Continue to tools for real-time data
4. Synthesize combines RAG knowledge + tool results
"""

import logging
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
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


def _rag_response_out_of_scope(result: dict[str, Any]) -> bool:
    """Heuristic: check if a scoped RAG response indicates out-of-scope.

    Uses the UKY `in_scope` field when available; otherwise falls back to
    text patterns in the response. Will be simplified once UKY ships the
    `in_scope` boolean.
    """
    # Check in_scope from UKYResponse stored as RAGMatch
    rag_matches = result.get("rag_matches", [])
    # in_scope is not on RAGMatch — check the final_answer text instead
    answer = result.get("final_answer", "")
    if not answer:
        # No final answer (combined query) — check the RAG match answer
        if rag_matches:
            answer = rag_matches[0].answer if hasattr(rag_matches[0], "answer") else ""

    if not answer:
        return False

    lower = answer.lower()
    out_of_scope_phrases = [
        "outside the scope",
        "outside of the scope",
        "don't have information about that",
        "do not have information about that",
        "not related to",
        "i can only answer questions about",
        "i can only help with",
        "no documents are currently available",
        "documents may not have been embedded",
    ]
    return any(phrase in lower for phrase in out_of_scope_phrases)


async def _ask_uky(
    search_query: str,
    query_type: str,
    rag_endpoint: Literal["general", "xdmod"],
    session_id: str,
    question_id: str,
    span: Span,
    rp_name: str | None = None,
) -> dict[str, Any] | None:
    """Query a UKY RAG endpoint and return a state update.

    Args:
        search_query: The query to send.
        query_type: Classification type (static/combined).
        rag_endpoint: Which UKY endpoint ("general" or "xdmod").
        session_id: Session identifier.
        question_id: Question identifier.
        span: Active OpenTelemetry span.
        rp_name: RP slug for resource-scoped queries (e.g. 'delta').

    Returns:
        State update dict, or None if UKY should be skipped/failed.
    """
    client = get_uky_client()
    if not client.is_configured:
        logger.info("UKY RAG not configured, returning empty result")
        return None

    try:
        uky_response = await client.ask(
            query=search_query,
            endpoint_type=rag_endpoint,
            session_id=session_id,
            question_id=question_id,
            rp_name=rp_name,
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

        # For combined/dynamic queries, store the answer for synthesis with tool results
        logger.info(f"{query_type.title()} query: Storing UKY {rag_endpoint} answer for synthesis")
        span.set_attribute("rag.result", "uky_for_synthesis")
        return {
            "rag_matches": [rag_match],
            "rag_used": True,
        }

    except Exception as e:
        logger.warning(f"UKY RAG ({rag_endpoint}) failed: {e}")
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
    """Retrieve answer from UKY document RAG.

    Always consults UKY regardless of query classification. Defaults to
    the "general" endpoint when the classifier sets rag_endpoint=null
    (e.g., dynamic queries).

    Args:
        state: Current agent state with query.

    Returns:
        State update with:
        - For static with match: final_answer, messages, tools_used, rag_matches, rag_used
        - For combined: rag_matches, rag_used (no final_answer - continues to tools)
        - For no match: rag_matches, rag_used=False
    """
    writer = get_stream_writer()
    writer({"type": "status", "message": "Searching ACCESS documentation..."})

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

        # Query UKY document RAG — always consult UKY regardless of classification.
        # Default to "general" endpoint when classifier sets rag_endpoint=null
        # (e.g., dynamic queries). UKY often has useful context even for
        # questions the classifier thinks are purely dynamic.
        #
        # pgvector Q&A pair fallback disabled (2026-03-23) — pairs were too
        # narrow/thin vs UKY docs, and 0.85 threshold never matched real-user
        # input. To re-enable for slam-dunk scenarios: call _search_pgvector()
        # when result is None. The function and qa_client.py are intact.
        effective_endpoint = rag_endpoint or "general"
        resource_context = state.get("resource_context")
        result = await _ask_uky(
            search_query=search_query,
            query_type=query_type,
            rag_endpoint=effective_endpoint,
            session_id=state.get("session_id", ""),
            question_id=state.get("question_id", ""),
            span=span,
            rp_name=resource_context,
        )

        # Scoped-RAG fallback: if response looks out-of-scope, retry general
        if result is not None and resource_context and _rag_response_out_of_scope(result):
            logger.info(
                f"Scoped RAG for '{resource_context}' looks out-of-scope, retrying general"
            )
            span.set_attribute("rag.scoped_fallback", True)
            result = await _ask_uky(
                search_query=search_query,
                query_type=query_type,
                rag_endpoint=effective_endpoint,
                session_id=state.get("session_id", ""),
                question_id=state.get("question_id", ""),
                span=span,
                rp_name=None,
            )

        if result is None:
            logger.info("No UKY result — continuing with empty RAG context")
            result = {"rag_matches": [], "rag_used": False}

        rag_matches = result.get("rag_matches", [])
        best_score = rag_matches[0].similarity_score if rag_matches else None
        final_answer = result.get("final_answer", "")
        # Check for hedge phrases in the answer
        hedge_phrases = [
            "do not contain",
            "does not contain",
            "do not explicitly",
            "does not explicitly",
            "not provided in",
            "not mentioned in",
            "no specific information",
            "do not have specific information",
            "currently do not have",
            "not available in the provided",
        ]
        lower = final_answer.lower() if final_answer else ""
        hedge_detected = bool(final_answer and any(p in lower for p in hedge_phrases))
        # Even if hedge detected, answer may have substance (urls, length)
        has_substance = bool(
            final_answer and ("http" in lower or "@" in final_answer or len(final_answer) > 500)
        )
        result["node_trace"] = [
            {
                "node": "rag_answer",
                "source": "uky" if result.get("rag_used") else "none",
                "match_count": len(rag_matches),
                "best_score": best_score,
                "rag_used": result.get("rag_used", False),
                "has_final_answer": bool(final_answer),
                "hedge_detected": hedge_detected,
                "hedge_has_substance": has_substance if hedge_detected else None,
            }
        ]
        return result
