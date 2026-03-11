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

import asyncio
import logging
import re
import time
from typing import Literal

from langchain_core.messages import AIMessage
from opentelemetry.trace import Span

from ...config import settings
from ...rag_comparison_logger import get_rag_comparison_logger
from ...services.qa_client import get_qa_client
from .synthesize import _format_rag_matches, _synthesize_with_rag_only
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
                span.set_attribute("rag.result", "direct_match_for_synthesis")
                span.set_attribute("rag.answer_length", len(best_match.answer))
                return {
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


async def _query_uky_raw(
    search_query: str,
    rag_endpoint: Literal["general", "xdmod"],
    session_id: str,
    question_id: str,
) -> dict[str, object]:
    """Query UKY endpoint and return raw results (no span/state side-effects).

    Used by _dual_rag_answer for parallel comparison queries.
    """
    client = get_uky_client()
    if not client.is_configured:
        return {"response": None, "duration_ms": None, "error": "not_configured"}

    try:
        uky_response = await client.ask(
            query=search_query,
            endpoint_type=rag_endpoint,
            session_id=session_id,
            question_id=question_id,
        )
        return {
            "response": uky_response.response or None,
            "duration_ms": uky_response.duration_ms,
            "error": None,
        }
    except Exception as e:
        return {"response": None, "duration_ms": None, "error": str(e)[:500]}


async def _query_pgvector_raw(
    search_query: str,
    query_type: str,
) -> dict[str, object]:
    """Query pgvector and return raw results (no span/state side-effects).

    Used by _dual_rag_answer for parallel comparison queries.
    """
    client = get_qa_client()
    if not client.is_configured:
        return {"matches": [], "duration_ms": None, "error": "not_configured"}

    threshold = _get_threshold_for_query_type(query_type)

    try:
        start = time.monotonic()
        matches = await client.search(
            query=search_query,
            limit=settings.RAG_TOP_K,
            threshold=threshold,
        )
        duration_ms = (time.monotonic() - start) * 1000
        return {
            "matches": matches,
            "duration_ms": duration_ms,
            "error": None,
        }
    except Exception as e:
        return {"matches": [], "duration_ms": None, "error": str(e)[:500]}


async def _dual_rag_answer(
    search_query: str,
    query: str,
    query_type: str,
    rag_endpoint: Literal["general", "xdmod"],
    session_id: str,
    question_id: str,
    span: Span,
) -> dict[str, object]:
    """Query both UKY and pgvector in parallel, log comparison, return state update.

    Serves the user-facing answer with the same priority as the normal flow
    (UKY primary, pgvector fallback) but always queries both and logs the
    side-by-side results for A.3 evaluation.
    """
    # Run both queries concurrently
    uky_raw, pg_raw = await asyncio.gather(
        _query_uky_raw(search_query, rag_endpoint, session_id, question_id),
        _query_pgvector_raw(search_query, query_type),
    )

    # Build pgvector RAGMatch objects (needed for state update and logging)
    pg_matches_raw = pg_raw.get("matches", [])
    pg_rag_matches = [
        RAGMatch(
            id=m.id,
            question=m.question,
            answer=m.answer,
            domain=m.domain,
            entity_id=m.entity_id,
            similarity_score=m.similarity_score,
            metadata=m.metadata,
        )
        for m in pg_matches_raw
    ]

    threshold = _get_threshold_for_query_type(query_type)

    # Determine which backend to serve (same priority as normal flow)
    served_by = "none"
    state_update: dict[str, object]
    uky_answer = uky_raw.get("response")

    if uky_answer:
        # UKY succeeded — serve its answer
        served_by = f"uky_{rag_endpoint}"
        span.set_attribute("rag.source", served_by)
        span.set_attribute("rag.answer_length", len(uky_answer))
        span.set_attribute("rag.dual_logging", True)

        rag_match = RAGMatch(
            id=f"uky-{rag_endpoint}",
            question=search_query,
            answer=uky_answer,
            domain=f"uky-{rag_endpoint}",
            entity_id=f"uky-{rag_endpoint}",
            similarity_score=1.0,
        )

        if query_type == "static":
            span.set_attribute("rag.result", "uky_direct_answer")
            state_update = {
                "final_answer": uky_answer,
                "messages": [AIMessage(content=uky_answer)],
                "tools_used": ["uky_rag_retrieval"],
                "rag_matches": [rag_match],
                "rag_used": True,
            }
        else:
            span.set_attribute("rag.result", "uky_for_synthesis")
            state_update = {
                "rag_matches": [rag_match],
                "rag_used": True,
            }

    elif pg_rag_matches:
        # pgvector has matches — serve from pgvector
        best_match = pg_rag_matches[0]
        span.set_attribute("rag.source", "pgvector")
        span.set_attribute("rag.matches_found", len(pg_rag_matches))
        span.set_attribute("rag.best_score", best_match.similarity_score)
        span.set_attribute("rag.dual_logging", True)

        if query_type == "static" and best_match.similarity_score >= threshold:
            served_by = "pgvector"
            span.set_attribute("rag.result", "direct_match_for_synthesis")
            span.set_attribute("rag.answer_length", len(best_match.answer))
            state_update = {
                "tools_used": ["rag_retrieval"],
                "rag_matches": pg_rag_matches,
                "rag_used": True,
            }
        elif query_type == "combined":
            served_by = "pgvector"
            span.set_attribute("rag.result", "matches_for_synthesis")
            state_update = {
                "rag_matches": pg_rag_matches,
                "rag_used": True,
            }
        else:
            span.set_attribute("rag.result", "below_threshold")
            state_update = {
                "rag_matches": pg_rag_matches,
                "rag_used": len(pg_rag_matches) > 0,
            }
    else:
        # Neither backend returned results
        span.set_attribute("rag.result", "no_matches")
        span.set_attribute("rag.dual_logging", True)
        state_update = {"rag_matches": [], "rag_used": False}

    # Determine served answer length
    served_answer = state_update.get("final_answer")
    served_answer_length = len(served_answer) if served_answer else None

    # Synthesize pgvector answer for fair comparison logging
    pgvector_synthesized = None
    if pg_rag_matches:
        try:
            rag_context = _format_rag_matches(pg_rag_matches)
            synth_result = await _synthesize_with_rag_only(query, rag_context)
            pgvector_synthesized = synth_result.get("final_answer")
        except Exception as e:
            logger.error(f"Failed to synthesize pgvector answer for comparison: {e}")

    # Log the comparison (fire-and-forget, never blocks response)
    try:
        comparison_logger = get_rag_comparison_logger()
        comparison_logger.log_comparison(
            query_text=query,
            expanded_query=search_query,
            session_id=session_id,
            question_id=question_id,
            query_type=query_type,
            rag_endpoint=rag_endpoint,
            uky_response=uky_answer,
            uky_duration_ms=uky_raw.get("duration_ms"),
            uky_error=uky_raw.get("error"),
            pgvector_matches=[
                {
                    "id": m.id,
                    "question": m.question,
                    "answer": m.answer[:500],
                    "similarity_score": m.similarity_score,
                    "domain": m.domain,
                    "entity_id": m.entity_id,
                }
                for m in pg_rag_matches
            ] or None,
            pgvector_best_score=pg_rag_matches[0].similarity_score if pg_rag_matches else None,
            pgvector_match_count=len(pg_rag_matches),
            pgvector_duration_ms=pg_raw.get("duration_ms"),
            pgvector_error=pg_raw.get("error"),
            pgvector_synthesized_answer=pgvector_synthesized,
            served_by=served_by,
            served_answer_length=served_answer_length,
        )
    except Exception as e:
        logger.error(f"Failed to log RAG comparison: {e}")

    return state_update


async def rag_answer_node(state: AgentState) -> dict[str, object]:
    """Retrieve answer from UKY RAG endpoints or pgvector Q&A service.

    Routes to UKY endpoints when classification.rag_endpoint is set,
    falling back to pgvector if UKY fails or isn't configured.

    Args:
        state: Current agent state with query.

    Returns:
        State update with:
        - For static with UKY match: final_answer, messages, tools_used, rag_matches, rag_used → END
        - For static with pgvector match: tools_used, rag_matches, rag_used (no final_answer) → synthesize
        - For combined: rag_matches, rag_used (no final_answer) → plan → tools
        - For no match: rag_matches, rag_used=False → plan (fallback to tools)
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

        # Dual-RAG path: query both backends in parallel, log comparison
        if settings.DUAL_RAG_LOGGING and rag_endpoint:
            logger.info("Dual-RAG logging enabled — querying UKY and pgvector in parallel")
            result = await _dual_rag_answer(
                search_query=search_query,
                query=query,
                query_type=query_type,
                rag_endpoint=rag_endpoint,
                session_id=state.get("session_id", ""),
                question_id=state.get("question_id", ""),
                span=span,
            )
        elif rag_endpoint:
            # Normal path: UKY first, pgvector fallback
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
