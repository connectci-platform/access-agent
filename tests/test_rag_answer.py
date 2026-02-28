"""Tests for the RAG answer node, including dual-RAG comparison logging.

Tests the dual-RAG path (_dual_rag_answer) which queries both UKY and pgvector
in parallel and logs results for A.3 evaluation. Also verifies the normal path
is unchanged when the flag is off.

Run with: uv run pytest tests/test_rag_answer.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.nodes.rag_answer import (
    _dual_rag_answer,
    _query_pgvector_raw,
    _query_uky_raw,
    process_citations,
)
from src.agent.state import QueryClassification


# --- Citation processing ---


class TestProcessCitations:
    def test_converts_citation_markers(self):
        text = "Delta has A100 GPUs <<SRC:compute-resources:delta.ncsa.access-ci.org>>"
        result = process_citations(text)
        assert result == "Delta has A100 GPUs [Source: compute-resources/delta.ncsa.access-ci.org]"

    def test_multiple_citations(self):
        text = "<<SRC:a:b>> and <<SRC:c:d>>"
        result = process_citations(text)
        assert result == "[Source: a/b] and [Source: c/d]"

    def test_no_citations(self):
        text = "No citations here"
        assert process_citations(text) == text


# --- Raw query helpers ---


class TestQueryUkyRaw:
    @pytest.mark.asyncio
    async def test_returns_response_on_success(self):
        mock_response = MagicMock()
        mock_response.response = "Delta has NVIDIA A100 GPUs"
        mock_response.duration_ms = 450

        mock_client = AsyncMock()
        mock_client.is_configured = True
        mock_client.ask.return_value = mock_response

        with patch("src.agent.nodes.rag_answer.get_uky_client", return_value=mock_client):
            result = await _query_uky_raw("What GPUs does Delta have?", "general", "sess_1", "q_1")

        assert result["response"] == "Delta has NVIDIA A100 GPUs"
        assert result["duration_ms"] == 450
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_returns_error_when_not_configured(self):
        mock_client = AsyncMock()
        mock_client.is_configured = False

        with patch("src.agent.nodes.rag_answer.get_uky_client", return_value=mock_client):
            result = await _query_uky_raw("test", "general", "s", "q")

        assert result["response"] is None
        assert result["error"] == "not_configured"

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        mock_client = AsyncMock()
        mock_client.is_configured = True
        mock_client.ask.side_effect = ConnectionError("timeout")

        with patch("src.agent.nodes.rag_answer.get_uky_client", return_value=mock_client):
            result = await _query_uky_raw("test", "general", "s", "q")

        assert result["response"] is None
        assert "timeout" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_response(self):
        mock_response = MagicMock()
        mock_response.response = ""
        mock_response.duration_ms = 100

        mock_client = AsyncMock()
        mock_client.is_configured = True
        mock_client.ask.return_value = mock_response

        with patch("src.agent.nodes.rag_answer.get_uky_client", return_value=mock_client):
            result = await _query_uky_raw("test", "general", "s", "q")

        assert result["response"] is None


class TestQueryPgvectorRaw:
    @pytest.mark.asyncio
    async def test_returns_matches_on_success(self):
        mock_match = MagicMock()
        mock_match.id = "qa-123"
        mock_match.question = "What GPUs does Delta have?"
        mock_match.answer = "Delta has A100s"
        mock_match.domain = "compute-resources"
        mock_match.entity_id = "delta"
        mock_match.similarity_score = 0.92
        mock_match.metadata = {}

        mock_client = AsyncMock()
        mock_client.is_configured = True
        mock_client.search.return_value = [mock_match]

        with patch("src.agent.nodes.rag_answer.get_qa_client", return_value=mock_client):
            result = await _query_pgvector_raw("What GPUs?", "static")

        assert len(result["matches"]) == 1
        assert result["duration_ms"] is not None
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_returns_error_when_not_configured(self):
        mock_client = AsyncMock()
        mock_client.is_configured = False

        with patch("src.agent.nodes.rag_answer.get_qa_client", return_value=mock_client):
            result = await _query_pgvector_raw("test", "static")

        assert result["matches"] == []
        assert result["error"] == "not_configured"

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        mock_client = AsyncMock()
        mock_client.is_configured = True
        mock_client.search.side_effect = ConnectionError("refused")

        with patch("src.agent.nodes.rag_answer.get_qa_client", return_value=mock_client):
            result = await _query_pgvector_raw("test", "static")

        assert result["matches"] == []
        assert "refused" in result["error"]


# --- Dual RAG answer ---


def _make_mock_span():
    """Create a mock OpenTelemetry span."""
    span = MagicMock()
    span.set_attribute = MagicMock()
    return span


def _make_mock_qa_match(
    id="qa-1",
    question="What GPUs does Delta have?",
    answer="Delta has A100 GPUs <<SRC:compute-resources:delta>>",
    domain="compute-resources",
    entity_id="delta",
    similarity_score=0.92,
):
    m = MagicMock()
    m.id = id
    m.question = question
    m.answer = answer
    m.domain = domain
    m.entity_id = entity_id
    m.similarity_score = similarity_score
    m.metadata = {}
    return m


class TestDualRagAnswer:
    """Tests for _dual_rag_answer — the parallel query + comparison logging path."""

    @pytest.mark.asyncio
    async def test_uky_succeeds_static_serves_uky(self):
        """When both succeed on a static query, UKY answer is served."""
        span = _make_mock_span()

        with (
            patch("src.agent.nodes.rag_answer._query_uky_raw", new_callable=AsyncMock) as mock_uky,
            patch(
                "src.agent.nodes.rag_answer._query_pgvector_raw", new_callable=AsyncMock
            ) as mock_pg,
            patch("src.agent.nodes.rag_answer.get_rag_comparison_logger") as mock_logger_fn,
        ):
            mock_uky.return_value = {
                "response": "UKY answer about Delta GPUs",
                "duration_ms": 500,
                "error": None,
            }
            mock_pg.return_value = {
                "matches": [_make_mock_qa_match()],
                "duration_ms": 100,
                "error": None,
            }
            mock_comparison_logger = MagicMock()
            mock_logger_fn.return_value = mock_comparison_logger

            result = await _dual_rag_answer(
                search_query="What GPUs does Delta have?",
                query="What GPUs does Delta have?",
                query_type="static",
                rag_endpoint="general",
                session_id="sess_1",
                question_id="q_1",
                span=span,
            )

        # UKY answer is served
        assert result["final_answer"] == "UKY answer about Delta GPUs"
        assert result["rag_used"] is True
        assert "uky_rag_retrieval" in result["tools_used"]

        # Both backends were queried
        mock_uky.assert_awaited_once()
        mock_pg.assert_awaited_once()

        # Comparison was logged
        mock_comparison_logger.log_comparison.assert_called_once()
        log_kwargs = mock_comparison_logger.log_comparison.call_args.kwargs
        assert log_kwargs["served_by"] == "uky_general"
        assert log_kwargs["uky_response"] == "UKY answer about Delta GPUs"
        assert log_kwargs["pgvector_match_count"] == 1

    @pytest.mark.asyncio
    async def test_uky_fails_static_falls_back_to_pgvector(self):
        """When UKY fails on a static query, pgvector answer is served."""
        span = _make_mock_span()

        with (
            patch("src.agent.nodes.rag_answer._query_uky_raw", new_callable=AsyncMock) as mock_uky,
            patch(
                "src.agent.nodes.rag_answer._query_pgvector_raw", new_callable=AsyncMock
            ) as mock_pg,
            patch("src.agent.nodes.rag_answer.get_rag_comparison_logger") as mock_logger_fn,
        ):
            mock_uky.return_value = {
                "response": None,
                "duration_ms": None,
                "error": "Connection refused",
            }
            mock_pg.return_value = {
                "matches": [_make_mock_qa_match(similarity_score=0.92)],
                "duration_ms": 80,
                "error": None,
            }
            mock_comparison_logger = MagicMock()
            mock_logger_fn.return_value = mock_comparison_logger

            result = await _dual_rag_answer(
                search_query="What GPUs does Delta have?",
                query="What GPUs does Delta have?",
                query_type="static",
                rag_endpoint="general",
                session_id="sess_1",
                question_id="q_1",
                span=span,
            )

        # pgvector answer is served (with citations processed)
        assert result["final_answer"] is not None
        assert "rag_retrieval" in result["tools_used"]
        assert result["rag_used"] is True

        # Comparison logged with UKY error
        log_kwargs = mock_comparison_logger.log_comparison.call_args.kwargs
        assert log_kwargs["served_by"] == "pgvector"
        assert log_kwargs["uky_error"] == "Connection refused"
        assert log_kwargs["pgvector_best_score"] == 0.92

    @pytest.mark.asyncio
    async def test_both_fail_returns_no_match(self):
        """When both backends fail, returns empty state."""
        span = _make_mock_span()

        with (
            patch("src.agent.nodes.rag_answer._query_uky_raw", new_callable=AsyncMock) as mock_uky,
            patch(
                "src.agent.nodes.rag_answer._query_pgvector_raw", new_callable=AsyncMock
            ) as mock_pg,
            patch("src.agent.nodes.rag_answer.get_rag_comparison_logger") as mock_logger_fn,
        ):
            mock_uky.return_value = {
                "response": None,
                "duration_ms": None,
                "error": "UKY down",
            }
            mock_pg.return_value = {
                "matches": [],
                "duration_ms": 50,
                "error": "not_configured",
            }
            mock_comparison_logger = MagicMock()
            mock_logger_fn.return_value = mock_comparison_logger

            result = await _dual_rag_answer(
                search_query="Something obscure",
                query="Something obscure",
                query_type="static",
                rag_endpoint="general",
                session_id="sess_1",
                question_id="q_1",
                span=span,
            )

        assert result["rag_matches"] == []
        assert result["rag_used"] is False
        assert "final_answer" not in result

        log_kwargs = mock_comparison_logger.log_comparison.call_args.kwargs
        assert log_kwargs["served_by"] == "none"
        assert log_kwargs["uky_error"] == "UKY down"

    @pytest.mark.asyncio
    async def test_combined_query_stores_matches_no_final_answer(self):
        """For combined queries, UKY answer goes to rag_matches, not final_answer."""
        span = _make_mock_span()

        with (
            patch("src.agent.nodes.rag_answer._query_uky_raw", new_callable=AsyncMock) as mock_uky,
            patch(
                "src.agent.nodes.rag_answer._query_pgvector_raw", new_callable=AsyncMock
            ) as mock_pg,
            patch("src.agent.nodes.rag_answer.get_rag_comparison_logger") as mock_logger_fn,
        ):
            mock_uky.return_value = {
                "response": "UKY answer for synthesis",
                "duration_ms": 600,
                "error": None,
            }
            mock_pg.return_value = {
                "matches": [_make_mock_qa_match()],
                "duration_ms": 90,
                "error": None,
            }
            mock_logger_fn.return_value = MagicMock()

            result = await _dual_rag_answer(
                search_query="Which A100 resources are available?",
                query="Which A100 resources are available?",
                query_type="combined",
                rag_endpoint="general",
                session_id="sess_1",
                question_id="q_1",
                span=span,
            )

        # Combined: rag_matches set, no final_answer
        assert "final_answer" not in result
        assert result["rag_used"] is True
        assert len(result["rag_matches"]) == 1

    @pytest.mark.asyncio
    async def test_pgvector_below_threshold_not_served(self):
        """When UKY fails and pgvector is below threshold, no answer is served."""
        span = _make_mock_span()

        with (
            patch("src.agent.nodes.rag_answer._query_uky_raw", new_callable=AsyncMock) as mock_uky,
            patch(
                "src.agent.nodes.rag_answer._query_pgvector_raw", new_callable=AsyncMock
            ) as mock_pg,
            patch("src.agent.nodes.rag_answer.get_rag_comparison_logger") as mock_logger_fn,
        ):
            mock_uky.return_value = {
                "response": None,
                "duration_ms": None,
                "error": "not_configured",
            }
            # Score below static threshold (0.85)
            mock_pg.return_value = {
                "matches": [_make_mock_qa_match(similarity_score=0.70)],
                "duration_ms": 80,
                "error": None,
            }
            mock_logger_fn.return_value = MagicMock()

            result = await _dual_rag_answer(
                search_query="Obscure question",
                query="Obscure question",
                query_type="static",
                rag_endpoint="general",
                session_id="sess_1",
                question_id="q_1",
                span=span,
            )

        # Below threshold — no final_answer, but rag_matches still populated
        assert "final_answer" not in result
        assert len(result["rag_matches"]) == 1

    @pytest.mark.asyncio
    async def test_comparison_logger_failure_does_not_block(self):
        """If the comparison logger throws, the response is still returned."""
        span = _make_mock_span()

        with (
            patch("src.agent.nodes.rag_answer._query_uky_raw", new_callable=AsyncMock) as mock_uky,
            patch(
                "src.agent.nodes.rag_answer._query_pgvector_raw", new_callable=AsyncMock
            ) as mock_pg,
            patch("src.agent.nodes.rag_answer.get_rag_comparison_logger") as mock_logger_fn,
        ):
            mock_uky.return_value = {
                "response": "UKY answer",
                "duration_ms": 500,
                "error": None,
            }
            mock_pg.return_value = {"matches": [], "duration_ms": 50, "error": None}

            mock_comparison_logger = MagicMock()
            mock_comparison_logger.log_comparison.side_effect = RuntimeError("DB connection lost")
            mock_logger_fn.return_value = mock_comparison_logger

            # Should NOT raise
            result = await _dual_rag_answer(
                search_query="test",
                query="test",
                query_type="static",
                rag_endpoint="general",
                session_id="s",
                question_id="q",
                span=span,
            )

        assert result["final_answer"] == "UKY answer"


# --- rag_answer_node flag gating ---


class TestRagAnswerNodeFlagGating:
    """Verify that DUAL_RAG_LOGGING flag controls which path is taken."""

    @pytest.mark.asyncio
    async def test_flag_off_uses_normal_path(self):
        """With DUAL_RAG_LOGGING=False, _dual_rag_answer is never called."""
        from src.agent.nodes.rag_answer import rag_answer_node

        state = {
            "query": "What GPUs does Delta have?",
            "query_classification": QueryClassification(
                query_type="static",
                reason="test",
                expanded_query="What GPUs does Delta have?",
                rag_endpoint="general",
            ),
            "session_id": "sess_1",
            "question_id": "q_1",
        }

        with (
            patch("src.agent.nodes.rag_answer.settings") as mock_settings,
            patch("src.agent.nodes.rag_answer.get_tracer") as mock_tracer,
            patch(
                "src.agent.nodes.rag_answer._dual_rag_answer", new_callable=AsyncMock
            ) as mock_dual,
            patch("src.agent.nodes.rag_answer._ask_uky", new_callable=AsyncMock) as mock_uky,
        ):
            mock_settings.DUAL_RAG_LOGGING = False
            mock_tracer.return_value.start_as_current_span.return_value.__enter__ = MagicMock()
            mock_tracer.return_value.start_as_current_span.return_value.__exit__ = MagicMock()
            mock_uky.return_value = {
                "final_answer": "Normal path answer",
                "messages": [],
                "tools_used": ["uky_rag_retrieval"],
                "rag_matches": [],
                "rag_used": True,
            }

            result = await rag_answer_node(state)

        mock_dual.assert_not_awaited()
        assert result["final_answer"] == "Normal path answer"

    @pytest.mark.asyncio
    async def test_flag_on_with_endpoint_uses_dual_path(self):
        """With DUAL_RAG_LOGGING=True and rag_endpoint set, uses _dual_rag_answer."""
        from src.agent.nodes.rag_answer import rag_answer_node

        state = {
            "query": "What GPUs does Delta have?",
            "query_classification": QueryClassification(
                query_type="static",
                reason="test",
                expanded_query="What GPUs does Delta have?",
                rag_endpoint="general",
            ),
            "session_id": "sess_1",
            "question_id": "q_1",
        }

        with (
            patch("src.agent.nodes.rag_answer.settings") as mock_settings,
            patch("src.agent.nodes.rag_answer.get_tracer") as mock_tracer,
            patch(
                "src.agent.nodes.rag_answer._dual_rag_answer", new_callable=AsyncMock
            ) as mock_dual,
        ):
            mock_settings.DUAL_RAG_LOGGING = True
            mock_tracer.return_value.start_as_current_span.return_value.__enter__ = MagicMock()
            mock_tracer.return_value.start_as_current_span.return_value.__exit__ = MagicMock()
            mock_dual.return_value = {
                "final_answer": "Dual path answer",
                "messages": [],
                "tools_used": ["uky_rag_retrieval"],
                "rag_matches": [],
                "rag_used": True,
            }

            result = await rag_answer_node(state)

        mock_dual.assert_awaited_once()
        assert result["final_answer"] == "Dual path answer"

    @pytest.mark.asyncio
    async def test_flag_on_without_endpoint_uses_normal_path(self):
        """With DUAL_RAG_LOGGING=True but no rag_endpoint, skips dual path."""
        from src.agent.nodes.rag_answer import rag_answer_node

        state = {
            "query": "Some dynamic query",
            "query_classification": QueryClassification(
                query_type="static",
                reason="test",
                expanded_query="Some dynamic query",
                rag_endpoint=None,
            ),
            "session_id": "sess_1",
            "question_id": "q_1",
        }

        with (
            patch("src.agent.nodes.rag_answer.settings") as mock_settings,
            patch("src.agent.nodes.rag_answer.get_tracer") as mock_tracer,
            patch(
                "src.agent.nodes.rag_answer._dual_rag_answer", new_callable=AsyncMock
            ) as mock_dual,
            patch(
                "src.agent.nodes.rag_answer._search_pgvector", new_callable=AsyncMock
            ) as mock_pg,
        ):
            mock_settings.DUAL_RAG_LOGGING = True
            mock_tracer.return_value.start_as_current_span.return_value.__enter__ = MagicMock()
            mock_tracer.return_value.start_as_current_span.return_value.__exit__ = MagicMock()
            mock_pg.return_value = {"rag_matches": [], "rag_used": False}

            result = await rag_answer_node(state)

        mock_dual.assert_not_awaited()
