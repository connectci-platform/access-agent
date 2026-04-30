"""Tests for runtime capability gates.

These tests verify that the capability registry's `enabled_rag_endpoints`,
`scoped_rag_enabled`, and `is_domain_enabled` queries are actually consulted
by rag_answer_node, route_after_rag, and domain_agent_node.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agent.graph import route_after_rag
from src.agent.nodes.domain_agent import domain_agent_node
from src.agent.nodes.rag_answer import rag_answer_node


@pytest.fixture(autouse=True)
def _reset_registry():
    """Ensure every test starts with a fresh capability registry build."""
    from src.agent.domains import capabilities as cap_module

    cap_module._registry = None
    yield
    cap_module._registry = None


@contextmanager
def _patched_settings(enabled: str = "", disabled: str = ""):
    """Patch both capability env vars for the duration of the block."""
    from src.config import settings

    with (
        patch.object(settings, "ENABLED_CAPABILITIES", enabled),
        patch.object(settings, "DISABLED_CAPABILITIES", disabled),
    ):
        yield


# ── rag_answer_node: RAG endpoint gating ────────────────────────────


class TestRagAnswerGate:
    """rag_answer_node should skip disabled RAG endpoints."""

    def _base_state(self) -> dict:
        return {
            "query": "Tell me about Delta",
            "query_classification": MagicMock(
                query_type="static",
                rag_endpoint="general",
                expanded_query="Tell me about Delta",
            ),
            "session_id": "test-session",
            "question_id": "test-question",
            "resource_context": None,
        }

    @pytest.mark.asyncio
    async def test_skips_when_all_rag_disabled(self):
        state = self._base_state()
        with (
            _patched_settings(disabled="ask_question,ask_xdmod_question,ask_about_resource"),
            patch("src.agent.nodes.rag_answer._ask_uky") as mock_ask,
            patch(
                "src.agent.nodes.rag_answer.get_stream_writer",
                return_value=lambda _: None,
            ),
        ):
            result = await rag_answer_node(state)
            mock_ask.assert_not_called()
            assert result.get("rag_used") is False
            assert result.get("rag_matches") == []

    @pytest.mark.asyncio
    async def test_scoped_rag_falls_back_to_unscoped_when_scoped_disabled(self):
        state = self._base_state()
        state["resource_context"] = "delta"
        # Disable only the scoped capability; keep general RAG
        with (
            _patched_settings(disabled="ask_about_resource"),
            patch(
                "src.agent.nodes.rag_answer._ask_uky",
                new=AsyncMock(return_value={"rag_matches": [], "rag_used": False}),
            ) as mock_ask,
            patch(
                "src.agent.nodes.rag_answer.get_stream_writer",
                return_value=lambda _: None,
            ),
        ):
            await rag_answer_node(state)
            mock_ask.assert_called_once()
            # rp_name should be None because scoped RAG is disabled
            assert mock_ask.call_args.kwargs.get("rp_name") is None

    @pytest.mark.asyncio
    async def test_calls_uky_when_endpoint_enabled(self):
        state = self._base_state()
        with (
            patch(
                "src.agent.nodes.rag_answer._ask_uky",
                new=AsyncMock(return_value={"rag_matches": [], "rag_used": True}),
            ) as mock_ask,
            patch(
                "src.agent.nodes.rag_answer.get_stream_writer",
                return_value=lambda _: None,
            ),
        ):
            await rag_answer_node(state)
            mock_ask.assert_called_once()


# ── route_after_rag: domain routing gate ────────────────────────────


class TestRouteAfterRagGate:
    """route_after_rag should bypass domain_agent for disabled domains."""

    def _domain_state(self, domain: str) -> dict:
        return {
            "query_classification": MagicMock(query_type="combined", domain=domain),
            "rag_matches": [],
        }

    def test_routes_to_domain_when_enabled(self):
        state = self._domain_state("announcements")
        assert route_after_rag(state) == "domain_agent"

    def test_falls_through_to_plan_when_domain_disabled(self):
        state = self._domain_state("announcements")
        with _patched_settings(disabled="search_announcements,manage_announcements"):
            assert route_after_rag(state) == "plan"

    def test_falls_through_when_jsm_disabled(self):
        state = self._domain_state("jsm")
        with _patched_settings(disabled="open_ticket,report_login_problem,report_security"):
            assert route_after_rag(state) == "plan"


# ── domain_agent_node: defense-in-depth ─────────────────────────────


class TestDomainAgentDefenseInDepth:
    """If routing somehow dispatches to a disabled domain, the node refuses."""

    @pytest.mark.asyncio
    async def test_refuses_disabled_domain(self):
        state = {
            "query": "Create an announcement",
            "query_classification": MagicMock(domain="announcements"),
        }
        with (
            _patched_settings(disabled="search_announcements,manage_announcements"),
            patch(
                "src.agent.nodes.domain_agent.get_stream_writer",
                return_value=lambda _: None,
            ),
        ):
            result = await domain_agent_node(state)
            # Defense-in-depth returns a user-visible message, not an
            # empty string — the user should get something actionable.
            assert "workflow isn't available" in result["final_answer"]
            assert result["messages"]  # non-empty
            assert result["node_trace"][0]["error"] == "domain_disabled"


# ── route_after_rag: deflection + no-rag-matches fallback paths ──────


class TestRouteAfterRagFallback:
    """route_after_rag must fall back to the tool path on (1) RAG deflections
    and (2) combined/dynamic queries with no RAG matches. These are the
    routing recovery paths a regression could break silently — covers the
    deflection branch and the combined-query no-RAG-matches log/return path
    in graph.py.
    """

    def test_static_query_with_deflection_falls_back_to_tools(self, monkeypatch):
        """A static RAG answer that's just a hedge ('do not contain X') must
        route to the tool path, not END. Without this, every RAG deflection
        would be served verbatim to the user."""
        monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", False)
        state = {
            "query_classification": MagicMock(query_type="static", domain=None),
            "rag_matches": [],
            "final_answer": (
                "The provided documents do not contain specific information about that."
            ),
        }
        assert route_after_rag(state) == "plan"

    def test_combined_query_with_no_rag_matches_routes_to_tools(self, monkeypatch):
        """Combined query, RAG returned nothing — should still route to tools.
        Exercises the empty-rag_matches branch (the alternative to the
        'continuing to plan for supplementary data' log line)."""
        monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", False)
        state = {
            "query_classification": MagicMock(query_type="combined", domain=None),
            "rag_matches": [],
        }
        assert route_after_rag(state) == "plan"
