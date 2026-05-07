"""Tests for runtime capability gates.

These tests verify that the capability registry's `enabled_rag_endpoints`
and `scoped_rag_enabled` queries are actually consulted by rag_answer_node
and domain_agent_node.
"""

from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
