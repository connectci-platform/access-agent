"""Tests for the rag_answer node helpers."""

from src.agent.nodes.rag_answer import _rag_response_out_of_scope
from src.agent.state import RAGMatch


def _make_rag_match(answer: str) -> RAGMatch:
    return RAGMatch(
        id="test",
        question="test query",
        answer=answer,
        domain="test",
        entity_id="test",
        similarity_score=1.0,
    )


class TestRagResponseOutOfScope:
    """Verify the out-of-scope detection prefers UKY in_scope, falls back to text."""

    # ── Authoritative signal: uky_in_scope ──────────────────────────────

    def test_in_scope_true_returns_false(self):
        """When UKY explicitly says in_scope=True, the response is in scope."""
        result = {
            "uky_in_scope": True,
            "final_answer": "I don't have any information about that topic.",
        }
        # Even with text that looks out-of-scope, UKY's signal wins
        assert _rag_response_out_of_scope(result) is False

    def test_in_scope_false_returns_true(self):
        """When UKY explicitly says in_scope=False, the response is out of scope."""
        result = {
            "uky_in_scope": False,
            "final_answer": "Here is a perfectly normal answer with no hedging at all.",
        }
        # Even with text that looks fine, UKY's signal wins
        assert _rag_response_out_of_scope(result) is True

    # ── Text fallback (when uky_in_scope is None or missing) ─────────────

    def test_text_fallback_detects_outside_scope(self):
        result = {
            "final_answer": "That question is outside the scope of my knowledge.",
        }
        assert _rag_response_out_of_scope(result) is True

    def test_text_fallback_detects_dont_have_info(self):
        result = {
            "final_answer": "I don't have information about that topic.",
        }
        assert _rag_response_out_of_scope(result) is True

    def test_text_fallback_detects_only_answer_about(self):
        result = {
            "final_answer": "I can only answer questions about Delta.",
        }
        assert _rag_response_out_of_scope(result) is True

    def test_text_fallback_detects_no_documents(self):
        result = {
            "final_answer": "No documents are currently available for that resource.",
        }
        assert _rag_response_out_of_scope(result) is True

    def test_text_fallback_in_scope_response(self):
        result = {
            "final_answer": "Delta has 124 GPU nodes with NVIDIA A100 accelerators.",
        }
        assert _rag_response_out_of_scope(result) is False

    def test_text_fallback_uses_rag_match_when_no_final_answer(self):
        """Combined queries don't set final_answer; check rag_matches instead."""
        result = {
            "rag_matches": [
                _make_rag_match("That topic is outside the scope of this collection."),
            ],
        }
        assert _rag_response_out_of_scope(result) is True

    def test_empty_result_returns_false(self):
        """An empty result dict shouldn't trigger out-of-scope."""
        assert _rag_response_out_of_scope({}) is False

    def test_no_answer_no_matches_returns_false(self):
        result = {"rag_matches": [], "final_answer": ""}
        assert _rag_response_out_of_scope(result) is False
