"""Tests for Argilla push/pull integration."""

from src.eval.argilla_pull import extract_scores_from_response
from src.eval.argilla_push import build_argilla_record, dataset_name_for_branch


class TestDatasetNaming:
    def test_branch_dataset_name(self):
        assert dataset_name_for_branch("feature/new-tool") == "eval-feature-new-tool"

    def test_main_branch(self):
        assert dataset_name_for_branch("main") == "eval-main"

    def test_production_dataset(self):
        assert dataset_name_for_branch(None) == "eval-production"


class TestBuildRecord:
    def test_record_has_required_fields(self):
        record = build_argilla_record(
            question_id="q-001",
            question_text="What is ACCESS?",
            answer_text="ACCESS is a program.",
            judge_scores={
                "correctness": 4,
                "completeness": 3,
                "relevance": 5,
                "citation_quality": 4,
                "hedging": 5,
            },
            composite_score=4.05,
            capability_area="general",
        )
        assert record["fields"]["user_query"] == "What is ACCESS?"
        assert record["fields"]["agent_answer"] == "ACCESS is a program."
        assert record["metadata"]["capability_area"] == "general"
        assert record["metadata"]["composite_score"] == "4.05"

    def test_record_includes_context(self):
        record = build_argilla_record(
            question_id="q-001",
            question_text="test",
            answer_text="test answer",
            judge_scores={
                "correctness": 5,
                "completeness": 5,
                "relevance": 5,
                "citation_quality": 5,
                "hedging": 5,
            },
            composite_score=5.0,
            rag_context="Some RAG context here",
            tool_results="Tool returned data",
            node_trace="classify -> rag -> synthesize",
        )
        assert record["fields"]["rag_context"] == "Some RAG context here"
        assert record["fields"]["tool_results"] == "Tool returned data"
        assert record["fields"]["agent_reasoning"] == "classify -> rag -> synthesize"

    def test_record_includes_judge_suggestions(self):
        record = build_argilla_record(
            question_id="q-001",
            question_text="test",
            answer_text="test",
            judge_scores={
                "correctness": 4,
                "completeness": 3,
                "relevance": 5,
                "citation_quality": 4,
                "hedging": 5,
            },
            composite_score=4.05,
        )
        assert record["suggestions"]["correctness"] == 4
        assert record["suggestions"]["completeness"] == 3


class TestExtractScores:
    def test_extract_complete_response(self):
        response = {
            "correctness": 4,
            "completeness": 3,
            "relevance": 5,
            "citation_quality": 4,
            "hedging": 5,
            "decision": "approved",
            "feedback": "Good answer",
        }
        result = extract_scores_from_response(response, reviewer_id="reviewer-1")
        assert result["correctness"] == 4
        assert "Decision: approved" in result["feedback"]
        assert "Good answer" in result["feedback"]
        assert result["reviewer_id"] == "reviewer-1"
        assert result["composite_score"] > 0

    def test_extract_missing_feedback(self):
        response = {
            "correctness": 5,
            "completeness": 5,
            "relevance": 5,
            "citation_quality": 5,
            "hedging": 5,
            "decision": "approved",
            "feedback": None,
        }
        result = extract_scores_from_response(response, reviewer_id="r1")
        assert result["feedback"] == "Decision: approved"
        assert result["composite_score"] == 5.0

    def test_extract_no_decision_no_feedback(self):
        response = {
            "correctness": 3,
            "completeness": 3,
            "relevance": 3,
            "citation_quality": 3,
            "hedging": 3,
        }
        result = extract_scores_from_response(response, reviewer_id="r1")
        assert result["feedback"] is None
