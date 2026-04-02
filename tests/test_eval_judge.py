"""Tests for the LLM judge."""

import json

from src.eval.judge import parse_judge_response


class TestParseJudgeResponse:
    def test_valid_response(self):
        raw = json.dumps(
            {
                "correctness": {"score": 4, "justification": "Good"},
                "completeness": {"score": 3, "justification": "Missing some"},
                "relevance": {"score": 5, "justification": "On point"},
                "citation_quality": {"score": 4, "justification": "URLs present"},
                "hedging": {"score": 5, "justification": "Well calibrated"},
            }
        )
        result = parse_judge_response(raw)
        assert result is not None
        assert result.scores["correctness"] == 4
        assert result.scores["completeness"] == 3
        assert result.justifications["relevance"] == "On point"
        assert abs(result.composite - 4.05) < 0.01

    def test_json_in_markdown_code_block(self):
        inner = json.dumps(
            {
                "correctness": {"score": 5, "justification": "x"},
                "completeness": {"score": 5, "justification": "x"},
                "relevance": {"score": 5, "justification": "x"},
                "citation_quality": {"score": 5, "justification": "x"},
                "hedging": {"score": 5, "justification": "x"},
            }
        )
        raw = f"```json\n{inner}\n```"
        result = parse_judge_response(raw)
        assert result is not None
        assert result.scores["correctness"] == 5

    def test_invalid_json_returns_none(self):
        result = parse_judge_response("this is not json")
        assert result is None

    def test_missing_dimension_returns_none(self):
        raw = json.dumps(
            {
                "correctness": {"score": 4, "justification": "Good"},
            }
        )
        result = parse_judge_response(raw)
        assert result is None

    def test_score_out_of_range_returns_none(self):
        raw = json.dumps(
            {
                "correctness": {"score": 6, "justification": "Too high"},
                "completeness": {"score": 3, "justification": "Ok"},
                "relevance": {"score": 5, "justification": "Ok"},
                "citation_quality": {"score": 4, "justification": "Ok"},
                "hedging": {"score": 5, "justification": "Ok"},
            }
        )
        result = parse_judge_response(raw)
        assert result is None
