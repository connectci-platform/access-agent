"""Tests for the eval rubric scoring."""

from src.eval.rubric import DEFAULT_WEIGHTS, DIMENSIONS, compute_composite


class TestDimensions:
    def test_all_dimensions_present(self):
        names = [d.name for d in DIMENSIONS]
        assert names == [
            "correctness",
            "completeness",
            "relevance",
            "citation_quality",
            "hedging",
        ]

    def test_weights_sum_to_one(self):
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001


class TestCompositeScore:
    def test_perfect_scores(self):
        scores = {
            "correctness": 5,
            "completeness": 5,
            "relevance": 5,
            "citation_quality": 5,
            "hedging": 5,
        }
        assert compute_composite(scores) == 5.0

    def test_weighted_calculation(self):
        scores = {
            "correctness": 5,  # weight 0.30 -> 1.50
            "completeness": 4,  # weight 0.25 -> 1.00
            "relevance": 3,  # weight 0.20 -> 0.60
            "citation_quality": 2,  # weight 0.15 -> 0.30
            "hedging": 1,  # weight 0.10 -> 0.10
        }
        expected = 1.50 + 1.00 + 0.60 + 0.30 + 0.10  # 3.50
        assert abs(compute_composite(scores) - expected) < 0.001

    def test_custom_weights(self):
        scores = {
            "correctness": 5,
            "completeness": 5,
            "relevance": 5,
            "citation_quality": 5,
            "hedging": 5,
        }
        custom = {
            "correctness": 1.0,
            "completeness": 0.0,
            "relevance": 0.0,
            "citation_quality": 0.0,
            "hedging": 0.0,
        }
        assert compute_composite(scores, custom) == 5.0

    def test_missing_dimension_raises(self):
        import pytest

        scores = {"correctness": 5}  # missing others
        with pytest.raises(KeyError):
            compute_composite(scores)
