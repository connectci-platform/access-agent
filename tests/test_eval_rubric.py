from src.eval.rubric import (
    DEFAULT_WEIGHTS,
    DIMENSION_MAX,
    DIMENSION_NAMES,
    compute_composite,
)


def test_v2_dimension_set():
    assert DIMENSION_NAMES == [
        "correctness",
        "specificity",
        "relevance",
        "citation_quality",
        "hedging",
    ]
    assert "completeness" not in DIMENSION_NAMES


def test_hedging_is_two_point_others_three():
    assert DIMENSION_MAX["hedging"] == 1
    assert DIMENSION_MAX["correctness"] == 2
    assert DIMENSION_MAX["specificity"] == 2
    assert DIMENSION_MAX["relevance"] == 2
    assert DIMENSION_MAX["citation_quality"] == 2


def test_weights_sum_to_one_over_v2_set():
    assert set(DEFAULT_WEIGHTS) == set(DIMENSION_NAMES)
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 1.0) < 1e-9


def test_composite_normalizes_per_dimension_to_unit_interval():
    # All-best answer → composite 1.0 (each dim at its own max normalizes to 1.0).
    scores = {
        "correctness": 2,
        "specificity": 2,
        "relevance": 2,
        "citation_quality": 2,
        "hedging": 1,
    }
    assert abs(compute_composite(scores) - 1.0) < 1e-9
    # All-worst → 0.0
    worst = dict.fromkeys(DIMENSION_NAMES, 0)
    assert compute_composite(worst) == 0.0


def test_composite_skips_na_specificity_and_renormalizes():
    # specificity N/A (None) drops its 0.25 weight; remaining weights renormalize.
    scores = {
        "correctness": 2,
        "specificity": None,
        "relevance": 2,
        "citation_quality": 2,
        "hedging": 1,
    }
    # All non-N/A dims at max → still 1.0 after renormalization.
    assert abs(compute_composite(scores) - 1.0) < 1e-9
