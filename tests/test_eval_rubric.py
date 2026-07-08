from src.eval.rubric import (
    DEFAULT_WEIGHTS,
    DIMENSION_MAX,
    DIMENSION_NAMES,
    compute_composite,
    flatten_required_facts,
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


# --- Task 12: stable fact_id keying ---


def test_flatten_carries_stable_fact_id_from_db_dict():
    # A DB-shaped fact dict (fact_id + fact_text) keeps its stable id, not F{n}.
    facts = [{"fact_id": 83, "fact_text": "Anvil has A100s"}]
    flat = flatten_required_facts(facts)
    assert flat == [("83", "Anvil has A100s")]


def test_flatten_fact_id_accepts_text_key_too():
    # Defensive: some callers use "text" instead of "fact_text".
    facts = [{"fact_id": "fx-abc", "text": "Bridges-2 has GPUs"}]
    flat = flatten_required_facts(facts)
    assert flat == [("fx-abc", "Bridges-2 has GPUs")]


def test_flatten_keeps_positional_fallbacks_for_legacy_shapes():
    # Plain strings → F{n}; {heading, items} → F{n}. Stable ids only when provided.
    facts = [
        "Delta has A100 GPUs",
        {"heading": "Software:", "items": ["GCC", "OpenMPI"]},
    ]
    flat = flatten_required_facts(facts)
    assert flat == [
        ("F1", "Delta has A100 GPUs"),
        ("F2", "Software: GCC"),
        ("F3", "Software: OpenMPI"),
    ]


def test_flatten_mixed_stable_and_positional():
    # A stable-id dict does not consume a positional counter slot.
    facts = [
        {"fact_id": 7, "fact_text": "Stable one"},
        "Legacy string",
    ]
    flat = flatten_required_facts(facts)
    assert flat == [("7", "Stable one"), ("F1", "Legacy string")]


def test_composite_all_none_is_zero():
    # Every dimension N/A/None → total weight 0 → composite 0.0, no divide-by-zero.
    scores = dict.fromkeys(DIMENSION_NAMES, None)
    assert compute_composite(scores) == 0.0
