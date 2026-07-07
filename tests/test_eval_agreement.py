from src.eval.agreement import weighted_kappa


def test_perfect_agreement_is_one():
    judge = [2, 1, 0, 2, 1]
    human = [2, 1, 0, 2, 1]
    assert abs(weighted_kappa(judge, human, max_value=2) - 1.0) < 1e-9


def test_excludes_none_pairs():
    # A None on either side drops the pair from the denominator.
    judge = [2, None, 0]
    human = [2, 1, 0]
    assert abs(weighted_kappa(judge, human, max_value=2) - 1.0) < 1e-9


def test_disagreement_lowers_kappa():
    judge = [2, 2, 2, 2]
    human = [0, 0, 0, 0]
    assert weighted_kappa(judge, human, max_value=2) <= 0.0


def test_linear_weighted_partial_agreement_known_value():
    # Hand-computed + cross-checked against sklearn cohen_kappa_score(weights="linear").
    # observed=0.875, expected=0.5625 -> kappa = 0.3125 / 0.4375 = 5/7.
    judge = [2, 1, 0, 1]
    human = [2, 2, 0, 1]
    assert abs(weighted_kappa(judge, human, max_value=2) - 5 / 7) < 1e-9


def test_quadratic_weight_option():
    # Same data under quadratic weighting; hand-computed observed=0.9375,
    # expected=0.6875 -> kappa = 0.25 / 0.3125 = 0.8, matching
    # sklearn cohen_kappa_score(weights="quadratic") = 0.8.
    judge = [2, 1, 0, 1]
    human = [2, 2, 0, 1]
    assert abs(weighted_kappa(judge, human, max_value=2, weight="quadratic") - 0.8) < 1e-9


def test_no_overlapping_pairs_returns_zero():
    assert weighted_kappa([None, None], [1, 2], max_value=2) == 0.0
