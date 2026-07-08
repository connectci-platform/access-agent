"""Weighted (ordinal) Cohen's kappa for judge-vs-human agreement (rubric-v2 §Scale)."""


def weighted_kappa(
    rater_a: list[int | None],
    rater_b: list[int | None],
    max_value: int,
    weight: str = "linear",
) -> float:
    """Linear-weighted Cohen's kappa over an ordinal 0..max_value scale.

    Pairs where either rater is None (e.g. specificity N/A) are excluded.
    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative below chance.
    """
    pairs = [
        (a, b) for a, b in zip(rater_a, rater_b, strict=True) if a is not None and b is not None
    ]
    if not pairs:
        return 0.0
    n = len(pairs)
    k = max_value + 1

    def w(i: int, j: int) -> float:
        if weight == "quadratic":
            return 1 - ((i - j) ** 2) / ((k - 1) ** 2)
        return 1 - abs(i - j) / (k - 1)

    observed = sum(w(a, b) for a, b in pairs) / n

    a_counts = [sum(1 for a, _ in pairs if a == i) for i in range(k)]
    b_counts = [sum(1 for _, b in pairs if b == j) for j in range(k)]
    expected = sum(
        w(i, j) * (a_counts[i] / n) * (b_counts[j] / n) for i in range(k) for j in range(k)
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)
