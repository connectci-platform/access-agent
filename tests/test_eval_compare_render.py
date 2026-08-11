"""print_comparison annotates cross-mode compares on BOTH sections.

A multiturn run and a single-turn run disagree twice over: the composite is a
macro Fair-only mean vs a micro all-rows mean, and the per-dimension means cover
different populations (Fair-judged turns only vs every judged row, including
answerable=False ones). Same-mode compares must stay unannotated.
"""

from src.eval.report import (
    COMPOSITE_INCOMMENSURABLE_NOTE,
    DIMENSION_POPULATION_NOTE,
    print_comparison,
)

RUN_A = {
    "run_id": "run-a",
    "agent_branch": "main",
    "composite_score": 0.60,
    "per_dimension": {"correctness": 1.5, "specificity": 1.0},
}
RUN_B = {
    "run_id": "run-b",
    "agent_branch": "feat",
    "composite_score": 0.72,
    "per_dimension": {"correctness": 1.8, "specificity": 1.2},
}


def test_cross_mode_compare_annotates_dimension_and_composite(capsys):
    print_comparison(RUN_A, RUN_B, composite_commensurable=False)
    out = capsys.readouterr().out

    assert DIMENSION_POPULATION_NOTE in out
    assert COMPOSITE_INCOMMENSURABLE_NOTE in out
    # The composite delta is suppressed, but the dimension rows still show deltas —
    # they carry directional signal, they are just not like-for-like.
    assert "n/a" in out
    assert "0.30" in out  # correctness delta still rendered


def test_same_mode_compare_carries_no_population_note(capsys):
    print_comparison(RUN_A, RUN_B, composite_commensurable=True)
    out = capsys.readouterr().out

    assert DIMENSION_POPULATION_NOTE not in out
    assert COMPOSITE_INCOMMENSURABLE_NOTE not in out
    # Both sections show real deltas: the dimension rows and a numeric composite.
    assert "0.30" in out
    assert "0.12" in out
    assert "n/a" not in out


def test_default_is_commensurable_and_unannotated(capsys):
    print_comparison(RUN_A, RUN_B)
    out = capsys.readouterr().out
    assert DIMENSION_POPULATION_NOTE not in out
    assert COMPOSITE_INCOMMENSURABLE_NOTE not in out
