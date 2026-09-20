"""Pre-flight refusal for questions that resolve to no required facts.

A fact-less question does not score zero. rubric.py renders no Required Facts
block while still telling the judge to grade correctness from per-fact verdicts,
so it falls back to plausibility and the question scores HIGHER than a graded
one. The refusal exists because that failure is silent and upward.
"""

import pytest

from src.eval.db import EvalDB
from src.eval.questions import EvalQuestion
from src.eval.scorer import preflight_fact_coverage


def _db(tmp_path) -> EvalDB:
    # No reporting schema: resolve_required_facts falls back to YAML metadata,
    # which is what these cases vary.
    return EvalDB(f"sqlite:///{tmp_path}/preflight.db")


def _q(qid: str, facts: list[str] | None, source: str | None = None) -> EvalQuestion:
    metadata: dict = {}
    if facts is not None:
        metadata["required_facts"] = facts
    if source is not None:
        metadata["source_battery"] = source
    return EvalQuestion(id=qid, question=f"question {qid}?", metadata=metadata)


def test_unscored_battery_runs_untouched(tmp_path):
    """A battery where NO question has facts is a smoke/coverage set, not a gap.

    friendly, real_user, mcp_coverage and four others author no facts at all;
    refusing them would make the CLI default unrunnable.
    """
    questions = [_q("sm-01", None), _q("sm-02", None), _q("sm-03", None)]
    preflight_fact_coverage(_db(tmp_path), questions, allow_factless=False)


def test_fully_covered_battery_passes(tmp_path):
    questions = [_q("sa-01", ["fact a"]), _q("sa-02", ["fact b"])]
    preflight_fact_coverage(_db(tmp_path), questions, allow_factless=False)


def test_gap_in_a_scored_battery_refuses(tmp_path):
    """Some questions graded, others silently not — the case worth catching."""
    questions = [_q("sa-01", ["fact a"]), _q("sa-02", None), _q("sa-03", [])]
    with pytest.raises(ValueError) as exc:
        preflight_fact_coverage(_db(tmp_path), questions, allow_factless=False)
    message = str(exc.value)
    assert "sa-02" in message
    assert "sa-03" in message
    assert "sa-01" not in message
    assert "--allow-factless" in message


def test_escape_hatch_downgrades_to_a_warning(tmp_path, caplog):
    questions = [_q("sa-01", ["fact a"]), _q("sa-02", None)]
    with caplog.at_level("WARNING"):
        preflight_fact_coverage(_db(tmp_path), questions, allow_factless=True)
    assert "sa-02" in caplog.text


def test_refusal_names_every_gap_not_just_the_first(tmp_path):
    """The message is the fix list, so a run is not a one-at-a-time bisect."""
    questions = [_q("sa-01", ["fact"])] + [_q(f"sa-{n:02d}", None) for n in range(2, 6)]
    with pytest.raises(ValueError) as exc:
        preflight_fact_coverage(_db(tmp_path), questions, allow_factless=False)
    for n in range(2, 6):
        assert f"sa-{n:02d}" in str(exc.value)


def test_mixed_source_battery_only_flags_its_scored_sources(tmp_path):
    """loop_smoke draws from four batteries — two grade, two do not.

    Its tc-* questions come from tool_coverage and carry facts; the comb-*,
    friendly-* and real-* ones come from batteries that never author facts. A
    factless question is a gap only when a sibling from the SAME source is
    graded.
    """
    questions = [
        _q("tc-01", ["fact a"], source="tool_coverage"),
        _q("tc-02", ["fact b"], source="tool_coverage"),
        _q("comb-011", None, source="combined"),
        _q("friendly-001", None, source="friendly"),
        _q("real-022", None, source="real_user"),
    ]
    preflight_fact_coverage(_db(tmp_path), questions, allow_factless=False)


def test_a_gap_within_a_scored_source_still_refuses(tmp_path):
    questions = [
        _q("tc-01", ["fact a"], source="tool_coverage"),
        _q("tc-02", None, source="tool_coverage"),
        _q("friendly-001", None, source="friendly"),
    ]
    with pytest.raises(ValueError) as exc:
        preflight_fact_coverage(_db(tmp_path), questions, allow_factless=False)
    assert "tc-02" in str(exc.value)
    assert "friendly-001" not in str(exc.value)
