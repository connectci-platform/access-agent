"""Attribution logic in src/eval/drift.py.

The module exists because a composite drop is ambiguous, so the tests are about
the ambiguous cases rather than the happy path: a world fact failing with the
corpus unchanged is a regression, not drift, and a behavioral fact failing with
the corpus changed is reported rather than guessed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.eval.drift import (
    DriftReport,
    Verdict,
    _retrieval_fingerprint,
    classify_change,
    compare_runs,
)


class TestClassifyChange:
    @pytest.mark.parametrize("kind", ["world", "snapshot"])
    def test_world_claim_failing_with_changed_retrieval_is_drift(self, kind):
        assert classify_change(kind, retrieval_changed=True) is Verdict.DRIFT

    @pytest.mark.parametrize("kind", ["world", "snapshot"])
    def test_world_claim_failing_with_unchanged_retrieval_is_regression(self, kind):
        """The retrieved world is the same, so a changed answer means we moved."""
        assert classify_change(kind, retrieval_changed=False) is Verdict.REGRESSION

    def test_behavioral_failing_with_unchanged_retrieval_is_regression(self):
        assert classify_change("behavioral", retrieval_changed=False) is Verdict.REGRESSION

    def test_behavioral_failing_with_changed_retrieval_is_reported_not_guessed(self):
        """Either the world moved and the fact is mistyped, or we regressed as the
        corpus shifted. Both readings are plausible, so neither is asserted."""
        assert classify_change("behavioral", retrieval_changed=True) is Verdict.SUSPECTED_MISTYPE

    def test_untyped_fact_cannot_be_attributed(self):
        assert classify_change(None, retrieval_changed=True) is Verdict.UNTYPED

    def test_unrecognized_kind_is_untyped_rather_than_assumed(self):
        assert classify_change("speculative", retrieval_changed=False) is Verdict.UNTYPED


class TestRetrievalFingerprint:
    def test_duration_lines_do_not_count_as_a_changed_world(self):
        """Timings differ on every run; treating them as change would call
        everything drift."""
        a = {"tool_results": "- arguments: {...}\n- duration_ms: 120\n- data: same"}
        b = {"tool_results": "- arguments: {...}\n- duration_ms: 4310\n- data: same"}
        assert _retrieval_fingerprint(a) == _retrieval_fingerprint(b)

    def test_changed_tool_content_changes_the_fingerprint(self):
        a = {"tool_results": "- data: Anvil outage posted"}
        b = {"tool_results": "- data: no current outages"}
        assert _retrieval_fingerprint(a) != _retrieval_fingerprint(b)

    def test_absent_context_is_stable(self):
        assert _retrieval_fingerprint(None) == _retrieval_fingerprint({})


def _score(question_id, *, verdicts, kinds=None, tool_results="ctx"):
    facts = [
        {"fact_id": fid, "fact_text": f"text {fid}", **({"fact_kind": k} if k else {})}
        for fid, k in (kinds or {}).items()
    ]
    return SimpleNamespace(
        question_id=question_id,
        context={
            "fact_verdicts": [{"id": fid, "verdict": v} for fid, v in verdicts.items()],
            "required_facts": facts,
            "tool_results": tool_results,
        },
    )


def _db(baseline_run, candidate_run, baseline_scores, candidate_scores):
    scores = {"base": baseline_scores, "cand": candidate_scores}
    runs = {"base": baseline_run, "cand": candidate_run}
    return SimpleNamespace(
        get_run=runs.get,
        get_scores_for_run=lambda rid: scores.get(rid, []),
    )


def _run(**kw):
    base = {"question_set": "b.yaml", "agent_commit": "abc", "llm_model": "qwen"}
    base.update(kw)
    return SimpleNamespace(**base)


class TestCompareRuns:
    def test_reports_the_case_that_was_misread_by_hand(self):
        """A snapshot fact failing as the corpus moves: drift, not regression.

        This is the incident the module was written for — a composite drop that
        read as a prompt regression and was two stale facts.
        """
        db = _db(
            _run(),
            _run(),
            [
                _score(
                    "mcp-cov-003",
                    verdicts={"40": "yes"},
                    kinds={"40": "snapshot"},
                    tool_results="Anvil outage July 18-21 postponed",
                )
            ],
            [
                _score(
                    "mcp-cov-003",
                    verdicts={"40": "no"},
                    kinds={"40": "snapshot"},
                    tool_results="no current Anvil items",
                )
            ],
        )
        report = compare_runs(db, "base", "cand")

        assert report.system_fixed
        assert len(report.changes) == 1
        change = report.changes[0]
        assert change.verdict is Verdict.DRIFT
        assert change.retrieval_changed
        assert change.fact_id == "40"
        assert report.regressions == []

    def test_a_behavioral_failure_with_a_stable_corpus_blocks(self):
        db = _db(
            _run(),
            _run(),
            [_score("q1", verdicts={"7": "yes"}, kinds={"7": "behavioral"})],
            [_score("q1", verdicts={"7": "no"}, kinds={"7": "behavioral"})],
        )
        report = compare_runs(db, "base", "cand")
        assert [c.verdict for c in report.changes] == [Verdict.REGRESSION]
        assert len(report.regressions) == 1

    def test_system_differences_are_reported_not_tolerated(self):
        """Comparing across a commit change is meaningless; say so."""
        db = _db(
            _run(agent_commit="abc"),
            _run(agent_commit="def", llm_model="other"),
            [_score("q1", verdicts={"1": "yes"}, kinds={"1": "behavioral"})],
            [_score("q1", verdicts={"1": "no"}, kinds={"1": "behavioral"})],
        )
        report = compare_runs(db, "base", "cand")
        assert not report.system_fixed
        assert any("agent_commit" in d for d in report.system_differences)
        assert any("llm_model" in d for d in report.system_differences)

    def test_fail_to_pass_is_not_a_finding(self):
        """Only pass->fail warrants attribution; a recovery needs no explanation."""
        db = _db(
            _run(),
            _run(),
            [_score("q1", verdicts={"1": "no"}, kinds={"1": "behavioral"})],
            [_score("q1", verdicts={"1": "yes"}, kinds={"1": "behavioral"})],
        )
        assert compare_runs(db, "base", "cand").changes == []

    def test_a_question_absent_from_the_baseline_is_skipped(self):
        db = _db(
            _run(),
            _run(),
            [],
            [_score("new-q", verdicts={"1": "no"}, kinds={"1": "behavioral"})],
        )
        assert compare_runs(db, "base", "cand").changes == []

    def test_unchanged_verdicts_produce_no_findings(self):
        db = _db(
            _run(),
            _run(),
            [_score("q1", verdicts={"1": "yes"}, kinds={"1": "behavioral"})],
            [_score("q1", verdicts={"1": "yes"}, kinds={"1": "behavioral"})],
        )
        assert compare_runs(db, "base", "cand").changes == []

    def test_missing_run_raises_with_the_offending_id(self):
        db = _db(None, _run(), [], [])
        with pytest.raises(ValueError, match="base"):
            compare_runs(db, "base", "cand")

    def test_kinds_fall_back_to_the_baseline_context(self):
        """A candidate run predating the fact_kind plumbing still attributes."""
        db = _db(
            _run(),
            _run(),
            [_score("q1", verdicts={"1": "yes"}, kinds={"1": "snapshot"}, tool_results="a")],
            [_score("q1", verdicts={"1": "no"}, kinds={}, tool_results="b")],
        )
        report = compare_runs(db, "base", "cand")
        assert [c.verdict for c in report.changes] == [Verdict.DRIFT]

    def test_malformed_context_does_not_raise(self):
        db = _db(
            _run(),
            _run(),
            [SimpleNamespace(question_id="q1", context={"fact_verdicts": "not-a-list"})],
            [SimpleNamespace(question_id="q1", context=None)],
        )
        assert compare_runs(db, "base", "cand").changes == []


class TestDriftReport:
    def test_counts_cover_every_verdict_even_at_zero(self):
        report = DriftReport(baseline_run_id="a", candidate_run_id="b", system_fixed=True)
        counts = report.counts()
        assert set(counts) == {v.value for v in Verdict}
        assert all(n == 0 for n in counts.values())
