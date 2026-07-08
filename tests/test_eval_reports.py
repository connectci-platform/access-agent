"""Tests for eval report generation."""

import inspect as _inspect
from datetime import UTC, datetime, timedelta

import src.eval.compare_judge as cj
from src.eval.db import EvalDB
from src.eval.models import EvalRun
from src.eval.report import (
    generate_leadership_report,
    generate_resource_report,
    generate_team_report,
)
from src.eval.report_data import build_report_data


class _FakeScore:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_team_report_uses_unit_interval_composite():
    data = {
        "period": "x",
        "resource": None,
        "total_scored": 1,
        "human_coverage": 0.0,
        "composite_score": 0.87,
        "per_dimension": {"correctness": 2.0},
        "worst_answers": [],
        "capability_gaps": [],
        "judge_human_agreement": None,
        "total_queries": 1,
        "human_reviewed": 0,
        "previous_composite": None,
        "capability_breakdown": [],
    }
    out = generate_team_report(data)
    assert "/ 5.00" not in out
    assert "0.87" in out


def test_report_dimensions_are_v2_and_exclude_na(monkeypatch):
    scores = [
        _FakeScore(
            question_id="q1",
            run_id="r1",
            source="judge",
            composite_score=1.0,
            question_text="a",
            correctness=2,
            specificity=2,
            relevance=2,
            citation_quality=2,
            hedging=1,
            answerable=True,
        ),
        _FakeScore(
            question_id="q2",
            run_id="r1",
            source="judge",
            composite_score=1.0,
            question_text="b",
            correctness=2,
            specificity=None,
            relevance=2,
            citation_quality=2,
            hedging=1,
            answerable=True,
        ),
    ]
    # exercise the per_dimension computation directly on the score list
    # (helper extracted below), asserting specificity averages only the non-None value.
    from src.eval.report_data import _per_dimension_means

    pd = _per_dimension_means(scores)
    assert set(pd) == {"correctness", "specificity", "relevance", "citation_quality", "hedging"}
    assert pd["specificity"] == 2.0  # q2's None excluded, not counted as 0
    assert "completeness" not in pd


class TestTeamReport:
    def test_generates_markdown(self):
        data = {
            "period": "Since 2026-03-25",
            "total_scored": 50,
            "human_coverage": 0.20,
            "composite_score": 4.12,
            "per_dimension": {
                "correctness": 4.35,
                "completeness": 3.90,
                "relevance": 4.50,
                "citation_quality": 3.80,
                "hedging": 4.10,
            },
            "worst_answers": [
                {"question": "How much space on Anvil?", "composite": 2.5, "question_id": "q1"}
            ],
            "capability_gaps": [{"area": "storage", "avg_score": 2.8, "count": 5}],
            "judge_human_agreement": 0.85,
        }
        report = generate_team_report(data)
        assert "4.12" in report
        assert "correctness" in report
        assert "Anvil" in report

    def test_handles_empty(self):
        data = {
            "period": "Since 2026-04-01",
            "total_scored": 0,
            "human_coverage": 0.0,
            "composite_score": 0.0,
            "per_dimension": {},
            "worst_answers": [],
            "capability_gaps": [],
            "judge_human_agreement": None,
        }
        report = generate_team_report(data)
        assert "No scores" in report or "0" in report


class TestLeadershipReport:
    def test_generates_summary(self):
        data = {
            "period": "March 2026",
            "total_queries": 500,
            "total_scored": 450,
            "human_reviewed": 50,
            "composite_score": 4.12,
            "previous_composite": 3.95,
            "capability_breakdown": [
                {"area": "general", "score": 4.5, "count": 200},
                {"area": "resources", "score": 3.8, "count": 100},
            ],
        }
        report = generate_leadership_report(data)
        assert "4.12" in report
        assert "March 2026" in report


class TestResourceReport:
    def test_generates_for_resource(self):
        data = {
            "resource": "Delta",
            "period": "Since 2026-04-01",
            "total_scored": 30,
            "composite_score": 4.2,
            "per_dimension": {
                "correctness": 4.5,
                "completeness": 4.0,
                "relevance": 4.3,
                "citation_quality": 3.8,
                "hedging": 4.0,
            },
            "worst_answers": [
                {"question": "How to login delta?", "composite": 3.1, "question_id": "q5"}
            ],
        }
        report = generate_resource_report(data)
        assert "Delta" in report
        assert "4.2" in report


class TestReportDataDedup:
    """Test that report data correctly deduplicates across runs."""

    def setup_method(self) -> None:
        self.db = EvalDB("sqlite:///:memory:")

    def test_latest_run_wins_for_same_question(self) -> None:
        """When two runs score the same question_id, the newer run's scores are used."""
        now = datetime.now(UTC)

        # Older run scores question q1 as 3.0
        old_run = self.db.create_run(
            run_type="pre_production",
            agent_commit="aaa",
            question_set="test",
            question_count=1,
        )
        # Manually set created_at to be older
        with self.db._session_factory() as session:
            run = session.query(EvalRun).filter_by(id=old_run.id).first()
            assert run is not None
            run.created_at = now - timedelta(hours=2)
            session.commit()

        self.db.add_score(
            run_id=old_run.id,
            question_id="q1",
            source="judge",
            question_text="What is ACCESS?",
            correctness=1,
            specificity=1,
            relevance=1,
            citation_quality=1,
            hedging=1,
            composite_score=3.0,
        )

        # Newer run scores same question q1 as 5.0
        new_run = self.db.create_run(
            run_type="pre_production",
            agent_commit="bbb",
            question_set="test",
            question_count=1,
        )
        self.db.add_score(
            run_id=new_run.id,
            question_id="q1",
            source="judge",
            question_text="What is ACCESS?",
            correctness=2,
            specificity=2,
            relevance=2,
            citation_quality=2,
            hedging=1,
            composite_score=5.0,
        )

        data = build_report_data(self.db, since="7d")
        # Should use the newer run's score (5.0), not the older (3.0)
        assert data["composite_score"] == 5.0
        assert data["total_scored"] == 1  # not 2

    def test_human_preferred_over_judge_within_same_run(self) -> None:
        """When both human and judge scores exist for the same question in the same run."""
        run = self.db.create_run(
            run_type="pre_production",
            agent_commit="ccc",
            question_set="test",
            question_count=1,
        )
        # Judge scores 5.0
        self.db.add_score(
            run_id=run.id,
            question_id="q2",
            source="judge",
            question_text="How to login?",
            correctness=2,
            specificity=2,
            relevance=2,
            citation_quality=2,
            hedging=1,
            composite_score=5.0,
        )
        # Human scores 3.0
        self.db.add_score(
            run_id=run.id,
            question_id="q2",
            source="human",
            reviewer_id="drew",
            question_text="How to login?",
            correctness=1,
            specificity=1,
            relevance=1,
            citation_quality=1,
            hedging=1,
            composite_score=3.0,
        )

        data = build_report_data(self.db, since="7d")
        # Should use human score (3.0), not judge (5.0)
        assert data["composite_score"] == 3.0


def test_compare_judge_has_no_completeness_or_1_5_scale():
    src = _inspect.getsource(cj)
    assert '"completeness"' not in src
    assert "1.0-5.0" not in src
    assert '"specificity"' in src


def test_main_score_to_dict_has_no_completeness():
    import src.eval.__main__ as em

    src = _inspect.getsource(em)
    assert '"completeness": s.completeness' not in src
    assert '"specificity": s.specificity' in src


def test_print_run_summary_renders_v2(capsys):
    from src.eval.report import print_run_summary

    summary = {
        "run_id": "r1",
        "agent_branch": "b",
        "agent_commit": "c",
        "questions": 3,
        "scored": 3,
        "skipped": 0,
        "composite_score": 1.0,
        "per_dimension": {
            "correctness": 2.0,
            "specificity": 2.0,
            "relevance": 2.0,
            "citation_quality": 2.0,
            "hedging": 1.0,
        },
    }
    print_run_summary(summary)
    out = capsys.readouterr().out
    assert "/ 1.00" in out  # v2 unit-interval composite label
    assert "correctness" in out
    assert "█████" in out  # perfect score → full bar (bar scaled to dim max)
