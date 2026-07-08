"""Tests for eval database models and operations."""

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.eval.models import EvalBase, EvalRun, EvalScore


def test_eval_scores_has_v2_columns_not_completeness():
    cols = {c.name for c in inspect(EvalScore).columns}
    assert "specificity" in cols
    assert "specificity_na" in cols
    assert "answerable" in cols
    assert "completeness" not in cols  # dropped, not renamed
    # context_completeness is a DIFFERENT column and must stay.
    assert "context_completeness" in cols


class TestEvalModels:
    def setup_method(self):
        self.engine = create_engine("sqlite:///:memory:")
        EvalBase.metadata.create_all(self.engine)
        self.SessionFactory = sessionmaker(bind=self.engine)

    def test_create_eval_run(self):
        session = self.SessionFactory()
        run = EvalRun(
            run_type="pre_production",
            agent_commit="abc123",
            agent_branch="feature/test",
            llm_model="gpt-4o-mini",
            question_set="friendly_battery.json",
            question_count=50,
            composite_score=4.2,
        )
        session.add(run)
        session.commit()

        result = session.query(EvalRun).first()
        assert result is not None
        assert result.run_type == "pre_production"
        assert result.agent_branch == "feature/test"
        assert result.question_count == 50
        assert result.composite_score == 4.2
        assert result.id is not None
        assert result.created_at is not None
        session.close()

    def test_create_eval_score(self):
        session = self.SessionFactory()
        run = EvalRun(
            run_type="pre_production",
            agent_commit="abc123",
            llm_model="gpt-4o-mini",
            question_set="test",
            question_count=1,
        )
        session.add(run)
        session.flush()

        score = EvalScore(
            run_id=run.id,
            question_id="q-001",
            source="judge",
            question_text="What GPUs does Delta have?",
            answer_text="Delta has NVIDIA A100 GPUs.",
            correctness=2,
            specificity=2,
            specificity_na=False,
            answerable=True,
            relevance=2,
            citation_quality=2,
            hedging=1,
            composite_score=0.95,
        )
        session.add(score)
        session.commit()

        result = session.query(EvalScore).first()
        assert result is not None
        assert result.source == "judge"
        assert result.correctness == 2
        assert result.composite_score == 0.95
        assert result.run_id == run.id
        session.close()

    def test_score_valid_sources(self):
        assert "judge" in ("judge", "human", "judge_error", "skipped")
        assert "human" in ("judge", "human", "judge_error", "skipped")
