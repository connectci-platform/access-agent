"""Database operations for the eval pipeline."""

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import EvalBase, EvalRun, EvalScore

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class EvalDB:
    def __init__(self, database_url: str) -> None:
        # Support both PostgreSQL and SQLite (for tests)
        if database_url.startswith("postgresql://"):
            db_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        else:
            db_url = database_url
        self._engine: Engine = create_engine(db_url)
        EvalBase.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)
        logger.info("Eval database initialized")

    def create_run(self, **kwargs: Any) -> EvalRun:
        with self._session_factory() as session:
            run = EvalRun(**kwargs)
            session.add(run)
            session.commit()
            session.refresh(run)
            session.expunge(run)
            return run

    def add_score(self, **kwargs: Any) -> EvalScore:
        """Add a score, skipping if a duplicate already exists for this run/question/source/reviewer.

        Uses application-level check-then-insert for portability (works on SQLite in tests).
        In production PostgreSQL, the partial unique indexes on eval_scores provide hard
        guarantees even under concurrency. For single-writer eval jobs this is sufficient.
        """
        with self._session_factory() as session:
            # Check for existing score to prevent duplicates on re-runs
            run_id = kwargs.get("run_id")
            question_id = kwargs.get("question_id")
            source = kwargs.get("source")
            reviewer_id = kwargs.get("reviewer_id")

            existing: EvalScore | None = (
                session.query(EvalScore)
                .filter_by(
                    run_id=run_id, question_id=question_id, source=source, reviewer_id=reviewer_id
                )
                .first()
            )
            if existing:
                logger.debug(
                    f"Score already exists for {question_id}/{source}/{reviewer_id}, skipping"
                )
                session.expunge(existing)
                return existing

            score = EvalScore(**kwargs)
            session.add(score)
            session.commit()
            session.refresh(score)
            session.expunge(score)
            return score

    def update_run_summary(
        self, run_id: str, scores_summary: dict[str, Any], composite_score: float
    ) -> None:
        with self._session_factory() as session:
            run: EvalRun | None = session.query(EvalRun).filter_by(id=run_id).first()
            if run:
                run.scores_summary = scores_summary  # type: ignore[assignment]
                run.composite_score = composite_score  # type: ignore[assignment]
                session.commit()

    def get_run(self, run_id: str) -> EvalRun | None:
        with self._session_factory() as session:
            run: EvalRun | None = session.query(EvalRun).filter_by(id=run_id).first()
            if run:
                session.expunge(run)
            return run

    def get_scores_for_run(self, run_id: str) -> list[EvalScore]:
        with self._session_factory() as session:
            scores: list[EvalScore] = (
                session.query(EvalScore)
                .filter_by(run_id=run_id)
                .order_by(EvalScore.created_at)
                .all()
            )
            for s in scores:
                session.expunge(s)
            return scores

    def query_scores(self, since: Any = None, resource: str | None = None) -> list[EvalScore]:
        """Query scores with optional filters. Used by reports."""
        with self._session_factory() as session:
            q = session.query(EvalScore).filter(
                EvalScore.source.in_(["judge", "human"]),
            )
            if since:
                q = q.filter(EvalScore.created_at >= since)
            if resource:
                q = q.filter(EvalScore.question_text.ilike(f"%{resource}%"))
            scores: list[EvalScore] = q.all()
            for s in scores:
                session.expunge(s)
            return scores

    def execute_readonly_sql(self, sql: str) -> tuple[list[Any], list[str]]:
        """Execute a read-only SQL query. Returns (rows, column_names). Used by ask."""
        from sqlalchemy import text as sa_text

        with self._session_factory() as session:
            session.execute(sa_text("SET TRANSACTION READ ONLY"))
            result = session.execute(sa_text(sql))
            rows: list[Any] = list(result.fetchall())
            columns = list(result.keys())
            return rows, columns
