"""SQLAlchemy models for eval pipeline tables."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.orm import declarative_base, relationship

# Use JSON type which maps to JSONB in PostgreSQL via SQLAlchemy's type system.
# SQLite (used in tests) does not support JSONB natively, so we use the
# portable JSON type. In production PostgreSQL this renders as JSONB.
JSONB = JSON

EvalBase = declarative_base()


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _genuuid() -> str:
    return str(uuid.uuid4())


class EvalRun(EvalBase):  # type: ignore[valid-type,misc]
    __tablename__ = "eval_runs"

    id = Column(String(36), primary_key=True, default=_genuuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    run_type = Column(String(16), nullable=False)
    agent_commit = Column(String(40))
    agent_branch = Column(String(128))
    tool_catalog = Column(JSONB)
    llm_model = Column(String(64))
    judge_model = Column(String(64))
    question_set = Column(String(128))
    question_count = Column(Integer)
    scores_summary = Column(JSONB)
    composite_score = Column(Float)
    metadata_ = Column("metadata", JSONB)

    scores = relationship("EvalScore", back_populates="run")


class EvalScore(EvalBase):  # type: ignore[valid-type,misc]
    __tablename__ = "eval_scores"

    id = Column(String(36), primary_key=True, default=_genuuid)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    run_id = Column(String(36), ForeignKey("eval_runs.id"), nullable=False, index=True)
    question_id = Column(String(64), nullable=False, index=True)
    source = Column(String(16), nullable=False)
    reviewer_id = Column(String(128))
    question_text = Column(Text)
    answer_text = Column(Text)
    context = Column(JSONB)
    context_completeness = Column(String(16))
    correctness = Column(Integer)
    completeness = Column(Integer)
    relevance = Column(Integer)
    citation_quality = Column(Integer)
    hedging = Column(Integer)
    composite_score = Column(Float)
    justifications = Column(JSONB)
    feedback = Column(Text)

    run = relationship("EvalRun", back_populates="scores")

    __table_args__ = (
        CheckConstraint(
            "source IN ('judge', 'human', 'judge_error', 'skipped')",
            name="ck_eval_scores_source",
        ),
        Index(
            "ix_eval_scores_machine_unique",
            "run_id",
            "question_id",
            "source",
            unique=True,
            postgresql_where=text("reviewer_id IS NULL"),
        ),
        Index(
            "ix_eval_scores_human_unique",
            "run_id",
            "question_id",
            "source",
            "reviewer_id",
            unique=True,
            postgresql_where=text("reviewer_id IS NOT NULL"),
        ),
    )
