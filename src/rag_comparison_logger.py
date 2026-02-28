"""Dual-RAG comparison logging for A.2 evaluation.

Logs side-by-side results from UKY document RAG and pgvector Q&A-pair RAG
for the same query. Used to compare backend performance in Project A.3.

No PII is stored. Follows the same pattern as usage_logger.py.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Session, declarative_base, sessionmaker

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

from .config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()


class RAGComparisonLog(Base):  # type: ignore[valid-type,misc]
    """Side-by-side RAG comparison log entry."""

    __tablename__ = "rag_comparison_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Query context
    session_id = Column(String(100), index=True)
    question_id = Column(String(100), index=True)
    query_text = Column(Text, nullable=False)
    expanded_query = Column(Text)
    query_type = Column(String(50), index=True)  # static, combined
    rag_endpoint = Column(String(20))  # general, xdmod

    # UKY result
    uky_response = Column(Text)
    uky_duration_ms = Column(Float)
    uky_error = Column(Text)

    # pgvector result
    pgvector_matches = Column(JSONB)  # [{id, question, answer, similarity_score, domain, entity_id}]
    pgvector_best_score = Column(Float)
    pgvector_match_count = Column(Integer)
    pgvector_duration_ms = Column(Float)
    pgvector_error = Column(Text)

    # Which backend served the user-facing answer
    served_by = Column(String(30), index=True)  # uky_general, uky_xdmod, pgvector, none
    served_answer_length = Column(Integer)


class RAGComparisonLogger:
    """Logger for dual-RAG comparison results."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True

        if not settings.DATABASE_URL:
            logger.warning("DATABASE_URL not set, RAG comparison logging disabled")
            return False

        try:
            db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
            self._engine = create_engine(db_url)
            Base.metadata.create_all(self._engine)
            self._session_factory = sessionmaker(bind=self._engine)
            self._initialized = True
            logger.info("RAG comparison logging initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG comparison logging: {e}")
            return False

    def log_comparison(
        self,
        *,
        query_text: str,
        expanded_query: str,
        session_id: str,
        question_id: str,
        query_type: str,
        rag_endpoint: str,
        uky_response: str | None = None,
        uky_duration_ms: float | None = None,
        uky_error: str | None = None,
        pgvector_matches: list[dict] | None = None,
        pgvector_best_score: float | None = None,
        pgvector_match_count: int | None = None,
        pgvector_duration_ms: float | None = None,
        pgvector_error: str | None = None,
        served_by: str = "none",
        served_answer_length: int | None = None,
    ) -> None:
        """Log a dual-RAG comparison result.

        All parameters are keyword-only to prevent positional mistakes.
        Errors are logged but never raised — this must not block the response.
        """
        if not self._ensure_initialized():
            return

        if self._session_factory is None:
            return

        try:
            session = self._session_factory()
            entry = RAGComparisonLog(
                query_text=query_text,
                expanded_query=expanded_query,
                session_id=session_id,
                question_id=question_id,
                query_type=query_type,
                rag_endpoint=rag_endpoint,
                uky_response=uky_response,
                uky_duration_ms=uky_duration_ms,
                uky_error=uky_error,
                pgvector_matches=pgvector_matches,
                pgvector_best_score=pgvector_best_score,
                pgvector_match_count=pgvector_match_count,
                pgvector_duration_ms=pgvector_duration_ms,
                pgvector_error=pgvector_error,
                served_by=served_by,
                served_answer_length=served_answer_length,
            )
            session.add(entry)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Failed to log RAG comparison: {e}")


# Global instance
_rag_comparison_logger: RAGComparisonLogger | None = None


def get_rag_comparison_logger() -> RAGComparisonLogger:
    """Get the global RAG comparison logger instance."""
    global _rag_comparison_logger
    if _rag_comparison_logger is None:
        _rag_comparison_logger = RAGComparisonLogger()
    return _rag_comparison_logger
