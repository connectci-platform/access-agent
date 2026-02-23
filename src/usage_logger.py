"""Usage logging for quarterly/annual reports.

Logs each query with classification, tools used, and timing for long-term reporting.
Stores in Postgres for unlimited retention.

No PII is stored - user IDs are hashed for anonymous tracking.
"""

import hashlib
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


class UsageLog(Base):  # type: ignore[valid-type,misc]
    """Usage log entry for reporting."""

    __tablename__ = "usage_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Query info
    session_id = Column(String(100), index=True)
    question_id = Column(String(100), index=True)
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), index=True)  # static, dynamic, combined

    # Classification
    topic = Column(String(100), index=True)
    subtopic = Column(String(100))
    confidence = Column(String(20))

    # Execution
    tools_used = Column(JSONB)  # List of tool names
    tool_count = Column(Integer, default=0)
    duration_ms = Column(Float)

    # Anonymous user tracking (hashed, not reversible)
    user_hash = Column(String(16), index=True)

    # Response
    response_length = Column(Integer)
    success = Column(String(10), default="true")


class UsageLogger:
    """Logger for recording query usage to Postgres."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Initialize database connection and create table if needed."""
        if self._initialized:
            return True

        if not settings.DATABASE_URL:
            logger.warning("DATABASE_URL not set, usage logging disabled")
            return False

        try:
            # Use psycopg (v3) driver — the container doesn't have psycopg2
            db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
            self._engine = create_engine(db_url)
            Base.metadata.create_all(self._engine)
            self._session_factory = sessionmaker(bind=self._engine)
            self._initialized = True
            logger.info("Usage logging initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize usage logging: {e}")
            return False

    @staticmethod
    def _hash_user(user_id: str | None) -> str | None:
        """Hash user ID for anonymous tracking. Not reversible."""
        if not user_id:
            return None
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    def log_query(
        self,
        query_text: str,
        session_id: str,
        question_id: str,
        query_type: str | None = None,
        topic: str | None = None,
        subtopic: str | None = None,
        confidence: str | None = None,
        tools_used: list[str] | None = None,
        duration_ms: float | None = None,
        response_length: int | None = None,
        acting_user: str | None = None,
        success: bool = True,
    ) -> None:
        """Log a query for reporting. User IDs are hashed (no PII stored).

        Args:
            query_text: The user's question
            session_id: Session identifier
            question_id: Question identifier
            query_type: Classification (static, dynamic, combined)
            topic: Main topic from classification
            subtopic: Subtopic from classification
            confidence: Confidence level
            tools_used: List of MCP tools called
            duration_ms: Total execution time
            response_length: Length of the response
            acting_user: User ID (will be hashed, not stored directly)
            success: Whether the query succeeded
        """
        if not self._ensure_initialized():
            return

        if self._session_factory is None:
            return

        try:
            session = self._session_factory()
            log_entry = UsageLog(
                query_text=query_text,
                session_id=session_id,
                question_id=question_id,
                query_type=query_type,
                topic=topic,
                subtopic=subtopic,
                confidence=confidence,
                tools_used=tools_used or [],
                tool_count=len(tools_used) if tools_used else 0,
                duration_ms=duration_ms,
                user_hash=self._hash_user(acting_user),
                response_length=response_length,
                success="true" if success else "false",
            )
            session.add(log_entry)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Failed to log query usage: {e}")


# Global instance
_usage_logger: UsageLogger | None = None


def get_usage_logger() -> UsageLogger:
    """Get the global usage logger instance."""
    global _usage_logger
    if _usage_logger is None:
        _usage_logger = UsageLogger()
    return _usage_logger
