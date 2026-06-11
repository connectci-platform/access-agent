"""Usage logging for quarterly/annual reports.

Logs each query with classification, tools used, and timing for long-term reporting.
Stores in Postgres for unlimited retention.

No PII is stored - user IDs are hashed for anonymous tracking.
"""

import hashlib
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    inspect,
    text,
)
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

    # Capability tracking
    capability_id = Column(String(64), index=True)
    category = Column(String(32))
    was_authenticated = Column(Boolean, default=False)

    # Rating (populated by POST /api/v1/rating)
    rating = Column(String(16))  # "helpful" or "not_helpful"
    rating_feedback = Column(Text)  # optional free-text

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
            self._migrate_columns()
            self._session_factory = sessionmaker(bind=self._engine)
            self._initialized = True
            logger.info("Usage logging initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize usage logging: {e}")
            return False

    def _migrate_columns(self) -> None:
        """Add new columns to usage_logs if they don't exist.

        create_all() only creates tables, not columns. This handles
        schema evolution for existing deployments.
        """
        if self._engine is None:
            return

        existing = {c["name"] for c in inspect(self._engine).get_columns("usage_logs")}
        migrations = {
            "capability_id": "VARCHAR(64)",
            "category": "VARCHAR(32)",
            "was_authenticated": "BOOLEAN DEFAULT FALSE",
            "rating": "VARCHAR(16)",
            "rating_feedback": "TEXT",
        }

        with self._engine.begin() as conn:
            for col_name, col_type in migrations.items():
                if col_name not in existing:
                    conn.execute(text(f"ALTER TABLE usage_logs ADD COLUMN {col_name} {col_type}"))
                    logger.info("Added column usage_logs.%s", col_name)

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
        capability_id: str | None = None,
        category: str | None = None,
    ) -> None:
        """Log a query for reporting. User IDs are hashed (no PII stored)."""
        if not self._ensure_initialized():
            return

        if self._session_factory is None:
            return

        session = self._session_factory()
        try:
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
                capability_id=capability_id,
                category=category,
                was_authenticated=acting_user is not None,
            )
            session.add(log_entry)
            session.commit()
        except Exception as e:
            logger.error(f"Failed to log query usage: {e}")
        finally:
            session.close()

    def log_rating(
        self,
        question_id: str,
        rating: str,
        feedback: str | None = None,
        acting_user: str | None = None,
        session_id: str | None = None,
    ) -> str:
        """Attach a rating to an existing usage log entry.

        Anti-spoofing checks (per capability registry spec):
        - question_id must exist in usage_logs
        - One rating per query (409 if already rated)
        - Ownership: authenticated users must match user_hash;
          anonymous users must match session_id
        - Time window: only within 24h of the original query

        Returns: "ok", "not_found", "already_rated", "forbidden", "expired", or "error".
        """
        if not self._ensure_initialized():
            return "error"

        if self._session_factory is None:
            return "error"

        try:
            session = self._session_factory()
            entry = session.query(UsageLog).filter_by(question_id=question_id).first()
            if not entry:
                logger.warning("Rating for unknown question_id: %s", question_id)
                session.close()
                return "not_found"

            # Already rated — one rating per query
            if entry.rating is not None:
                logger.warning("Duplicate rating for question_id: %s", question_id)
                session.close()
                return "already_rated"

            # Time window — 24h from original query. entry.timestamp is naive
            # UTC (the column default is datetime.utcnow), so compare naive.
            if entry.timestamp:
                cutoff = datetime.now(tz=UTC).replace(tzinfo=None) - timedelta(hours=24)
                if entry.timestamp < cutoff:
                    logger.warning("Expired rating for question_id: %s", question_id)
                    session.close()
                    return "expired"

            # Ownership check
            if acting_user:
                # Authenticated: user_hash must match
                caller_hash = self._hash_user(acting_user)
                if entry.user_hash and entry.user_hash != caller_hash:
                    logger.warning("Ownership mismatch for question_id: %s", question_id)
                    session.close()
                    return "forbidden"
            # Anonymous: session_id must match
            elif session_id and entry.session_id and entry.session_id != session_id:
                logger.warning("Session mismatch for question_id: %s", question_id)
                session.close()
                return "forbidden"

            entry.rating = rating
            entry.rating_feedback = feedback
            session.commit()
            session.close()
            return "ok"
        except Exception as e:
            logger.error(f"Failed to log rating: {e}")
            return "error"


# Global instance
_usage_logger: UsageLogger | None = None


def get_usage_logger() -> UsageLogger:
    """Get the global usage logger instance."""
    global _usage_logger
    if _usage_logger is None:
        _usage_logger = UsageLogger()
    return _usage_logger
