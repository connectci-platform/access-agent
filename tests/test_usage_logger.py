"""Tests for UsageLogger.log_rating.

Uses an in-memory SQLite database wired directly into UsageLogger (same
pattern as TestWrite._reporter in test_turn_reporter.py).  The usage_logs
table is created with raw DDL because the ORM model uses PostgreSQL-specific
JSONB, which SQLite cannot compile.
"""

from datetime import UTC, datetime

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.usage_logger import UsageLogger


def _naive_utcnow() -> datetime:
    """Return a naive UTC datetime (matches the naive column default in UsageLog)."""
    return datetime.now(tz=UTC).replace(tzinfo=None)


def _make_logger():
    """Return a UsageLogger backed by an in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE usage_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    session_id VARCHAR(100),
                    question_id VARCHAR(100),
                    query_text TEXT NOT NULL,
                    query_type VARCHAR(50),
                    topic VARCHAR(100),
                    subtopic VARCHAR(100),
                    confidence VARCHAR(20),
                    tools_used TEXT,
                    tool_count INTEGER DEFAULT 0,
                    duration_ms FLOAT,
                    user_hash VARCHAR(16),
                    capability_id VARCHAR(64),
                    category VARCHAR(32),
                    was_authenticated BOOLEAN DEFAULT 0,
                    rating VARCHAR(16),
                    rating_feedback TEXT,
                    response_length INTEGER,
                    success VARCHAR(10) DEFAULT 'true'
                )
            """)
        )
    sf = sessionmaker(bind=engine)
    ul = UsageLogger()
    ul._engine = engine
    ul._session_factory = sf
    ul._initialized = True
    return ul, sf


class TestLogRating:
    def test_log_rating_within_window_succeeds(self):
        # entry.timestamp is naive UTC; the cutoff comparison must not raise
        # (regression: naive-vs-aware TypeError made every rating fail).
        ul, sf = _make_logger()
        s = sf()
        s.execute(
            text(
                "INSERT INTO usage_logs (question_id, query_text, timestamp, session_id)"
                " VALUES ('q-tz', 'x', :ts, 's-1')"
            ),
            {"ts": _naive_utcnow()},
        )
        s.commit()
        s.close()

        result = ul.log_rating(question_id="q-tz", rating="helpful", session_id="s-1")
        assert result == "ok"

    def test_log_rating_not_found(self):
        ul, _ = _make_logger()
        result = ul.log_rating(question_id="missing", rating="helpful", session_id="s-1")
        assert result == "not_found"

    def test_log_rating_already_rated(self):
        ul, sf = _make_logger()
        s = sf()
        s.execute(
            text(
                "INSERT INTO usage_logs (question_id, query_text, timestamp, session_id, rating)"
                " VALUES ('q-dup', 'x', :ts, 's-1', 'helpful')"
            ),
            {"ts": _naive_utcnow()},
        )
        s.commit()
        s.close()

        result = ul.log_rating(question_id="q-dup", rating="helpful", session_id="s-1")
        assert result == "already_rated"

    def test_log_rating_session_mismatch_is_forbidden(self):
        ul, sf = _make_logger()
        s = sf()
        s.execute(
            text(
                "INSERT INTO usage_logs (question_id, query_text, timestamp, session_id)"
                " VALUES ('q-own', 'x', :ts, 's-owner')"
            ),
            {"ts": _naive_utcnow()},
        )
        s.commit()
        s.close()

        result = ul.log_rating(question_id="q-own", rating="helpful", session_id="s-other")
        assert result == "forbidden"
