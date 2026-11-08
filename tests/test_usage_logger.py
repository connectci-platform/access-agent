"""Tests for UsageLogger.log_rating.

Uses an in-memory SQLite database wired directly into UsageLogger (same
pattern as TestWrite._reporter in test_turn_reporter.py).  The usage_logs
table is created with raw DDL because the ORM model uses PostgreSQL-specific
JSONB, which SQLite cannot compile.
"""

from datetime import UTC, datetime, timedelta

import pytest
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

    def _seed(self, ul, *, user_hash=None, rating=None, age_hours=0):
        session = ul._session_factory()
        session.execute(
            text(
                "INSERT INTO usage_logs "
                "(timestamp, session_id, question_id, query_text, user_hash, rating) "
                "VALUES (:ts, 's-1', 'q-1', 'hi', :user_hash, :rating)"
            ),
            {
                "ts": _naive_utcnow() - timedelta(hours=age_hours),
                "user_hash": user_hash,
                "rating": rating,
            },
        )
        session.commit()
        session.close()

    def _rating_of(self, ul):
        session = ul._session_factory()
        row = session.execute(text("SELECT rating FROM usage_logs")).one()
        session.close()
        return row.rating

    @pytest.mark.parametrize(
        ("row_user", "acting_user", "session_id", "expected"),
        [
            pytest.param(None, None, "s-1", "ok", id="widget-anon-turn"),
            pytest.param(None, "bob@x", "s-1", "ok", id="widget-signed-in-after-anon-turn"),
            pytest.param("alice@x", "alice@x", None, "ok", id="owner-no-session"),
            pytest.param("alice@x", "alice@x", "s-other", "ok", id="owner-other-session"),
            pytest.param("alice@x", None, "s-1", "ok", id="matching-session-on-owned-row"),
            pytest.param("alice@x", None, None, "not_found", id="anon-omits-session-on-owned"),
            pytest.param("alice@x", None, "s-other", "not_found", id="anon-wrong-session-on-owned"),
            pytest.param("alice@x", "bob@x", None, "not_found", id="other-user-on-owned"),
            pytest.param("alice@x", "bob@x", "s-other", "not_found", id="other-user-wrong-session"),
            pytest.param(None, "bob@x", None, "not_found", id="signed-in-omits-session-on-anon"),
            pytest.param(None, None, None, "not_found", id="anon-omits-session-on-anon"),
            pytest.param(None, None, "s-other", "not_found", id="anon-wrong-session-on-anon"),
        ],
    )
    def test_rating_needs_the_rows_owner_or_its_session(
        self, row_user, acting_user, session_id, expected
    ):
        ul, _ = _make_logger()
        self._seed(ul, user_hash=ul._hash_user(row_user))

        result = ul.log_rating(
            question_id="q-1", rating="helpful", acting_user=acting_user, session_id=session_id
        )

        assert result == expected
        assert self._rating_of(ul) == ("helpful" if expected == "ok" else None)

    def test_a_stranger_cant_tell_a_rated_turn_from_an_unknown_one(self):
        ul, _ = _make_logger()
        self._seed(ul, rating="helpful")

        assert ul.log_rating(question_id="q-1", rating="helpful") == "not_found"

    def test_a_stranger_cant_tell_an_expired_turn_from_an_unknown_one(self):
        ul, _ = _make_logger()
        self._seed(ul, age_hours=25)

        assert ul.log_rating(question_id="q-1", rating="helpful") == "not_found"

    def test_the_owner_still_hears_about_a_duplicate_or_expired_rating(self):
        ul, _ = _make_logger()
        self._seed(ul, rating="helpful")
        assert ul.log_rating(question_id="q-1", rating="helpful", session_id="s-1") == (
            "already_rated"
        )

        ul, _ = _make_logger()
        self._seed(ul, age_hours=25)
        assert ul.log_rating(question_id="q-1", rating="helpful", session_id="s-1") == "expired"


class TestSessionOf:
    def test_returns_the_rows_session(self):
        ul, sf = _make_logger()
        s = sf()
        s.execute(
            text(
                "INSERT INTO usage_logs (session_id, question_id, query_text) "
                "VALUES ('s-1', 'q-1', 'hi')"
            )
        )
        s.commit()
        s.close()

        assert ul.session_of("q-1") == "s-1"
        assert ul.session_of("q-unknown") is None

    def test_uninitialized_is_none(self):
        ul = UsageLogger()
        ul._ensure_initialized = lambda: False  # type: ignore[method-assign]
        assert ul.session_of("q-1") is None

    def test_db_error_is_none(self):
        ul = UsageLogger()
        ul._engine = create_engine("sqlite:///:memory:")
        ul._session_factory = sessionmaker(bind=ul._engine)  # no table: the query raises
        ul._initialized = True
        assert ul.session_of("q-1") is None
