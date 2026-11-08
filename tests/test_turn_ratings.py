"""Answer ratings for the full-screen client (D8-3022), at the storage layer.

A rating is keyed by (thread, answer message id) in turn_ratings and mirrored
onto the turn_reports row the dashboard reads. Either can be written first:
the report lands when the run's stream ends, and a fast click can beat it.
"""

from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import sessionmaker

from src.turn_reporter import (
    TurnRating,
    TurnReport,
    TurnReportBase,
    TurnReporter,
    _assemble_turn_report,
    turn_answer_ids,
)

THREAD = "11111111-2222-3333-4444-555555555555"


def _reporter() -> TurnReporter:
    r = TurnReporter()
    r._engine = create_engine("sqlite:///:memory:")
    TurnReportBase.metadata.create_all(r._engine)
    r._session_factory = sessionmaker(bind=r._engine)
    r._initialized = True
    return r


def _broken_reporter() -> TurnReporter:
    """Initialized, but with no tables, so every query raises."""
    r = TurnReporter()
    r._engine = create_engine("sqlite:///:memory:")
    r._session_factory = sessionmaker(bind=r._engine)
    r._initialized = True
    return r


def _report(r: TurnReporter, *, message_id: str | None, session_id: str = THREAD) -> None:
    s = r._session_factory()
    s.add(TurnReport(session_id=session_id, query_text="q", message_id=message_id))
    s.commit()
    s.close()


def _report_row(r: TurnReporter, message_id: str) -> TurnReport:
    s = r._session_factory()
    row = s.query(TurnReport).filter_by(message_id=message_id).one()
    s.close()
    return row


def _rating_rows(r: TurnReporter) -> list[TurnRating]:
    s = r._session_factory()
    rows = s.query(TurnRating).all()
    s.close()
    return rows


class TestTurnAnswerIds:
    def test_the_last_text_answer_of_each_turn(self):
        messages = [
            HumanMessage(content="first", id="h1"),
            AIMessage(content="", id="a-call", tool_calls=[{"name": "t", "args": {}, "id": "c"}]),
            ToolMessage(content="out", tool_call_id="c", id="t1"),
            AIMessage(content="Let me check.", id="a-interim"),
            AIMessage(content="Here it is.", id="a1"),
            HumanMessage(content="second", id="h2"),
            AIMessage(content=[{"type": "text", "text": "Again."}], id="a2"),
        ]
        assert turn_answer_ids(messages) == ["a1", "a2"]

    def test_accepts_serialized_messages(self):
        messages = [
            {"type": "human", "content": "q", "id": "h1"},
            {"type": "ai", "content": "answer", "id": "a1"},
        ]
        assert turn_answer_ids(messages) == ["a1"]

    def test_turns_without_a_usable_answer_are_skipped(self):
        messages = [
            AIMessage(content="before any question", id="a0"),
            HumanMessage(content="q1", id="h1"),
            AIMessage(content="   ", id="blank"),
            HumanMessage(content="q2", id="h2"),
            AIMessage(content="no id"),
            HumanMessage(content="q3", id="h3"),
            AIMessage(content=[{"type": "image"}], id="pic"),
        ]
        assert turn_answer_ids(messages) == []

    def test_non_list_is_empty(self):
        assert turn_answer_ids(None) == []

    def test_content_that_is_not_text_is_no_answer(self):
        messages = [{"type": "human", "id": "h1"}, {"type": "ai", "content": None, "id": "a1"}]
        assert turn_answer_ids(messages) == []


class TestReportRecordsItsAnswer:
    def _assemble(self, messages):
        report, _ = _assemble_turn_report(
            final_state={"messages": messages},
            session_id=THREAD,
            turn_index=1,
            question_id="run-1",
            query_text="q",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        return report

    def test_message_id_is_this_turns_answer(self):
        messages = [
            HumanMessage(content="q1", id="h1"),
            AIMessage(content="old", id="a1"),
            HumanMessage(content="q2", id="h2"),
            AIMessage(content="new", id="a2"),
        ]
        assert self._assemble(messages)["message_id"] == "a2"

    def test_a_failed_turn_never_claims_the_previous_answer(self):
        messages = [
            HumanMessage(content="q1", id="h1"),
            AIMessage(content="old", id="a1"),
            HumanMessage(content="q2", id="h2"),
        ]
        assert self._assemble(messages)["message_id"] is None

    def test_an_update_with_only_this_turns_messages(self):
        """A node update carries what the turn added, without its question."""
        messages = [AIMessage(content="new", id="a2")]
        assert self._assemble(messages)["message_id"] == "a2"

    def test_no_messages(self):
        assert self._assemble(None)["message_id"] is None


class TestRateAnswer:
    def test_rating_a_reported_turn_sets_the_report(self):
        r = _reporter()
        _report(r, message_id="a1")

        assert r.rate_answer(THREAD, "a1", "not_helpful", "wrong link") is True

        row = _report_row(r, "a1")
        assert (row.rating, row.rating_feedback) == ("not_helpful", "wrong link")

    def test_changing_a_rating_keeps_one_row(self):
        r = _reporter()
        _report(r, message_id="a1")

        r.rate_answer(THREAD, "a1", "not_helpful", "wrong")
        r.rate_answer(THREAD, "a1", "not_helpful", "wrong")
        r.rate_answer(THREAD, "a1", "helpful", None)

        rows = _rating_rows(r)
        assert len(rows) == 1
        assert (rows[0].rating, rows[0].feedback) == ("helpful", None)
        row = _report_row(r, "a1")
        assert (row.rating, row.rating_feedback) == ("helpful", None)

    def test_only_that_threads_answer_is_touched(self):
        r = _reporter()
        _report(r, message_id="a1")
        _report(r, message_id="a1", session_id="other-thread")

        r.rate_answer(THREAD, "a1", "helpful", None)

        s = r._session_factory()
        other = s.query(TurnReport).filter_by(session_id="other-thread").one()
        s.close()
        assert other.rating is None

    def test_a_rating_before_the_report_lands_when_it_is_written(self):
        r = _reporter()

        assert r.rate_answer(THREAD, "a1", "not_helpful", "stale") is True
        r.log_turn_report(
            final_state={
                "messages": [HumanMessage(content="q", id="h1"), AIMessage(content="x", id="a1")]
            },
            session_id=THREAD,
            turn_index=1,
            question_id="run-1",
            query_text="q",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )

        row = _report_row(r, "a1")
        assert (row.rating, row.rating_feedback) == ("not_helpful", "stale")

    def test_a_report_with_nothing_pending_stays_unrated(self):
        r = _reporter()
        r.log_turn_report(
            final_state={
                "messages": [HumanMessage(content="q", id="h1"), AIMessage(content="x", id="a1")]
            },
            session_id=THREAD,
            turn_index=1,
            question_id="run-1",
            query_text="q",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        assert _report_row(r, "a1").rating is None

    def test_uninitialized_is_false(self):
        r = TurnReporter()
        r._ensure_initialized = lambda: False  # type: ignore[method-assign]
        assert r.rate_answer(THREAD, "a1", "helpful", None) is False

    def test_db_error_is_false(self):
        assert _broken_reporter().rate_answer(THREAD, "a1", "helpful", None) is False


class TestRatingsForThread:
    def test_returns_this_threads_ratings(self):
        r = _reporter()
        r.rate_answer(THREAD, "a1", "helpful", None)
        r.rate_answer(THREAD, "a2", "not_helpful", "secret comment")
        r.rate_answer("other-thread", "a3", "helpful", None)

        assert r.ratings_for_thread(THREAD) == {"a1": "helpful", "a2": "not_helpful"}

    def test_uninitialized_is_none(self):
        r = TurnReporter()
        r._ensure_initialized = lambda: False  # type: ignore[method-assign]
        assert r.ratings_for_thread(THREAD) is None

    def test_db_error_is_none(self):
        assert _broken_reporter().ratings_for_thread(THREAD) is None


class TestPendingRatingFailure:
    def test_a_failed_pending_lookup_still_keeps_the_report(self):
        r = _reporter()
        with r._engine.begin() as conn:
            conn.execute(text("DROP TABLE turn_ratings"))
        r.log_turn_report(
            final_state={
                "messages": [HumanMessage(content="q", id="h1"), AIMessage(content="x", id="a1")]
            },
            session_id=THREAD,
            turn_index=1,
            question_id="run-1",
            query_text="q",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        assert _report_row(r, "a1").rating is None

    def test_a_failed_report_write_looks_for_no_rating(self):
        r = _broken_reporter()
        r._copy_rating_to_report = MagicMock()  # type: ignore[method-assign]
        r.log_turn_report(
            final_state={
                "messages": [HumanMessage(content="q", id="h1"), AIMessage(content="x", id="a1")]
            },
            session_id=THREAD,
            turn_index=1,
            question_id="run-1",
            query_text="q",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        r._copy_rating_to_report.assert_not_called()


class TestMigration:
    def test_message_id_column_and_index_added_to_an_existing_table(self):
        r = TurnReporter()
        r._engine = create_engine("sqlite:///:memory:")
        TurnReportBase.metadata.create_all(r._engine)
        with r._engine.begin() as conn:
            conn.execute(text("DROP INDEX IF EXISTS ix_turn_reports_message_id"))
            conn.execute(text("ALTER TABLE turn_reports DROP COLUMN message_id"))
        r._migrate_columns()
        insp = inspect(r._engine)
        assert "message_id" in {c["name"] for c in insp.get_columns("turn_reports")}
        names = {ix["name"] for ix in insp.get_indexes("turn_reports")}
        assert "ix_turn_reports_message_id" in names

    def test_a_failed_message_id_index_leaves_reporting_up(self, caplog):
        r = TurnReporter()
        r._engine = create_engine("sqlite:///:memory:")
        TurnReportBase.metadata.create_all(r._engine)

        @event.listens_for(r._engine, "before_cursor_execute")
        def _refuse(conn, cursor, statement, *args):
            if "ix_turn_reports_message_id" in statement or "battery_run_id" in statement:
                raise RuntimeError("no privilege")

        r._migrate_columns()

        assert "Failed to ensure message_id index" in caplog.text
        assert "Failed to ensure battery_run_id index" in caplog.text

    def test_ratings_table_is_created(self):
        engine = create_engine("sqlite:///:memory:")
        TurnReportBase.metadata.create_all(engine)
        cols = {c["name"] for c in inspect(engine).get_columns("turn_ratings")}
        assert {"thread_id", "message_id", "rating", "feedback", "updated_at"} <= cols


class TestRatingRobustness:
    """Findings from review: the save never fails on the copy, never on a
    race, and the copy never leaves an older rating on the report."""

    def test_a_failed_copy_still_reports_the_rating_saved(self, caplog):
        r = _reporter()

        def _boom(*args, **kwargs):
            raise RuntimeError("copy failed: secret words")

        r._copy_rating_to_report = _boom  # type: ignore[method-assign]

        assert r.rate_answer(THREAD, "a1", "not_helpful", "secret words") is True
        assert [(row.rating, row.feedback) for row in _rating_rows(r)] == [
            ("not_helpful", "secret words")
        ]
        assert "secret words" not in caplog.text

    def test_two_first_ratings_at_once_both_save(self, tmp_path):
        """The other writer's row lands between this one's start and its insert."""
        url = f"sqlite:///{tmp_path / 'race.db'}"
        r = TurnReporter()
        r._engine = create_engine(url)
        TurnReportBase.metadata.create_all(r._engine)
        r._session_factory = sessionmaker(bind=r._engine)
        r._initialized = True
        other = create_engine(url)
        raced = []

        @event.listens_for(r._engine, "before_cursor_execute")
        def _other_writer_first(conn, cursor, statement, *args):
            if statement.lstrip().upper().startswith("INSERT INTO TURN_RATINGS") and not raced:
                raced.append(True)
                with other.begin() as c:
                    c.execute(
                        text(
                            "INSERT INTO turn_ratings (thread_id, message_id, rating, updated_at) "
                            "VALUES (:t, 'a1', 'helpful', CURRENT_TIMESTAMP)"
                        ),
                        {"t": THREAD},
                    )

        assert r.rate_answer(THREAD, "a1", "not_helpful", "late") is True
        assert raced
        assert [(row.rating, row.feedback) for row in _rating_rows(r)] == [("not_helpful", "late")]

    def test_the_copy_is_one_update_from_the_ratings(self):
        r = _reporter()
        _report(r, message_id="a1")
        r.rate_answer(THREAD, "a1", "helpful", None)
        seen: list[str] = []

        @event.listens_for(r._engine, "before_cursor_execute")
        def _record(conn, cursor, statement, *args):
            seen.append(" ".join(statement.split()).upper())

        s = r._session_factory()
        r._copy_rating_to_report(s, THREAD, "a1")
        s.close()

        statements = [st for st in seen if not st.startswith(("BEGIN", "COMMIT"))]
        assert len(statements) == 1
        assert statements[0].startswith("UPDATE TURN_REPORTS")
        assert "FROM TURN_RATINGS" in statements[0]
        assert _report_row(r, "a1").rating == "helpful"


class TestRatingErrorsLogNoComment:
    def test_update_rating_logs_only_the_error_type(self, caplog, monkeypatch):
        r = _reporter()
        _report(r, message_id="a1")
        s = r._session_factory()
        s.query(TurnReport).update({"question_id": "q-1"})
        s.commit()
        s.close()
        session = r._session_factory()

        def _boom():
            raise RuntimeError("[parameters: ('private comment',)]")

        monkeypatch.setattr(session, "commit", _boom)
        monkeypatch.setattr(r, "_session_factory", lambda: session)

        r.update_rating(question_id="q-1", rating="not_helpful", feedback="private comment")

        assert "Failed to update turn report rating (RuntimeError)" in caplog.text
        assert "private comment" not in caplog.text


def test_postgres_gets_its_own_upsert():
    from types import SimpleNamespace

    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from src.turn_reporter import _dialect_insert

    session = SimpleNamespace(
        get_bind=lambda: SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
    )
    assert _dialect_insert(session) is pg_insert  # type: ignore[arg-type]
