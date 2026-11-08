"""Per-turn reporting: a denormalized read model for the reporting dashboard.

Mirrors src/usage_logger.py. Writes one turn_reports row (+ report_tool_calls
children) per completed query turn, off the response path. No PII beyond what
usage_logs already stores; user ids are hashed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    func,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .agent.domains.capabilities import WRITE_MCP_TOOL_NAMES
from .config import settings
from .llm.providers import active_model_name

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# Renders as JSONB on Postgres (GIN-indexable, supports @> containment for the
# dashboard's capability filters) and as plain TEXT-backed JSON on SQLite, which
# the tests run on (bare JSONB will not compile on SQLite).
_JSONB = JSON().with_variant(JSONB(), "postgresql")
TurnReportBase = declarative_base()


class TurnReport(TurnReportBase):  # type: ignore[valid-type,misc]
    __tablename__ = "turn_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False, index=True)

    session_id = Column(String(100), index=True)
    turn_index = Column(Integer)
    question_id = Column(String(100), index=True)
    query_text = Column(Text, nullable=False)

    origin = Column(String(16), default="real", index=True)  # real | battery | redteam
    battery_id = Column(String(64))
    battery_run_id = Column(
        String(64), index=True
    )  # run id; set when origin in ('battery', 'redteam')
    agent_version = Column(String(64), index=True)
    env = Column(String(16), index=True)
    trace_id = Column(
        String(32)
    )  # OTEL trace id of the turn's root span; display-only, never filtered
    model_id = Column(String(64))

    capabilities = Column(_JSONB)  # list[str]
    resources = Column(_JSONB)  # list[str] — RP slugs mentioned in the turn
    resource_context = Column(String(64), index=True)
    was_authenticated = Column(Boolean, default=False)
    user_hash = Column(String(16), index=True)

    success = Column(Boolean, default=True, index=True)
    duration_ms = Column(Float)
    tool_count = Column(Integer, default=0)
    tool_failure_count = Column(Integer, default=0)
    any_tool_failed = Column(Boolean, default=False, index=True)
    invoked_write = Column(Boolean, default=False, index=True)

    # A2 derived signals
    rag_chunk_count = Column(Integer, default=0)
    rag_zero_hits = Column(Boolean, default=False, index=True)
    total_tokens = Column(Integer)
    citation_count = Column(Integer, default=0)
    summarized = Column(Boolean, default=False, index=True)

    # Judge signals — reserved for a future REPORTING-SIDE judging pass over
    # stored rows (per PR #75 review, judging doesn't belong in the request
    # path). The agent never writes these today; they stay NULL.
    query_intent = Column(String(16), index=True)
    refused = Column(Boolean, index=True)
    # First reason the identity gate refused a tool call this turn; the full
    # list is payload["refusals"]. Distinct from `refused`, the judge's verdict.
    permission_refusal = Column(String(32))
    is_deflection = Column(Boolean, index=True)

    # Rating (late UPDATE via POST /api/v1/rating, after the usage_logs write)
    rating = Column(String(16), index=True)  # "helpful" or "not_helpful"
    rating_feedback = Column(Text)
    # The answer the full-screen client shows for this turn, which is how it
    # names the turn it rates. NULL when the turn produced no answer.
    message_id = Column(String(128), index=True)

    payload = Column(_JSONB)  # answer, tool_results, node_trace, params, etc.


class TurnRating(TurnReportBase):  # type: ignore[valid-type,misc]
    """A full-screen rating, keyed by the answer it rates.

    The system of record for those ratings: a click can come before the run's
    turn report is written, and the report copies the rating when it lands.
    One row per answer, so rating again changes it rather than adding one.
    """

    __tablename__ = "turn_ratings"

    thread_id = Column(String(100), primary_key=True)
    message_id = Column(String(128), primary_key=True)
    rating = Column(String(16), nullable=False)
    feedback = Column(Text)
    updated_at = Column(DateTime, default=lambda: datetime.now(UTC), nullable=False)


class ReportToolCall(TurnReportBase):  # type: ignore[valid-type,misc]
    __tablename__ = "report_tool_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("turn_reports.id"), nullable=False, index=True)
    step_index = Column(Integer, default=0)
    tool_name = Column(String(128), index=True)
    server = Column(String(64), index=True)
    success = Column(Boolean, default=True)
    args_hash = Column(String(64), index=True)
    arguments = Column(_JSONB)
    duration_ms = Column(Integer, default=0)  # wall-clock ms measured at the call site


# ── Pure payload builder ──────────────────────────────────────────────────


def _hash_user(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


def _args_hash(arguments: dict[str, Any] | None) -> str:
    blob = json.dumps(arguments or {}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


_URL_RE = re.compile(r"https?://[^\s)\]>}\"']+")


def _count_citations(answer: str | None) -> int:
    """Count distinct cited source URLs in an answer.

    The system prompt instructs the model to cite source URLs; distinct URLs
    are the citation signal. No validity check — citations_valid has no source
    in the agent (an A3/Area-D concern).
    """
    if not answer:
        return 0
    urls = {m.rstrip(".,;:!?") for m in _URL_RE.findall(answer)}
    return len(urls)


def _message_field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _has_text(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, list):
        return any(
            isinstance(block, dict)
            and block.get("type") == "text"
            and str(block.get("text") or "").strip()
            for block in content
        )
    return False


def _turn_answers(messages: Any) -> list[str | None]:
    """Per question, in order: the id of the last AI message with text after it.

    That message is the answer a reader sees for the turn. Anything before the
    first question belongs to no turn.
    """
    answers: list[str | None] = []
    for message in messages if isinstance(messages, list) else []:
        kind = _message_field(message, "type")
        if kind == "human":
            answers.append(None)
        elif kind == "ai" and answers and _has_text(_message_field(message, "content")):
            answers[-1] = _message_field(message, "id") or None
    return answers


def turn_answer_ids(messages: Any) -> list[str]:
    """The answer message id of every turn in a thread that has one."""
    return [answer for answer in _turn_answers(messages) if answer]


def _this_turns_answer(messages: Any) -> str | None:
    """The answer to the last question, or None if it has none yet.

    A failed turn's messages may end at its question, so this never falls back
    to the turn before. A node update with no question in it holds only what
    this turn added.
    """
    if not isinstance(messages, list):
        return None
    if not any(_message_field(m, "type") == "human" for m in messages):
        messages = [{"type": "human"}, *messages]
    return _turn_answers(messages)[-1]


def _dialect_insert(session: Session) -> Any:
    """The INSERT with ON CONFLICT support for the database in use."""
    if session.get_bind().dialect.name == "postgresql":
        return pg_insert
    return sqlite_insert


def _hinted_write_names(tool_catalog: Any) -> set[str]:
    """Tools whose server publishes readOnlyHint: false in this run's catalog.

    invoked_write sees only tool names, so the server's own declaration has to
    come from the catalog the run used.
    """
    if not isinstance(tool_catalog, dict):
        return set()
    return {
        t["name"]
        for server in tool_catalog.get("servers") or []
        if isinstance(server, dict)
        for t in server.get("tools") or []
        if isinstance(t, dict) and t.get("read_only") is False
    }


def _assemble_turn_report(
    *,
    final_state: dict[str, Any],
    session_id: str,
    turn_index: int | None,
    question_id: str,
    query_text: str,
    duration_ms: float | None,
    acting_user: str | None,
    success: bool,
    capabilities: list[str],
    resources: list[str] | None = None,
    turn_capture: dict[str, Any] | None = None,
    judge: dict[str, Any] | None = None,
    origin: str = "real",
    battery_id: str | None = None,
    battery_run_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turn final_state + ids into a (turn_report dict, tool_call dicts) pair.

    Pure: no DB, no I/O. ToolResult objects may be pydantic models or dicts.
    """
    raw_results = final_state.get("tool_results", []) or []
    # A refused call never ran: it isn't a tool call, a failure or a write.
    results = [
        d
        for d in (r if isinstance(r, dict) else r.model_dump() for r in raw_results)
        if not d.get("refused")
    ]
    refusals = list(final_state.get("tool_refusals") or [])
    tools_used = final_state.get("tools_used", []) or []
    capture = turn_capture or {}
    judged = judge or {}
    rag_chunks = capture.get("chunks", []) or []
    rag_searched = bool(capture.get("searched"))

    failures = sum(1 for r in results if not r.get("success", True))
    hinted_writes = _hinted_write_names(final_state.get("tool_catalog"))
    invoked_write = any(
        r.get("tool_name") in WRITE_MCP_TOOL_NAMES or r.get("tool_name") in hinted_writes
        for r in results
    )

    report: dict[str, Any] = {
        "session_id": session_id,
        "turn_index": turn_index,
        "question_id": question_id,
        "query_text": query_text,
        "origin": origin,
        "battery_id": battery_id,
        "battery_run_id": battery_run_id,
        "agent_version": settings.AGENT_VERSION or None,
        "env": settings.DEPLOY_ENV or None,
        "trace_id": capture.get("trace_id"),
        "model_id": active_model_name(),
        "capabilities": capabilities,
        "resources": resources or [],
        "resource_context": final_state.get("resource_context"),
        "was_authenticated": acting_user is not None,
        "user_hash": _hash_user(acting_user),
        "success": success,
        "duration_ms": duration_ms,
        # Invocations, not distinct names: the dashboard buckets tool_count == 0
        # as a "zero_tool" turn, so this has to be the real call count.
        "tool_count": final_state.get("tool_call_count") or len(tools_used),
        "tool_failure_count": failures,
        "any_tool_failed": failures > 0,
        "invoked_write": invoked_write,
        "total_tokens": final_state.get("total_tokens"),
        "citation_count": _count_citations(final_state.get("final_answer")),
        "rag_chunk_count": len(rag_chunks),
        "rag_zero_hits": rag_searched and len(rag_chunks) == 0,
        "summarized": bool(capture.get("summarized")),
        "query_intent": judged.get("query_intent"),
        "refused": judged.get("refused"),
        "is_deflection": judged.get("is_deflection"),
        "permission_refusal": refusals[0]["reason"] if refusals else None,
        "message_id": _this_turns_answer(final_state.get("messages")),
        "payload": {
            "answer": final_state.get("final_answer"),
            "tool_results": results,
            "node_trace": final_state.get("node_trace", []),
            "model_calls": final_state.get("model_calls") or [],
            "reasoning": capture.get("model_reasoning") or [],
            "params": {},
            "retrieved_chunks": rag_chunks,
            "refusals": refusals,
        },
    }
    tool_calls = [
        {
            "step_index": i,
            "tool_name": r.get("tool_name"),
            "server": r.get("server"),
            "success": bool(r.get("success", True)),
            "args_hash": _args_hash(r.get("arguments")),
            "arguments": r.get("arguments") or {},
            "duration_ms": int(r.get("duration_ms") or 0),
        }
        for i, r in enumerate(results)
    ]
    return report, tool_calls


# ── Writer ────────────────────────────────────────────────────────────────


class TurnReporter:
    """Writes turn_reports + report_tool_calls. Mirrors UsageLogger."""

    def __init__(self) -> None:
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        if not settings.DATABASE_URL:
            logger.warning("DATABASE_URL not set, turn reporting disabled")
            return False
        try:
            db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)
            self._engine = create_engine(db_url)
            TurnReportBase.metadata.create_all(self._engine)
            self._migrate_columns()
            self._session_factory = sessionmaker(bind=self._engine)
            self._initialized = True
            logger.info("Turn reporting initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize turn reporting: {e}")
            return False

    def _migrate_columns(self) -> None:
        if self._engine is None:
            return
        existing = {c["name"] for c in inspect(self._engine).get_columns("turn_reports")}
        migrations: dict[str, str] = {
            "rag_chunk_count": "INTEGER DEFAULT 0",
            "rag_zero_hits": "BOOLEAN DEFAULT FALSE",
            "total_tokens": "INTEGER",
            "citation_count": "INTEGER DEFAULT 0",
            "summarized": "BOOLEAN DEFAULT FALSE",
            "query_intent": "VARCHAR(16)",
            "refused": "BOOLEAN",
            "is_deflection": "BOOLEAN",
            "rating": "VARCHAR(16)",
            "rating_feedback": "TEXT",
            "resources": "JSONB",
            "battery_run_id": "VARCHAR(64)",
            "trace_id": "VARCHAR(32)",
            "permission_refusal": "VARCHAR(32)",
            "message_id": "VARCHAR(128)",
        }
        # Each ALTER runs in its own transaction with its own guard: a failure
        # (insufficient DB privileges, transient error) degrades that one column
        # rather than propagating up to _ensure_initialized and disabling ALL
        # turn reporting for the process lifetime.
        for name, col_type in migrations.items():
            if name in existing:
                continue
            try:
                with self._engine.begin() as conn:
                    conn.execute(text(f"ALTER TABLE turn_reports ADD COLUMN {name} {col_type}"))
                logger.info("Added column turn_reports.%s", name)
            except Exception as e:
                logger.warning(
                    "Failed to add column turn_reports.%s (%s); reporting continues "
                    "with that column degraded",
                    name,
                    e,
                )

        # ALTER-migrated columns never get the index their model declaration
        # specifies. Create the one the dashboard actually filters on.
        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_turn_reports_rating ON turn_reports (rating)"
                    )
                )
        except Exception as e:
            logger.warning("Failed to ensure rating index (%s); continuing", e)

        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_turn_reports_battery_run_id "
                        "ON turn_reports (battery_run_id)"
                    )
                )
        except Exception as e:
            logger.warning("Failed to ensure battery_run_id index (%s); continuing", e)

        try:
            with self._engine.begin() as conn:
                conn.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS ix_turn_reports_message_id "
                        "ON turn_reports (message_id)"
                    )
                )
        except Exception as e:
            logger.warning("Failed to ensure message_id index (%s); continuing", e)

    def count_turns_for_session(self, session_id: str) -> int | None:
        """How many turn_reports already exist for this session (prior turns).

        Returns None when the count is unknown (uninitialized or DB error) —
        callers write turn_index NULL rather than a wrong 1 (review finding:
        a failed turn must not claim to be turn 1 when it's really turn 5).
        """
        if not self._ensure_initialized() or self._session_factory is None:
            return None
        session = self._session_factory()
        try:
            return session.query(TurnReport).filter(TurnReport.session_id == session_id).count()
        except Exception as e:
            logger.error(f"Failed to count turns for session: {e}")
            return None
        finally:
            session.close()

    def list_threads_for_user(self, user_hash: str, limit: int) -> list[dict[str, Any]]:
        """Sidebar read: one row per session_id owned by ``user_hash``.

        Single round trip, no N+1: a window function computes the per-session
        label (earliest available query_text) in the same query that groups by
        session_id, rather than a per-session follow-up lookup. NULLS LAST is
        explicit on turn_index so a failed first turn (NULL turn_index) never
        sorts ahead of a real turn_index=1 row when picking the label — this
        must produce identical results on Postgres and SQLite.

        Returns [] (not an error) when uninitialized/on failure — the sidebar
        degrades to an empty list rather than a 500.
        """
        if not self._ensure_initialized() or self._session_factory is None:
            return []
        session = self._session_factory()
        try:
            label = (
                func.first_value(TurnReport.query_text)
                .over(
                    partition_by=TurnReport.session_id,
                    order_by=(
                        TurnReport.turn_index.asc().nulls_last(),
                        TurnReport.created_at.asc(),
                    ),
                )
                .label("label")
            )
            per_row = (
                select(
                    TurnReport.session_id.label("session_id"),
                    TurnReport.created_at.label("created_at"),
                    label,
                )
                .where(TurnReport.user_hash == user_hash)
                .subquery()
            )
            stmt = (
                select(
                    per_row.c.session_id,
                    func.min(per_row.c.created_at).label("min_created_at"),
                    func.max(per_row.c.created_at).label("max_created_at"),
                    # label is identical across all rows of a session (the
                    # window function computed it per-partition), so any()/
                    # min() over it just picks that one value out of the group.
                    func.min(per_row.c.label).label("label"),
                )
                .group_by(per_row.c.session_id)
                .order_by(func.max(per_row.c.created_at).desc())
                .limit(limit)
            )
            rows = session.execute(stmt).all()
            return [
                {
                    "session_id": row.session_id,
                    "created_at": row.min_created_at,
                    "updated_at": row.max_created_at,
                    "label": row.label,
                }
                for row in rows
            ]
        except Exception as e:
            logger.error(f"Failed to list threads for user: {e}")
            return []
        finally:
            session.close()

    def update_rating(self, question_id: str, rating: str, feedback: str | None = None) -> None:
        """Copy a user rating onto the matching turn_reports row.

        Best-effort: the usage_logs write is the system of record; this keeps
        the read model complete. Never raises.
        """
        if not self._ensure_initialized() or self._session_factory is None:
            return
        session = self._session_factory()
        try:
            report = session.query(TurnReport).filter_by(question_id=question_id).first()
            if report is None:
                logger.warning("No turn_report for rated question_id: %s", question_id)
                return
            report.rating = rating  # type: ignore[assignment]
            report.rating_feedback = feedback  # type: ignore[assignment]
            session.commit()
        except Exception as e:
            # Never the exception text: it can carry the bound feedback value.
            logger.error("Failed to update turn report rating (%s)", type(e).__name__)
        finally:
            session.close()

    def rate_answer(
        self, thread_id: str, message_id: str, rating: str, feedback: str | None
    ) -> bool:
        """Record a full-screen rating, replacing any earlier one for that answer.

        Commits the rating before copying it to the turn report, and the report
        writer commits the report before looking for a rating, so whichever runs
        second sees the other's write. True once the rating is saved, whatever
        happens to the copy. Never raises.
        """
        if not self._ensure_initialized() or self._session_factory is None:
            return False
        session = self._session_factory()
        try:
            # One statement, so two first ratings at once can't collide.
            values = {
                "thread_id": thread_id,
                "message_id": message_id,
                "rating": rating,
                "feedback": feedback,
                "updated_at": datetime.now(UTC),
            }
            insert = _dialect_insert(session)(TurnRating).values(**values)
            session.execute(
                insert.on_conflict_do_update(
                    index_elements=[TurnRating.thread_id, TurnRating.message_id],
                    set_={k: insert.excluded[k] for k in ("rating", "feedback", "updated_at")},
                )
            )
            session.commit()
        except Exception as e:
            # Never the exception text: it can carry the bound feedback value.
            logger.error("Failed to save answer rating (%s)", type(e).__name__)
            session.close()
            return False
        try:
            self._copy_rating_to_report(session, thread_id, message_id)
        except Exception as e:
            # Saved all the same: the report picks it up when it's next written.
            logger.error("Failed to copy answer rating to its report (%s)", type(e).__name__)
        finally:
            session.close()
        return True

    @staticmethod
    def _copy_rating_to_report(session: Session, thread_id: str, message_id: str) -> None:
        """One UPDATE ... FROM, so the report gets the rating as committed now."""
        session.execute(
            update(TurnReport)
            .where(
                TurnRating.thread_id == thread_id,
                TurnRating.message_id == message_id,
                TurnReport.session_id == TurnRating.thread_id,
                TurnReport.message_id == TurnRating.message_id,
            )
            .values(rating=TurnRating.rating, rating_feedback=TurnRating.feedback)
        )
        session.commit()

    def ratings_for_thread(self, thread_id: str) -> dict[str, str] | None:
        """Answer id to rating for one thread; None when they can't be read."""
        if not self._ensure_initialized() or self._session_factory is None:
            return None
        session = self._session_factory()
        try:
            rows = session.query(TurnRating).filter_by(thread_id=thread_id).all()
            return {str(row.message_id): str(row.rating) for row in rows}
        except Exception as e:
            logger.error("Failed to read answer ratings (%s)", type(e).__name__)
            return None
        finally:
            session.close()

    def backfill_user_hash(self, session_id: str, acting_user: str) -> None:
        """Sweep NULL-hash turn_reports rows for a session onto acting_user's hash.

        Best-effort, mirrors update_rating: this is a read-model repair, not
        the system of record. The guarded WHERE (user_hash IS NULL) makes
        repeated calls idempotent and self-terminating — once no NULL rows
        remain for the session, the UPDATE matches zero and this is a no-op.
        Never raises.
        """
        if not self._ensure_initialized() or self._session_factory is None:
            return
        session = self._session_factory()
        try:
            session.execute(
                update(TurnReport)
                .where(
                    TurnReport.session_id == session_id,
                    TurnReport.user_hash.is_(None),
                )
                .values(user_hash=_hash_user(acting_user), was_authenticated=True)
            )
            session.commit()
        except Exception as e:
            logger.error(f"Failed to backfill turn report user_hash: {e}")
        finally:
            session.close()

    def log_turn_report(self, **kwargs: Any) -> None:
        if not self._ensure_initialized() or self._session_factory is None:
            return
        session = self._session_factory()
        try:
            report_data, tool_calls = _assemble_turn_report(**kwargs)
            report = TurnReport(**report_data)
            session.add(report)
            session.flush()
            for tc in tool_calls:
                session.add(ReportToolCall(report_id=report.id, **tc))
            session.commit()
        except Exception as e:
            logger.error(f"Failed to write turn report: {e}")
            return
        finally:
            session.close()
        if not report_data["message_id"]:
            return
        # A rating made before this report existed. After the report's commit,
        # so a rating saved meanwhile is seen here or copies onto the report.
        session = self._session_factory()
        try:
            self._copy_rating_to_report(
                session, report_data["session_id"], report_data["message_id"]
            )
        except Exception as e:
            logger.error("Failed to apply pending answer rating (%s)", type(e).__name__)
        finally:
            session.close()


_turn_reporter: TurnReporter | None = None


def get_turn_reporter() -> TurnReporter:
    global _turn_reporter
    if _turn_reporter is None:
        _turn_reporter = TurnReporter()
    return _turn_reporter
