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
    inspect,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
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
    turn_index = Column(Integer, default=1)
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
    is_deflection = Column(Boolean, index=True)

    # Rating (late UPDATE via POST /api/v1/rating, after the usage_logs write)
    rating = Column(String(16), index=True)  # "helpful" or "not_helpful"
    rating_feedback = Column(Text)

    payload = Column(_JSONB)  # answer, tool_results, node_trace, params, etc.


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
    results = [r if isinstance(r, dict) else r.model_dump() for r in raw_results]
    tools_used = final_state.get("tools_used", []) or []
    capture = turn_capture or {}
    judged = judge or {}
    rag_chunks = capture.get("chunks", []) or []
    rag_searched = bool(capture.get("searched"))

    failures = sum(1 for r in results if not r.get("success", True))
    invoked_write = any(r.get("tool_name") in WRITE_MCP_TOOL_NAMES for r in results)

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
        "payload": {
            "answer": final_state.get("final_answer"),
            "tool_results": results,
            "node_trace": final_state.get("node_trace", []),
            "model_calls": final_state.get("model_calls") or [],
            "reasoning": capture.get("model_reasoning") or [],
            "params": {},
            "retrieved_chunks": rag_chunks,
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
            logger.error(f"Failed to update turn report rating: {e}")
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
        finally:
            session.close()


_turn_reporter: TurnReporter | None = None


def get_turn_reporter() -> TurnReporter:
    global _turn_reporter
    if _turn_reporter is None:
        _turn_reporter = TurnReporter()
    return _turn_reporter
