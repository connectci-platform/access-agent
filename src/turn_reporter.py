"""Per-turn reporting: a denormalized read model for the reporting dashboard.

Mirrors src/usage_logger.py. Writes one turn_reports row (+ report_tool_calls
children) per completed query turn, off the response path. No PII beyond what
usage_logs already stores; user ids are hashed.
"""

from __future__ import annotations

import hashlib
import json
import logging
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
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .agent.domains.capabilities import WRITE_MCP_TOOL_NAMES
from .config import settings
from .llm.providers import active_model_name

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# JSON renders as JSONB on Postgres and as TEXT-backed JSON on SQLite (tests).
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
    agent_version = Column(String(64), index=True)
    env = Column(String(16), index=True)
    model_id = Column(String(64))

    capabilities = Column(JSON)  # list[str]
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

    # A3 (LLM turn-judge) — added nullable now to stabilize the schema for
    # Plan B; populated by Plan A3, never written here.
    query_intent = Column(String(16), index=True)
    refused = Column(Boolean, index=True)
    is_deflection = Column(Boolean, index=True)

    payload = Column(JSON)  # answer, tool_results, node_trace, params, etc.


class ReportToolCall(TurnReportBase):  # type: ignore[valid-type,misc]
    __tablename__ = "report_tool_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("turn_reports.id"), nullable=False, index=True)
    step_index = Column(Integer, default=0)
    tool_name = Column(String(128), index=True)
    server = Column(String(64), index=True)
    success = Column(Boolean, default=True)
    args_hash = Column(String(64), index=True)
    arguments = Column(JSON)
    duration_ms = Column(Integer, default=0)  # per-call timing deferred to A2


# ── Pure payload builder ──────────────────────────────────────────────────


def _hash_user(user_id: str | None) -> str | None:
    if not user_id:
        return None
    return hashlib.sha256(user_id.encode()).hexdigest()[:16]


def _args_hash(arguments: dict[str, Any] | None) -> str:
    blob = json.dumps(arguments or {}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _assemble_turn_report(
    *,
    final_state: dict[str, Any],
    session_id: str,
    turn_index: int,
    question_id: str,
    query_text: str,
    duration_ms: float | None,
    acting_user: str | None,
    success: bool,
    capabilities: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turn final_state + ids into a (turn_report dict, tool_call dicts) pair.

    Pure: no DB, no I/O. ToolResult objects may be pydantic models or dicts.
    """
    raw_results = final_state.get("tool_results", []) or []
    results = [r if isinstance(r, dict) else r.model_dump() for r in raw_results]
    tools_used = final_state.get("tools_used", []) or []

    failures = sum(1 for r in results if not r.get("success", True))
    invoked_write = any(r.get("tool_name") in WRITE_MCP_TOOL_NAMES for r in results)

    report: dict[str, Any] = {
        "session_id": session_id,
        "turn_index": turn_index,
        "question_id": question_id,
        "query_text": query_text,
        "origin": "real",
        "agent_version": settings.AGENT_VERSION or None,
        "env": settings.DEPLOY_ENV or None,
        "model_id": active_model_name(),
        "capabilities": capabilities,
        "resource_context": final_state.get("resource_context"),
        "was_authenticated": acting_user is not None,
        "user_hash": _hash_user(acting_user),
        "success": success,
        "duration_ms": duration_ms,
        "tool_count": len(tools_used),
        "tool_failure_count": failures,
        "any_tool_failed": failures > 0,
        "invoked_write": invoked_write,
        "payload": {
            "answer": final_state.get("final_answer"),
            "tool_results": results,
            "node_trace": final_state.get("node_trace", []),
            "params": {},
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
        }
        with self._engine.begin() as conn:
            for name, col_type in migrations.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE turn_reports ADD COLUMN {name} {col_type}"))

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
