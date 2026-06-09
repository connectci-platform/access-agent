"""Per-turn reporting: a denormalized read model for the reporting dashboard.

Mirrors src/usage_logger.py. Writes one turn_reports row (+ report_tool_calls
children) per completed query turn, off the response path. No PII beyond what
usage_logs already stores; user ids are hashed.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

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
)
from sqlalchemy.orm import declarative_base

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
