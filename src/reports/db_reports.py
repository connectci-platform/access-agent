"""PostgreSQL usage_logs reports for AI agent metrics.

Queries the usage_logs table for agent performance data:
total queries, unique users, success rates, response times,
top topics, tool usage, and content gaps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import create_engine, text

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


@dataclass
class AgentReport:
    """Structured agent metrics report data."""

    period: str
    total_queries: int = 0
    unique_users: int = 0
    success_count: int = 0
    failure_count: int = 0
    success_rate: float = 0.0

    # Response time percentiles (ms)
    duration_p50: float = 0.0
    duration_p95: float = 0.0
    duration_p99: float = 0.0
    duration_avg: float = 0.0

    # Breakdowns
    query_types: dict[str, int] = field(default_factory=dict)
    top_topics: list[tuple[str, int]] = field(default_factory=list)
    tool_usage: list[tuple[str, int]] = field(default_factory=list)
    daily_counts: list[dict[str, str | int]] = field(default_factory=list)

    # Content gaps — low confidence or failed queries
    content_gaps: list[dict[str, str | int | list[str]]] = field(default_factory=list)

    error_available: bool = True


class DBReporter:
    """Reporter that queries the usage_logs PostgreSQL table."""

    def __init__(self, database_url: str) -> None:
        # Ensure we use psycopg3 driver (not psycopg2)
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
        self._engine: Engine = create_engine(database_url)

    def get_agent_report(
        self, start_date: str | None = None, end_date: str | None = None, days: int = 7
    ) -> AgentReport:
        """Generate an agent metrics report for the given period.

        Args:
            start_date: Start date (YYYY-MM-DD). If None, uses `days` ago.
            end_date: End date (YYYY-MM-DD). If None, uses today.
            days: Number of days to look back if start_date is None.
        """
        end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()  # noqa: DTZ005, DTZ007
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")  # noqa: DTZ007
        else:
            from datetime import timedelta

            start_dt = end_dt - timedelta(days=days)

        period = f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d, %Y')}"
        report = AgentReport(period=period)

        try:
            params = {"start": start_dt, "end": end_dt}
            self._fetch_overview(report, params)
            self._fetch_percentiles(report, params)
            self._fetch_query_types(report, params)
            self._fetch_top_topics(report, params)
            self._fetch_tool_usage(report, params)
            self._fetch_daily_counts(report, params)
            self._fetch_content_gaps(report, params)
        except Exception:
            logger.exception("Failed to query usage_logs")
            report.error_available = False

        return report

    def _fetch_overview(self, report: AgentReport, params: Mapping[str, Any]) -> None:
        """Fetch total queries, unique users, success/failure counts."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(DISTINCT user_hash) as unique_users,
                        COUNT(*) FILTER (WHERE success = 'true') as successes,
                        COUNT(*) FILTER (WHERE success != 'true') as failures
                    FROM usage_logs
                    WHERE timestamp >= :start AND timestamp < :end
                """),
                params,
            ).fetchone()
            if row:
                report.total_queries = row[0]
                report.unique_users = row[1]
                report.success_count = row[2]
                report.failure_count = row[3]
                if report.total_queries > 0:
                    report.success_rate = round(
                        report.success_count / report.total_queries * 100, 1
                    )

    def _fetch_percentiles(self, report: AgentReport, params: Mapping[str, Any]) -> None:
        """Fetch response time percentiles."""
        with self._engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT
                        COALESCE(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_ms), 0),
                        COALESCE(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms), 0),
                        COALESCE(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms), 0),
                        COALESCE(AVG(duration_ms), 0)
                    FROM usage_logs
                    WHERE timestamp >= :start AND timestamp < :end
                        AND duration_ms IS NOT NULL
                """),
                params,
            ).fetchone()
            if row:
                report.duration_p50 = round(float(row[0]), 0)
                report.duration_p95 = round(float(row[1]), 0)
                report.duration_p99 = round(float(row[2]), 0)
                report.duration_avg = round(float(row[3]), 0)

    def _fetch_query_types(self, report: AgentReport, params: Mapping[str, Any]) -> None:
        """Fetch query type distribution."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT COALESCE(query_type, 'unknown'), COUNT(*)
                    FROM usage_logs
                    WHERE timestamp >= :start AND timestamp < :end
                    GROUP BY query_type
                    ORDER BY COUNT(*) DESC
                """),
                params,
            ).fetchall()
            report.query_types = {row[0]: row[1] for row in rows}

    def _fetch_top_topics(self, report: AgentReport, params: Mapping[str, Any]) -> None:
        """Fetch top 10 query topics."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT COALESCE(topic, 'unclassified'), COUNT(*)
                    FROM usage_logs
                    WHERE timestamp >= :start AND timestamp < :end
                    GROUP BY topic
                    ORDER BY COUNT(*) DESC
                    LIMIT 10
                """),
                params,
            ).fetchall()
            report.top_topics = [(row[0], row[1]) for row in rows]

    def _fetch_tool_usage(self, report: AgentReport, params: Mapping[str, Any]) -> None:
        """Fetch most-used MCP tools (unnesting the JSONB array)."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT tool, COUNT(*) as uses
                    FROM usage_logs,
                         jsonb_array_elements_text(tools_used) AS tool
                    WHERE timestamp >= :start AND timestamp < :end
                    GROUP BY tool
                    ORDER BY uses DESC
                    LIMIT 10
                """),
                params,
            ).fetchall()
            report.tool_usage = [(row[0], row[1]) for row in rows]

    def _fetch_daily_counts(self, report: AgentReport, params: Mapping[str, Any]) -> None:
        """Fetch daily query counts for trend."""
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        DATE(timestamp) as day,
                        COUNT(*) as queries,
                        COUNT(DISTINCT user_hash) as users
                    FROM usage_logs
                    WHERE timestamp >= :start AND timestamp < :end
                    GROUP BY DATE(timestamp)
                    ORDER BY day
                """),
                params,
            ).fetchall()
            report.daily_counts = [
                {"date": row[0].strftime("%Y-%m-%d"), "queries": row[1], "users": row[2]}
                for row in rows
            ]

    def _fetch_content_gaps(self, report: AgentReport, params: Mapping[str, Any]) -> None:
        """Find queries with low confidence or failures, grouped by topic.

        These represent areas where the knowledge base needs improvement.
        """
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT
                        COALESCE(topic, 'unclassified') as topic,
                        COUNT(*) as count,
                        ARRAY_AGG(DISTINCT LEFT(query_text, 100)) as sample_queries
                    FROM usage_logs
                    WHERE timestamp >= :start AND timestamp < :end
                        AND (confidence = 'low' OR success != 'true')
                    GROUP BY topic
                    ORDER BY count DESC
                    LIMIT 10
                """),
                params,
            ).fetchall()
            report.content_gaps = [
                {
                    "topic": row[0],
                    "count": row[1],
                    "samples": row[2][:3] if row[2] else [],
                }
                for row in rows
            ]
