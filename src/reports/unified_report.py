"""Unified report combining GA4 and PostgreSQL analytics.

Generates a markdown report suitable for email or Slack delivery.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .db_reports import AgentReport, DBReporter
    from .ga4_client import GA4Client, GA4Report

logger = logging.getLogger(__name__)


def _format_ms(ms: float) -> str:
    """Format milliseconds as human-readable duration."""
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _pct_bar(value: int, total: int, width: int = 20) -> str:
    """Create a simple text percentage bar."""
    if total == 0:
        return ""
    pct = value / total
    filled = int(pct * width)
    return f"{'█' * filled}{'░' * (width - filled)} {pct:.0%}"


def generate_report(
    ga4_client: GA4Client | None,
    db_reporter: DBReporter,
    start_date: str | None = None,
    end_date: str | None = None,
    days: int = 7,
) -> str:
    """Generate a unified markdown report.

    Args:
        ga4_client: GA4 client (None to skip GA4 section).
        db_reporter: PostgreSQL reporter.
        start_date: Start date (YYYY-MM-DD).
        end_date: End date (YYYY-MM-DD).
        days: Lookback period if dates not specified.

    Returns:
        Markdown-formatted report string.
    """
    # Calculate date range for display
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()  # noqa: DTZ005, DTZ007
    if start_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")  # noqa: DTZ007
    else:
        start_dt = end_dt - timedelta(days=days)

    period_str = f"{start_dt.strftime('%b %d')} - {end_dt.strftime('%b %d, %Y')}"

    # Use relative dates for GA4 if no explicit dates
    ga4_start = start_date or f"{days}daysAgo"
    ga4_end = end_date or "yesterday"

    # Fetch data
    agent_report = db_reporter.get_agent_report(start_date, end_date, days)

    ga4_report: GA4Report | None = None
    if ga4_client:
        try:
            ga4_report = ga4_client.get_chatbot_overview(ga4_start, ga4_end)
        except Exception:
            logger.exception("GA4 report failed, continuing with agent data only")

    return _render_markdown(period_str, ga4_report, agent_report)


def _render_ga4_summary(lines: list[str], ga4: GA4Report) -> None:
    """Render GA4 summary bullet points."""
    lines.append(f"- **Sessions:** {ga4.total_sessions:,}")
    lines.append(f"- **Unique users:** {ga4.total_users:,}")

    # Engagement
    opens = ga4.event_counts.get("chatbot_open", 0)
    new_chats = ga4.event_counts.get("chatbot_new_chat", 0)
    if opens or new_chats:
        lines.append(f"- **Engagement:** {opens} opens | {new_chats} new chats")

    # Login prompts
    login_prompts = ga4.event_counts.get("chatbot_login_prompt_shown", 0)
    if login_prompts:
        lines.append(f"- **Login prompts shown:** {login_prompts}")

    # Tickets
    started = ga4.event_counts.get("chatbot_ticket_started", 0)
    submitted = ga4.ticket_submitted
    errors = ga4.ticket_errors
    completion = f"{submitted / started:.0%}" if started > 0 else "N/A"
    lines.append(
        f"- **Tickets:** {started} started → {submitted} submitted "
        f"({completion} completion) | {errors} errors"
    )

    # Security reports
    security = ga4.event_counts.get("chatbot_security_started", 0)
    security_sub = ga4.event_counts.get("chatbot_security_submitted", 0)
    if security > 0:
        lines.append(f"- **Security reports:** {security} started → {security_sub} submitted")

    # AI questions (general Q&A)
    questions = ga4.event_counts.get("chatbot_question_sent", 0)
    answers = ga4.event_counts.get("chatbot_answer_received", 0)
    q_errors = ga4.event_counts.get("chatbot_answer_error", 0)
    if questions:
        lines.append(
            f"- **AI questions (general):** {questions} asked → {answers} answered"
            + (f" | {q_errors} errors" if q_errors else "")
        )

    # AI questions (XDMoD)
    metrics_q = ga4.event_counts.get("chatbot_metrics_question_sent", 0)
    if metrics_q > 0:
        lines.append(f"- **AI questions (XDMoD):** {metrics_q}")

    # Feedback ratings
    ratings = ga4.event_counts.get("chatbot_rating_sent", 0)
    if ratings:
        lines.append(f"- **Feedback ratings:** {ratings}")


def _render_ga4_breakdowns(lines: list[str], ga4: GA4Report) -> None:
    """Render GA4 breakdown subsections."""
    if ga4.menu_selections:
        lines.append("\n### Menu Selections\n")
        total_menu = sum(ga4.menu_selections.values())
        for selection, count in sorted(ga4.menu_selections.items(), key=lambda x: -x[1]):
            lines.append(f"- {selection}: {count} ({count / total_menu:.0%})")

    if ga4.ticket_types:
        lines.append("\n### Ticket Types\n")
        for ttype, count in sorted(ga4.ticket_types.items(), key=lambda x: -x[1]):
            lines.append(f"- {ttype}: {count}")

    if ga4.page_breakdown:
        lines.append("\n### Top Pages\n")
        total_pages = sum(ga4.page_breakdown.values())
        for page, count in list(ga4.page_breakdown.items())[:10]:
            pct = f"{count / total_pages:.0%}" if total_pages else ""
            lines.append(f"- `{page}`: {count} ({pct})")

    if ga4.embed_breakdown:
        lines.append("\n### Embedded vs Floating\n")
        total_embed = sum(ga4.embed_breakdown.values())
        for label, count in sorted(ga4.embed_breakdown.items(), key=lambda x: -x[1]):
            pct = f"{count / total_embed:.0%}" if total_embed else ""
            lines.append(f"- {label.title()}: {count} ({pct})")


def _render_ga4_section(lines: list[str], ga4: GA4Report | None) -> None:
    """Render the GA4 chatbot UI section."""
    if ga4 and ga4.error_available:
        lines.append("## Chatbot UI (GA4)\n")
        _render_ga4_summary(lines, ga4)
        _render_ga4_breakdowns(lines, ga4)
        lines.append("")
    elif ga4 and not ga4.error_available:
        lines.append("## Chatbot UI (GA4)\n")
        lines.append("*GA4 data unavailable — check API credentials and permissions.*\n")


def _render_agent_section(lines: list[str], agent: AgentReport) -> None:
    """Render the agent metrics section."""
    if not agent.error_available:
        lines.append("## AI Agent (PostgreSQL)\n")
        lines.append("*Agent data unavailable — check DATABASE_URL.*\n")
        return

    lines.append("## AI Agent (PostgreSQL)\n")
    lines.append(f"- **Total queries:** {agent.total_queries:,}")
    lines.append(f"- **Unique users:** {agent.unique_users:,}")
    lines.append(f"- **Success rate:** {agent.success_rate}%")
    lines.append(
        f"- **Response time:** avg {_format_ms(agent.duration_avg)} | "
        f"p50 {_format_ms(agent.duration_p50)} | "
        f"p95 {_format_ms(agent.duration_p95)} | "
        f"p99 {_format_ms(agent.duration_p99)}"
    )

    if agent.query_types:
        lines.append("\n### Query Types\n")
        total_qt = sum(agent.query_types.values())
        for qt, count in sorted(agent.query_types.items(), key=lambda x: -x[1]):
            lines.append(f"- {qt}: {count} {_pct_bar(count, total_qt)}")

    if agent.top_topics:
        lines.append("\n### Top Topics\n")
        for topic, count in agent.top_topics:
            lines.append(f"- {topic}: {count}")

    if agent.tool_usage:
        lines.append("\n### Most Used Tools\n")
        for tool, count in agent.tool_usage:
            lines.append(f"- `{tool}`: {count}")

    if agent.daily_counts:
        lines.append("\n### Daily Trend\n")
        lines.append("| Date | Queries | Users |")
        lines.append("|------|---------|-------|")
        for day in agent.daily_counts:
            lines.append(f"| {day['date']} | {day['queries']} | {day['users']} |")

    lines.append("")

    if agent.content_gaps:
        lines.append("## Content Gaps\n")
        lines.append(
            "*Queries with low confidence or failures — areas needing knowledge base improvement.*\n"
        )
        for gap in agent.content_gaps:
            lines.append(f"### {gap['topic']} ({gap['count']} queries)\n")
            samples = gap.get("samples", [])
            for sample in samples if isinstance(samples, list) else []:
                lines.append(f'- "{sample}"')
            lines.append("")


def _render_markdown(
    period: str,
    ga4: GA4Report | None,
    agent: AgentReport,
) -> str:
    """Render the combined report as markdown."""
    lines: list[str] = []
    lines.append("# ACCESS Support Bot — Weekly Report")
    lines.append(f"**{period}**\n")

    _render_ga4_section(lines, ga4)
    _render_agent_section(lines, agent)

    lines.append("---")
    lines.append(
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by access-agent reports*"  # noqa: DTZ005
    )

    return "\n".join(lines)
