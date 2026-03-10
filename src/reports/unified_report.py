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


def _pct(numerator: int, denominator: int) -> str:
    """Format a percentage, returning 'N/A' if denominator is zero."""
    if denominator == 0:
        return "N/A"
    return f"{numerator / denominator:.0%}"


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


# ---------------------------------------------------------------------------
# Engagement section
# ---------------------------------------------------------------------------


def _render_engagement(lines: list[str], ga4: GA4Report) -> None:
    """Render engagement overview — who used the chatbot and how."""
    lines.append("## Engagement\n")

    opens = ga4.event_counts.get("chatbot_open", 0)
    closes = ga4.event_counts.get("chatbot_close", 0)
    new_chats = ga4.event_counts.get("chatbot_new_chat", 0)

    lines.append(f"- **Users:** {ga4.total_users:,}")
    lines.append(f"- **Sessions:** {ga4.total_sessions:,}")
    lines.append(f"- **Chatbot opened:** {opens}")
    if new_chats:
        lines.append(f"- **New conversations:** {new_chats}")
    if closes:
        lines.append(f"- **Chatbot closed:** {closes}")

    if ga4.embed_breakdown:
        total_embed = sum(ga4.embed_breakdown.values())
        parts = []
        for label, count in sorted(ga4.embed_breakdown.items(), key=lambda x: -x[1]):
            parts.append(f"{label} {count} ({_pct(count, total_embed)})")
        lines.append(f"- **Mode:** {', '.join(parts)}")

    lines.append("")


# ---------------------------------------------------------------------------
# Login barrier section
# ---------------------------------------------------------------------------


def _render_login_barrier(lines: list[str], ga4: GA4Report) -> None:
    """Render login barrier funnel — how login gating affects usage."""
    login_prompts = ga4.event_counts.get("chatbot_login_prompt_shown", 0)
    login_clicks = ga4.event_counts.get("chatbot_login_clicked", 0)

    # Only show if there's data
    if not login_prompts and not login_clicks:
        return

    lines.append("## Login Barrier\n")

    if login_prompts:
        lines.append(f"- **Login prompts shown:** {login_prompts:,}")
    if login_clicks:
        lines.append(f"- **Login button clicked:** {login_clicks:,}")
        if login_prompts:
            lines.append(f"- **Click-through rate:** {_pct(login_clicks, login_prompts)}")

    # Estimate abandonment: people who saw the prompt but didn't click login
    if login_prompts and login_clicks:
        abandoned = login_prompts - login_clicks
        if abandoned > 0:
            lines.append(
                f"- **Abandoned at login:** ~{abandoned:,} ({_pct(abandoned, login_prompts)})"
            )

    lines.append("")


# ---------------------------------------------------------------------------
# User funnel section
# ---------------------------------------------------------------------------


def _render_qa_funnel(lines: list[str], ga4: GA4Report) -> None:
    """Render Q&A funnel subsection."""
    qa_selected = ga4.menu_selections.get("Ask a question about ACCESS", 0)
    questions = ga4.event_counts.get("chatbot_question_sent", 0)
    if not qa_selected and not questions:
        return

    qa_line = f"- **Ask a question:** {qa_selected} selected"
    if questions:
        answers = ga4.event_counts.get("chatbot_answer_received", 0)
        q_errors = ga4.event_counts.get("chatbot_answer_error", 0)
        qa_line += f" → {questions} questions → {answers} answered"
        if q_errors:
            qa_line += f" | {q_errors} errors"
    lines.append(qa_line)

    ratings = ga4.event_counts.get("chatbot_rating_sent", 0)
    if ratings:
        lines.append(f"  - {ratings} feedback ratings submitted")
    link_clicks = ga4.event_counts.get("chatbot_link_clicked", 0)
    if link_clicks:
        lines.append(f"  - {link_clicks} links clicked in answers")


def _render_ticket_funnel(lines: list[str], ga4: GA4Report) -> None:
    """Render ticket funnel subsection."""
    ticket_selected = ga4.menu_selections.get("Open a Help Ticket", 0)
    ticket_started = ga4.event_counts.get("chatbot_ticket_started", 0)
    if not ticket_selected and not ticket_started:
        return

    ticket_line = f"- **Help tickets:** {ticket_selected} selected"
    if ticket_started:
        ticket_line += (
            f" → {ticket_started} started → {ga4.ticket_submitted} submitted"
            f" ({_pct(ga4.ticket_submitted, ticket_started)})"
        )
        if ga4.ticket_errors:
            ticket_line += f" | {ga4.ticket_errors} errors"
    lines.append(ticket_line)


def _render_xdmod_funnel(lines: list[str], ga4: GA4Report) -> None:
    """Render XDMoD funnel subsection."""
    xdmod_selected = ga4.menu_selections.get("Usage and performance of ACCESS resources (XDMoD)", 0)
    xdmod_questions = ga4.event_counts.get("chatbot_metrics_question_sent", 0)
    if not xdmod_selected and not xdmod_questions:
        return

    xdmod_line = f"- **XDMoD metrics:** {xdmod_selected} selected"
    if xdmod_questions:
        xdmod_line += f" → {xdmod_questions} questions"
    lines.append(xdmod_line)


def _render_security_funnel(lines: list[str], ga4: GA4Report) -> None:
    """Render security funnel subsection."""
    security_selected = ga4.menu_selections.get("Report a security issue", 0)
    security_started = ga4.event_counts.get("chatbot_security_started", 0)
    if not security_selected and not security_started:
        return

    security_line = f"- **Security reports:** {security_selected} selected"
    if security_started:
        security_submitted = ga4.event_counts.get("chatbot_security_submitted", 0)
        security_line += f" → {security_started} started → {security_submitted} submitted"
    lines.append(security_line)


def _render_funnel(lines: list[str], ga4: GA4Report) -> None:
    """Render the user activity funnel — what people did after opening the chatbot."""
    menu_total = sum(ga4.menu_selections.values()) if ga4.menu_selections else 0
    if not menu_total:
        return

    lines.append(f"## What Users Did ({menu_total} menu selections)\n")

    _render_qa_funnel(lines, ga4)
    _render_ticket_funnel(lines, ga4)
    _render_xdmod_funnel(lines, ga4)
    _render_security_funnel(lines, ga4)

    # Any other menu selections not covered above
    known_selections = {
        "Ask a question about ACCESS",
        "Open a Help Ticket",
        "Usage and performance of ACCESS resources (XDMoD)",
        "Report a security issue",
    }
    other_selections = {k: v for k, v in ga4.menu_selections.items() if k not in known_selections}
    for selection, count in sorted(other_selections.items(), key=lambda x: -x[1]):
        lines.append(f"- **{selection}:** {count}")

    lines.append("")


# ---------------------------------------------------------------------------
# GA4 breakdowns section
# ---------------------------------------------------------------------------


def _render_ga4_breakdowns(lines: list[str], ga4: GA4Report) -> None:
    """Render GA4 breakdown subsections (ticket types, top pages)."""
    if ga4.ticket_types:
        lines.append("### Ticket Types\n")
        for ttype, count in sorted(ga4.ticket_types.items(), key=lambda x: -x[1]):
            lines.append(f"- {ttype}: {count}")
        lines.append("")

    if ga4.page_breakdown:
        lines.append("### Top Pages\n")
        total_pages = sum(ga4.page_breakdown.values())
        for page, count in list(ga4.page_breakdown.items())[:10]:
            pct = f" ({_pct(count, total_pages)})" if total_pages else ""
            lines.append(f"- `{page}`: {count}{pct}")
        lines.append("")


# ---------------------------------------------------------------------------
# Agent performance section
# ---------------------------------------------------------------------------


def _render_agent_section(lines: list[str], agent: AgentReport) -> None:
    """Render the AI agent backend metrics."""
    if not agent.error_available:
        lines.append("## Agent Performance\n")
        lines.append("*Agent data unavailable — check DATABASE_URL.*\n")
        return

    # Skip section entirely if no queries
    if agent.total_queries == 0:
        return

    lines.append("## Agent Performance\n")
    lines.append(f"- **Queries processed:** {agent.total_queries:,}")
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


# ---------------------------------------------------------------------------
# Content gaps section
# ---------------------------------------------------------------------------


def _render_content_gaps(lines: list[str], agent: AgentReport) -> None:
    """Render content gaps — the most actionable part of the report."""
    if not agent.error_available or not agent.content_gaps:
        return

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


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------


def _render_markdown(
    period: str,
    ga4: GA4Report | None,
    agent: AgentReport,
) -> str:
    """Render the combined report as markdown."""
    lines: list[str] = []
    lines.append("# ACCESS Support Bot — Weekly Report")
    lines.append(f"**{period}**\n")

    # GA4 sections
    if ga4 and ga4.error_available:
        _render_engagement(lines, ga4)
        _render_login_barrier(lines, ga4)
        _render_funnel(lines, ga4)
        _render_ga4_breakdowns(lines, ga4)
    elif ga4 and not ga4.error_available:
        lines.append("## Chatbot UI (GA4)\n")
        lines.append("*GA4 data unavailable — check API credentials and permissions.*\n")

    # Agent sections
    _render_agent_section(lines, agent)
    _render_content_gaps(lines, agent)

    lines.append("---")
    lines.append(
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by access-agent reports*"  # noqa: DTZ005
    )

    return "\n".join(lines)
