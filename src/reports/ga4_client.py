"""GA4 Data API client for chatbot UI analytics.

Pulls event data from Google Analytics 4 for chatbot interactions
tracked via GTM (menu selections, ticket flows, submissions, errors).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Filter,
    FilterExpression,
    Metric,
    OrderBy,
    RunReportRequest,
    RunReportResponse,
)

logger = logging.getLogger(__name__)

# Chatbot events tracked via GTM — only intentional user actions.
# chatbot_login_prompt_shown is excluded because it fires passively
# on every page load for unauthenticated visitors (inflates user/session counts).
CHATBOT_EVENTS = [
    # Core chatbot events (qa-bot-core)
    "chatbot_open",
    "chatbot_close",
    "chatbot_new_chat",
    "chatbot_question_sent",
    "chatbot_answer_received",
    "chatbot_answer_error",
    "chatbot_rating_sent",
    "chatbot_login_clicked",
    "chatbot_link_clicked",
    # ACCESS agent layer events
    "chatbot_menu_selected",
    "chatbot_ticket_started",
    "chatbot_ticket_submitted",
    "chatbot_ticket_error",
    "chatbot_ticket_step",
    "chatbot_security_started",
    "chatbot_security_submitted",
    "chatbot_metrics_question_sent",
    "chatbot_file_uploaded",
]


@dataclass
class GA4Report:
    """Structured GA4 report data."""

    period: str  # e.g., "Feb 10 - Feb 17, 2026"
    total_sessions: int = 0
    total_users: int = 0
    event_counts: dict[str, int] = field(default_factory=dict)
    daily_events: list[dict[str, str | int]] = field(default_factory=list)
    menu_selections: dict[str, int] = field(default_factory=dict)
    ticket_types: dict[str, int] = field(default_factory=dict)
    ticket_submitted: int = 0
    ticket_errors: int = 0
    page_breakdown: dict[str, int] = field(default_factory=dict)
    embed_breakdown: dict[str, int] = field(default_factory=dict)
    error_available: bool = True  # False if GA4 query failed


class GA4Client:
    """Client for querying GA4 chatbot analytics."""

    def __init__(self, property_id: str, credentials_path: str | None = None) -> None:
        self.property = f"properties/{property_id}"
        if credentials_path and Path(credentials_path).exists():
            self.client = BetaAnalyticsDataClient.from_service_account_json(credentials_path)
        else:
            # Falls back to Application Default Credentials
            self.client = BetaAnalyticsDataClient()

    def _run_report(self, request: RunReportRequest) -> RunReportResponse:
        """Execute a GA4 report request."""
        result: RunReportResponse = self.client.run_report(request)
        return result

    def get_chatbot_overview(
        self, start_date: str = "7daysAgo", end_date: str = "yesterday"
    ) -> GA4Report:
        """Get overview metrics for all chatbot events.

        Args:
            start_date: Start date (YYYY-MM-DD or relative like "7daysAgo")
            end_date: End date (YYYY-MM-DD or relative like "yesterday")
        """
        report = GA4Report(period=f"{start_date} to {end_date}")

        try:
            self._fetch_event_counts(report, start_date, end_date)
            self._fetch_login_events(report, start_date, end_date)
            self._fetch_daily_breakdown(report, start_date, end_date)
            self._fetch_session_user_counts(report, start_date, end_date)
            self._fetch_menu_selections(report, start_date, end_date)
            self._fetch_ticket_types(report, start_date, end_date)
            self._fetch_page_breakdown(report, start_date, end_date)
            self._fetch_embed_breakdown(report, start_date, end_date)
        except Exception:
            logger.exception("Failed to fetch GA4 data")
            report.error_available = False

        return report

    def _chatbot_event_filter(self) -> FilterExpression:
        """Filter for all chatbot events."""
        return FilterExpression(
            filter=Filter(
                field_name="eventName",
                in_list_filter=Filter.InListFilter(values=CHATBOT_EVENTS),
            )
        )

    def _single_event_filter(self, event_name: str) -> FilterExpression:
        """Filter for a single event name."""
        return FilterExpression(
            filter=Filter(
                field_name="eventName",
                string_filter=Filter.StringFilter(value=event_name),
            )
        )

    def _fetch_event_counts(self, report: GA4Report, start_date: str, end_date: str) -> None:
        """Fetch total count for each chatbot event type."""
        response = self._run_report(
            RunReportRequest(
                property=self.property,
                dimensions=[Dimension(name="eventName")],
                metrics=[Metric(name="eventCount")],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimension_filter=self._chatbot_event_filter(),
            )
        )
        for row in response.rows:
            event_name = row.dimension_values[0].value
            count = int(row.metric_values[0].value)
            report.event_counts[event_name] = count

        report.ticket_submitted = report.event_counts.get("chatbot_ticket_submitted", 0)
        report.ticket_errors = report.event_counts.get("chatbot_ticket_error", 0)

    def _fetch_login_events(self, report: GA4Report, start_date: str, end_date: str) -> None:
        """Fetch login-related events separately.

        chatbot_login_prompt_shown fires passively on page load for logged-out
        users, so it's excluded from CHATBOT_EVENTS (which drives session/user
        counts). We fetch it here to populate event_counts for the login barrier
        funnel in the report.
        """
        login_events = ["chatbot_login_prompt_shown", "chatbot_login_clicked"]
        try:
            response = self._run_report(
                RunReportRequest(
                    property=self.property,
                    dimensions=[Dimension(name="eventName")],
                    metrics=[Metric(name="eventCount")],
                    date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                    dimension_filter=FilterExpression(
                        filter=Filter(
                            field_name="eventName",
                            in_list_filter=Filter.InListFilter(values=login_events),
                        )
                    ),
                )
            )
            for row in response.rows:
                event_name = row.dimension_values[0].value
                count = int(row.metric_values[0].value)
                report.event_counts[event_name] = count
        except Exception:
            logger.debug("Failed to fetch login events from GA4")

    def _fetch_daily_breakdown(self, report: GA4Report, start_date: str, end_date: str) -> None:
        """Fetch daily event counts for trend analysis."""
        response = self._run_report(
            RunReportRequest(
                property=self.property,
                dimensions=[
                    Dimension(name="date"),
                    Dimension(name="eventName"),
                ],
                metrics=[Metric(name="eventCount")],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimension_filter=self._chatbot_event_filter(),
                order_bys=[OrderBy(dimension=OrderBy.DimensionOrderBy(dimension_name="date"))],
            )
        )
        for row in response.rows:
            report.daily_events.append(
                {
                    "date": row.dimension_values[0].value,
                    "event": row.dimension_values[1].value,
                    "count": int(row.metric_values[0].value),
                }
            )

    def _fetch_session_user_counts(self, report: GA4Report, start_date: str, end_date: str) -> None:
        """Fetch total sessions and unique users who triggered chatbot events."""
        response = self._run_report(
            RunReportRequest(
                property=self.property,
                metrics=[
                    Metric(name="sessions"),
                    Metric(name="totalUsers"),
                ],
                date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                dimension_filter=self._chatbot_event_filter(),
            )
        )
        if response.rows:
            report.total_sessions = int(response.rows[0].metric_values[0].value)
            report.total_users = int(response.rows[0].metric_values[1].value)

    def _fetch_menu_selections(self, report: GA4Report, start_date: str, end_date: str) -> None:
        """Fetch menu selection breakdown by custom dimension.

        Requires 'selection' to be registered as a custom dimension in GA4.
        Falls back gracefully if not available.
        """
        try:
            response = self._run_report(
                RunReportRequest(
                    property=self.property,
                    dimensions=[Dimension(name="customEvent:selection")],
                    metrics=[Metric(name="eventCount")],
                    date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                    dimension_filter=self._single_event_filter("chatbot_menu_selected"),
                )
            )
            for row in response.rows:
                selection = row.dimension_values[0].value
                if selection and selection != "(not set)":
                    report.menu_selections[selection] = int(row.metric_values[0].value)
        except Exception:
            logger.debug(
                "Custom dimension 'selection' not available in GA4 — "
                "register it in Admin > Custom Definitions"
            )

    def _fetch_ticket_types(self, report: GA4Report, start_date: str, end_date: str) -> None:
        """Fetch ticket type breakdown by custom dimension.

        Requires 'ticketType' to be registered as a custom dimension in GA4.
        Falls back gracefully if not available.
        """
        try:
            response = self._run_report(
                RunReportRequest(
                    property=self.property,
                    dimensions=[Dimension(name="customEvent:ticketType")],
                    metrics=[Metric(name="eventCount")],
                    date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                    dimension_filter=self._single_event_filter("chatbot_ticket_started"),
                )
            )
            for row in response.rows:
                ticket_type = row.dimension_values[0].value
                if ticket_type and ticket_type != "(not set)":
                    report.ticket_types[ticket_type] = int(row.metric_values[0].value)
        except Exception:
            logger.debug(
                "Custom dimension 'ticketType' not available in GA4 — "
                "register it in Admin > Custom Definitions"
            )

    def _fetch_page_breakdown(self, report: GA4Report, start_date: str, end_date: str) -> None:
        """Fetch chatbot events grouped by page path.

        Uses the built-in GA4 `pagePath` dimension (no custom dimension needed).
        """
        try:
            response = self._run_report(
                RunReportRequest(
                    property=self.property,
                    dimensions=[Dimension(name="pagePath")],
                    metrics=[Metric(name="eventCount")],
                    date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                    dimension_filter=self._chatbot_event_filter(),
                    order_bys=[
                        OrderBy(
                            metric=OrderBy.MetricOrderBy(metric_name="eventCount"),
                            desc=True,
                        )
                    ],
                )
            )
            for row in response.rows:
                page = row.dimension_values[0].value
                if page and page != "(not set)":
                    report.page_breakdown[page] = int(row.metric_values[0].value)
        except Exception:
            logger.debug("Failed to fetch page breakdown from GA4")

    def _fetch_embed_breakdown(self, report: GA4Report, start_date: str, end_date: str) -> None:
        """Fetch chatbot events grouped by embedded vs floating widget.

        Requires 'isEmbedded' to be registered as a custom event dimension in GA4.
        Falls back gracefully if not available.
        """
        try:
            response = self._run_report(
                RunReportRequest(
                    property=self.property,
                    dimensions=[Dimension(name="customEvent:isEmbedded")],
                    metrics=[Metric(name="eventCount")],
                    date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
                    dimension_filter=self._chatbot_event_filter(),
                )
            )
            for row in response.rows:
                raw = row.dimension_values[0].value
                if raw and raw != "(not set)":
                    label = "embedded" if raw == "true" else "floating"
                    report.embed_breakdown[label] = report.embed_breakdown.get(label, 0) + int(
                        row.metric_values[0].value
                    )
        except Exception:
            logger.debug(
                "Custom dimension 'isEmbedded' not available in GA4 — "
                "register it in Admin > Custom Definitions"
            )
