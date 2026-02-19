"""CLI entrypoint for generating and delivering reports.

Usage:
    python -m src.reports weekly                    # Print to stdout
    python -m src.reports weekly --email             # Send via email
    python -m src.reports weekly --slack             # Send via Slack
    python -m src.reports weekly --email --slack     # Both
    python -m src.reports weekly --days 30           # Last 30 days
    python -m src.reports weekly --start 2026-01-01 --end 2026-01-31
"""

from __future__ import annotations

import argparse
import logging
import sys

from .db_reports import DBReporter
from .deliver import send_email, send_slack
from .ga4_client import GA4Client
from .unified_report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _load_settings() -> dict[str, str]:
    """Load settings from environment / .env file.

    Uses the existing pydantic-settings config pattern but adds
    report-specific settings.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()

    return {
        "database_url": os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/langgraph",
        ),
        "ga4_property_id": os.getenv("GA4_PROPERTY_ID", ""),
        "ga4_credentials_file": os.getenv("GA4_CREDENTIALS_FILE", ""),
        # Email settings (Mailgun HTTP API)
        "mailgun_api_key": os.getenv("MAILGUN_API_KEY", ""),
        "mailgun_domain": os.getenv("MAILGUN_DOMAIN", ""),
        "email_from": os.getenv("REPORT_EMAIL_FROM", "reports@mg.sweetandfizzy.com"),
        "email_to": os.getenv("REPORT_EMAIL_TO", ""),  # Comma-separated
        # Slack
        "slack_webhook_url": os.getenv("REPORT_SLACK_WEBHOOK_URL", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="ACCESS Support Bot analytics reports")
    parser.add_argument(
        "command",
        choices=["weekly", "monthly"],
        help="Report type (weekly=7 days, monthly=30 days)",
    )
    parser.add_argument("--days", type=int, help="Override lookback period (days)")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument("--email", action="store_true", help="Send via email")
    parser.add_argument("--slack", action="store_true", help="Send via Slack")
    parser.add_argument("--no-ga4", action="store_true", help="Skip GA4 data (agent metrics only)")
    parser.add_argument("--output", "-o", help="Write report to file instead of stdout")

    args = parser.parse_args()
    settings = _load_settings()

    # Determine lookback period
    if args.days:
        days = args.days
    elif args.command == "monthly":
        days = 30
    else:
        days = 7

    # Initialize GA4 client
    ga4_client = None
    if not args.no_ga4 and settings["ga4_property_id"]:
        ga4_client = GA4Client(
            property_id=settings["ga4_property_id"],
            credentials_path=settings["ga4_credentials_file"] or None,
        )
        logger.info(f"GA4 client initialized (property {settings['ga4_property_id']})")
    elif not args.no_ga4:
        logger.warning(
            "GA4_PROPERTY_ID not set — skipping GA4 data. "
            "Set GA4_PROPERTY_ID and GA4_CREDENTIALS_FILE in .env"
        )

    # Initialize DB reporter
    db_reporter = DBReporter(settings["database_url"])

    # Generate report
    report = generate_report(
        ga4_client=ga4_client,
        db_reporter=db_reporter,
        start_date=args.start,
        end_date=args.end,
        days=days,
    )

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        logger.info(f"Report written to {args.output}")
    else:
        print(report)

    # Deliver
    subject = f"ACCESS Support Bot — {'Weekly' if days <= 7 else 'Monthly'} Report"

    if args.email:
        to_addrs = [a.strip() for a in settings["email_to"].split(",") if a.strip()]
        if not settings["mailgun_api_key"] or not settings["mailgun_domain"] or not to_addrs:
            logger.error(
                "Email delivery requires MAILGUN_API_KEY, MAILGUN_DOMAIN, "
                "and REPORT_EMAIL_TO in .env"
            )
            sys.exit(1)
        send_email(
            report,
            subject=subject,
            to_addresses=to_addrs,
            from_address=settings["email_from"],
            mailgun_api_key=settings["mailgun_api_key"],
            mailgun_domain=settings["mailgun_domain"],
        )

    if args.slack:
        if not settings["slack_webhook_url"]:
            logger.error("Slack delivery requires REPORT_SLACK_WEBHOOK_URL in .env")
            sys.exit(1)
        send_slack(report, webhook_url=settings["slack_webhook_url"])


if __name__ == "__main__":
    main()
