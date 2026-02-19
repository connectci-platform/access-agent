"""Report delivery via email (Mailgun API) and Slack webhook."""

from __future__ import annotations

import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)


def send_email(
    report_markdown: str,
    *,
    subject: str,
    to_addresses: list[str],
    from_address: str,
    mailgun_api_key: str,
    mailgun_domain: str,
) -> bool:
    """Send the report via Mailgun HTTP API as both plain text and HTML.

    Args:
        report_markdown: The markdown report content.
        subject: Email subject line.
        to_addresses: List of recipient email addresses.
        from_address: Sender email address.
        mailgun_api_key: Mailgun sending API key.
        mailgun_domain: Mailgun sending domain (e.g., mg.sweetandfizzy.com).

    Returns:
        True if sent successfully, False otherwise.
    """
    html_body = f"""\
<html>
<body style="font-family: monospace; white-space: pre-wrap; padding: 20px;">
{_markdown_to_basic_html(report_markdown)}
</body>
</html>"""

    try:
        response = httpx.post(
            f"https://api.mailgun.net/v3/{mailgun_domain}/messages",
            auth=("api", mailgun_api_key),
            data={
                "from": from_address,
                "to": to_addresses,
                "subject": subject,
                "text": report_markdown,
                "html": html_body,
            },
            timeout=30,
        )
        if response.status_code == 200:
            logger.info(f"Report emailed to {', '.join(to_addresses)}")
            return True
        logger.error(f"Mailgun returned {response.status_code}: {response.text}")
        return False
    except Exception:
        logger.exception("Failed to send email")
        return False


def send_slack(
    report_markdown: str,
    *,
    webhook_url: str,
) -> bool:
    """Send the report to a Slack channel via incoming webhook.

    Slack webhooks accept markdown-like formatting (mrkdwn).
    We convert the report to Slack's block format for better rendering.

    Args:
        report_markdown: The markdown report content.
        webhook_url: Slack incoming webhook URL.

    Returns:
        True if sent successfully, False otherwise.
    """
    # Slack has a 3000-char limit per text block, so split into sections
    sections = _split_for_slack(report_markdown)

    blocks = []
    for section in sections:
        blocks.append(
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": section},
            }
        )

    payload = {"blocks": blocks}

    try:
        response = httpx.post(
            webhook_url,
            content=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if response.status_code == 200:
            logger.info("Report sent to Slack")
            return True
        logger.error(f"Slack webhook returned {response.status_code}: {response.text}")
        return False
    except Exception:
        logger.exception("Failed to send Slack message")
        return False


def _split_for_slack(markdown: str, max_len: int = 2900) -> list[str]:
    """Split markdown into chunks that fit Slack's block size limit.

    Tries to split on section headers (##) for clean breaks.
    """
    lines = markdown.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for newline
        # Split on headers if we're getting close to the limit
        if line.startswith("## ") and current_len > 0 and current_len + line_len > max_len:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

        current.append(line)
        current_len += line_len

        if current_len >= max_len:
            chunks.append("\n".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append("\n".join(current))

    return chunks


def _convert_inline_formatting(text: str) -> str:
    """Apply bold and inline code formatting to a line of text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", text)


def _convert_table_row(line: str) -> str:
    """Convert a markdown table row to HTML."""
    cells = [c.strip() for c in line.split("|")[1:-1]]
    row = "".join(f"<td style='padding: 4px 8px; border: 1px solid #ddd;'>{c}</td>" for c in cells)
    return f"<tr>{row}</tr>"


def _convert_text_line(line: str) -> str:
    """Convert a plain text line (non-header, non-table) to HTML."""
    formatted = _convert_inline_formatting(line)
    if formatted.strip() == "---":
        return "<hr>"
    if formatted.startswith("- "):
        return f"<li>{formatted[2:]}</li>"
    if formatted.startswith("*") and formatted.endswith("*") and not formatted.startswith("**"):
        return f"<em>{formatted[1:-1]}</em>"
    return f"{formatted}<br>"


def _markdown_to_basic_html(md: str) -> str:
    """Minimal markdown-to-HTML for email rendering.

    Handles headers, bold, code, and tables. Not a full parser —
    just enough for our structured reports.
    """
    lines = md.split("\n")
    html_lines: list[str] = []
    in_table = False

    for line in lines:
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif re.match(r"^\|[-|]+\|$", line):
            continue
        elif line.startswith("|"):
            if not in_table:
                html_lines.append("<table style='border-collapse: collapse;'>")
                in_table = True
            html_lines.append(_convert_table_row(line))
        else:
            if in_table:
                html_lines.append("</table>")
                in_table = False
            html_lines.append(_convert_text_line(line))

    if in_table:
        html_lines.append("</table>")

    return "\n".join(html_lines)
