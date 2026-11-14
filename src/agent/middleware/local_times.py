"""Middleware: tool results give their times in the user's own time zone.

Event tools return UTC times like ``"start_date": "2026-11-20T18:00:00Z"``, and
models convert those to local time unreliably, or not at all. So the conversion
happens here: every dict in a JSON tool result that has a ``start_date`` or
``end_date`` with a time part also gets ``start_local`` / ``end_local``, such as
"Fri, Nov 20, 2026, 1:00 PM EST". The model is told to quote those as given.

It is generic, not tied to the events tools, so any tool that uses these field
names benefits. It only adds keys, so a result still reads the same to
everything else that parses it (WriteGateMiddleware reads ``status``).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

LOCAL_FIELDS = {"start_date": "start_local", "end_date": "end_local"}

# A date followed by a time, so a bare "2026-11-20" (an all-day date with no
# time to convert) is left alone.
_HAS_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


def format_local(value: Any, zone: tzinfo) -> str | None:
    """``value`` as "Fri, Nov 20, 2026, 1:00 PM EST" in ``zone``, or None.

    None for anything that isn't an ISO datetime with a time part. A naive
    datetime is taken to be UTC, as the tools send.
    """
    if not isinstance(value, str) or not _HAS_TIME.match(value):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    local = parsed.astimezone(zone)
    hour = local.hour % 12 or 12
    return (
        f"{local:%a, %b} {local.day}, {local.year}, "
        f"{hour}:{local:%M %p} {local.tzname() or ''}".rstrip()
    )


def add_local_times(data: Any, zone: tzinfo) -> Any:
    """A copy of ``data`` with local times added beside every datetime field.

    Each local field goes right after the field it comes from, and everything
    else keeps its place and value.
    """
    if isinstance(data, list):
        return [add_local_times(item, zone) for item in data]
    if not isinstance(data, dict):
        return data
    out: dict[str, Any] = {}
    for key, value in data.items():
        out[key] = add_local_times(value, zone)
        local_key = LOCAL_FIELDS.get(key)
        if local_key and local_key not in data:
            local = format_local(value, zone)
            if local is not None:
                out[local_key] = local
    return out


def _zone(name: str | None) -> tzinfo:
    """The named zone, or UTC for none. The route has checked the name already,
    but a bad one still falls back to UTC rather than failing the turn."""
    if not name:
        return UTC
    try:
        return ZoneInfo(name)
    except (ValueError, ZoneInfoNotFoundError):
        return UTC


class LocalTimesMiddleware(AgentMiddleware):
    """Add the user's local time beside each UTC time in a JSON tool result."""

    def __init__(self, time_zone: str | None) -> None:
        super().__init__()
        self.zone: tzinfo = _zone(time_zone)

    def rewrite(self, message: ToolMessage) -> ToolMessage:
        if not isinstance(message.content, str):
            return message
        try:
            data = json.loads(message.content)
        except json.JSONDecodeError:
            return message
        if not isinstance(data, dict | list):
            return message
        updated = add_local_times(data, self.zone)
        if updated == data:
            return message
        return message.model_copy(update={"content": json.dumps(updated, ensure_ascii=False)})

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        result = await handler(request)
        if isinstance(result, ToolMessage):
            return self.rewrite(result)
        return result
