"""Shared Server-Sent Event formatting.

Lifted out of routes.py so both the widget path (routes.py) and the thread/run
protocol path (thread_routes.py) format events through one definition, without
routes.py and thread_routes.py importing each other (circular import).
"""

import json
from typing import Any


def format_sse_event(event: str, data: Any) -> str:
    """Format a Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
