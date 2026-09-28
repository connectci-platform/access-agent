"""Coverage for _stream_events branches not exercised elsewhere:
the custom/status SSE event, the TimeoutError branch, and the
CancelledError re-raise.

Harness copied from tests/test_sse_token_stream.py: drive the generator
directly with stream_agent/get_registry mocked, no HTTP client needed.
usage_logger and turn_reporter calls inside the generator hit Postgres in
production but log-and-swallow their own failures (DATABASE_URL="" in the
CI test env), so the paths under test still run to completion.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.api import routes


async def _fake_registry():
    return SimpleNamespace(catalog={"servers": []})


def _raw_request():
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(checkpointer=None)))


async def _collect_events(fake_stream) -> list[tuple[str, dict]]:
    """Drain _stream_events, returning (event_name, data) pairs."""
    events: list[tuple[str, dict]] = []
    with (
        patch.object(routes, "stream_agent", fake_stream),
        patch.object(routes, "get_registry", _fake_registry),
    ):
        gen = routes._stream_events(
            routes.QueryRequest(query="q", session_id="s1"),
            acting_user=None,
            session_id="s1",
            question_id="q1",
            include_trace=False,
            raw_request=_raw_request(),
        )
        try:
            async for line in gen:
                event_name = None
                data = None
                for part in line.split("\n"):
                    if part.startswith("event: "):
                        event_name = part[len("event: ") :]
                    elif part.startswith("data: "):
                        data = json.loads(part[len("data: ") :])
                if event_name is not None:
                    events.append((event_name, data))
        finally:
            await gen.aclose()
    return events


# ---------------------------------------------------------------------------
# custom/status SSE event (routes.py 334-335)
# ---------------------------------------------------------------------------


def test_custom_status_chunk_emits_status_event():
    async def fake_stream(**_kwargs):
        yield "custom", {"type": "status", "message": "Searching documents..."}

    events = asyncio.run(_collect_events(fake_stream))

    status_events = [data for name, data in events if name == "status"]
    assert status_events == [{"message": "Searching documents..."}]


def test_custom_chunk_without_status_type_is_ignored():
    """Non-dict or non-'status' custom chunks must not emit a status event —
    guards the isinstance(chunk, dict) and chunk.get('type') == 'status' checks."""

    async def fake_stream(**_kwargs):
        yield "custom", {"type": "other"}
        yield "custom", "not-a-dict"

    events = asyncio.run(_collect_events(fake_stream))

    assert [name for name, _ in events if name == "status"] == []


# ---------------------------------------------------------------------------
# TimeoutError branch (routes.py 356-358)
# ---------------------------------------------------------------------------


def test_turn_timeout_emits_error_and_done_then_stops():
    async def fake_stream(**_kwargs):
        raise TimeoutError
        yield  # pragma: no cover - unreachable, makes this an async generator

    events = asyncio.run(_collect_events(fake_stream))

    assert events == [
        ("error", {"message": "Query timed out", "code": "timeout"}),
        ("done", {"success": False, "error": "timeout"}),
    ]


# ---------------------------------------------------------------------------
# CancelledError propagation (routes.py 492)
# ---------------------------------------------------------------------------


def test_cancelled_error_propagates_not_swallowed():
    async def fake_stream(**_kwargs):
        raise asyncio.CancelledError
        yield  # pragma: no cover - unreachable, makes this an async generator

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_collect_events(fake_stream))
