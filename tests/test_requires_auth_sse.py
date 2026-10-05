"""requires_auth SSE threading on both stream paths: /api/v1/query
(src/api/routes.py's _stream_events) and the thread/run protocol
(src/api/thread_routes.py's _stream_run_events).

Harness copied from tests/test_stream_events_branches.py (routes.py) and
tests/test_thread_routes_helpers.py (thread_routes.py, no HTTP/JWKS needed
for a direct generator call) — stream_agent/get_registry mocked, no LLM/MCP.

The loop node surfaces `requires_auth` via its return dict (a node_output
dict under the "updates" stream type); both generators fold node_output into
final_state the same way, so a scripted "updates" chunk carrying that key is
enough to exercise the SSE emission without going through the real agent.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from src.api import routes, thread_routes

REQUIRES_AUTH_PAYLOAD = {
    "login_url": "https://support.access-ci.org/user/login",
    "reason": "write_action",
}


async def _fake_registry():
    return SimpleNamespace(catalog={"servers": []})


async def _collect_query_events(fake_stream) -> list[tuple[str, dict]]:
    """Drain routes._stream_events, returning (event_name, data) pairs."""
    events: list[tuple[str, dict]] = []
    with (
        patch.object(routes, "stream_agent", fake_stream),
        patch.object(routes, "get_registry", _fake_registry),
    ):
        gen = routes._stream_events(
            routes.QueryRequest(query="register me", session_id="s1"),
            acting_user=None,
            session_id="s1",
            question_id="q1",
            include_trace=False,
            raw_request=SimpleNamespace(
                app=SimpleNamespace(state=SimpleNamespace(checkpointer=None))
            ),
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


async def _collect_run_events(fake_stream) -> list[tuple[str, dict]]:
    """Drain thread_routes._stream_run_events, returning (event_name, data) pairs."""
    events: list[tuple[str, dict]] = []
    final_state: dict = {}
    with (
        patch.object(thread_routes, "stream_agent", fake_stream),
        patch("src.api.routes.get_registry", _fake_registry),
    ):
        gen = thread_routes._stream_run_events(
            thread_id="t1",
            run_id="r1",
            query="register me",
            caller=None,
            resource_context=None,
            checkpointer=None,
            final_state=final_state,
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
# /api/v1/query (routes.py _stream_events)
# ---------------------------------------------------------------------------


def test_query_stream_emits_requires_auth_and_stops():
    async def fake_stream(**_kwargs):
        yield "updates", {"tool_calling_loop": {"requires_auth": REQUIRES_AUTH_PAYLOAD}}

    events = asyncio.run(_collect_query_events(fake_stream))

    assert events == [("requires_auth", REQUIRES_AUTH_PAYLOAD)]


def test_query_stream_without_requires_auth_runs_normal_done_path():
    async def fake_stream(**_kwargs):
        yield "updates", {"tool_calling_loop": {"final_answer": "hi there"}}

    events = asyncio.run(_collect_query_events(fake_stream))

    assert [name for name, _ in events] == ["done"]
    assert events[0][1]["response"] == "hi there"


# ---------------------------------------------------------------------------
# /threads/{id}/runs/stream (thread_routes.py _stream_run_events)
# ---------------------------------------------------------------------------


def test_thread_run_stream_emits_requires_auth_and_stops():
    async def fake_stream(**_kwargs):
        yield "updates", {"tool_calling_loop": {"requires_auth": REQUIRES_AUTH_PAYLOAD}}

    events = asyncio.run(_collect_run_events(fake_stream))

    event_names = [name for name, _ in events]
    assert "requires_auth" in event_names
    assert dict(events)["requires_auth"] == REQUIRES_AUTH_PAYLOAD
    # Ends cleanly: no messages/complete or values after the signal.
    assert event_names[-1] == "requires_auth"
    assert "values" not in event_names


def test_thread_run_stream_without_requires_auth_runs_normal_values_path():
    async def fake_stream(**_kwargs):
        yield "updates", {"tool_calling_loop": {"final_answer": "hi there", "messages": []}}

    events = asyncio.run(_collect_run_events(fake_stream))

    event_names = [name for name, _ in events]
    assert "requires_auth" not in event_names
    assert "values" in event_names
