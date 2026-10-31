"""Tests for request_identity, the Bearer dependency on every identity-bearing
endpoint (src/api/identity.py).

Each refusal must land before the handler body: before Turnstile, before any
ownership write, and before a run starts. Tokens come from the shared
``agent_tokens`` fixture.
"""

import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.agent_token import AgentTokenUnavailableError
from src.api.identity import current_agent_token
from src.config import settings
from src.main import app
from src.thread_owners import get_thread_owner_store

# Every endpoint that depends on request_identity.
ENDPOINTS = {
    "query": ("POST", "/api/v1/query", {"query": "hi", "session_id": "t-1"}),
    "capabilities": ("GET", "/api/v1/capabilities", None),
    "whoami": ("GET", "/api/v1/whoami", None),
    "rating": ("POST", "/api/v1/rating", {"query_id": "q-1", "rating": "helpful"}),
    "search": ("POST", "/api/v1/threads/search", {"limit": 10}),
    "history": ("POST", "/api/v1/threads/t-1/history", {"limit": 10}),
    "handoff": ("POST", "/api/v1/threads/t-1/handoff", None),
    "exchange": ("POST", "/api/v1/handoff/exchange", {"token": "x"}),
    "run": (
        "POST",
        "/api/v1/threads/t-1/runs/stream",
        {"input": {"messages": [{"role": "user", "content": "hi"}]}},
    ),
    "cancel": ("POST", "/api/v1/threads/t-1/runs/r-1/cancel", None),
}


@pytest.fixture(autouse=True)
def _sqlite_stores(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    import src.thread_owners as owners
    import src.turn_reporter as reporter

    owners._store = None
    reporter._turn_reporter = None
    yield
    owners._store = None
    reporter._turn_reporter = None


@pytest.fixture(autouse=True)
def _reset_run_registry():
    import src.api.thread_runs as tr

    tr._active_by_thread.clear()
    tr._task_by_run.clear()
    yield
    tr._active_by_thread.clear()
    tr._task_by_run.clear()


@pytest.fixture
def downstream(monkeypatch):
    """Everything a refused request must never reach."""
    mocks = SimpleNamespace(
        turnstile=AsyncMock(return_value=None),
        query_agent=MagicMock(),
        run_agent=MagicMock(),
        rating=MagicMock(),
    )
    monkeypatch.setattr("src.api.routes.check_turnstile", mocks.turnstile)
    monkeypatch.setattr("src.api.thread_routes.check_turnstile", mocks.turnstile)
    monkeypatch.setattr("src.api.routes.stream_agent", mocks.query_agent)
    monkeypatch.setattr("src.api.thread_routes.stream_agent", mocks.run_agent)
    monkeypatch.setattr(
        "src.api.routes.get_usage_logger",
        MagicMock(return_value=MagicMock(log_rating=mocks.rating)),
    )
    return mocks


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        app.state.checkpointer = None
        yield c


async def _call(client, endpoint, headers):
    method, path, body = ENDPOINTS[endpoint]
    return await client.request(method, path, json=body, headers=headers)


def _assert_nothing_ran(downstream):
    downstream.turnstile.assert_not_awaited()
    downstream.query_agent.assert_not_called()
    downstream.run_agent.assert_not_called()
    downstream.rating.assert_not_called()
    assert get_thread_owner_store().resolve_owner("t-1") is None


# --- Refusals, on every endpoint ---


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_bad_token_is_401_before_anything_runs(client, agent_tokens, downstream, endpoint):
    wrong_audience = agent_tokens.token("jsmith@access-ci.org", aud="https://elsewhere.test")

    r = await _call(client, endpoint, {"Authorization": f"Bearer {wrong_audience}"})

    assert r.status_code == 401
    assert r.headers["www-authenticate"] == 'Bearer error="invalid_token"'
    _assert_nothing_ran(downstream)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_other_authorization_scheme_is_401(client, agent_tokens, downstream, endpoint):
    r = await _call(client, endpoint, {"Authorization": "Basic dXNlcjpwYXNz"})

    assert r.status_code == 401
    assert r.headers["www-authenticate"] == 'Bearer error="invalid_token"'
    _assert_nothing_ran(downstream)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_unavailable_key_set_is_503(client, agent_tokens, downstream, endpoint, monkeypatch):
    unavailable = SimpleNamespace(verify=AsyncMock(side_effect=AgentTokenUnavailableError()))
    monkeypatch.setattr(app.state, "agent_token_verifier", unavailable)

    r = await _call(client, endpoint, agent_tokens.headers("jsmith@access-ci.org"))

    assert r.status_code == 503
    _assert_nothing_ran(downstream)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
async def test_token_expiring_within_a_turn_is_401(client, agent_tokens, downstream, endpoint):
    exp = int(time.time()) + settings.AGENT_TURN_TIMEOUT_S + 29

    r = await _call(client, endpoint, agent_tokens.headers("jsmith@access-ci.org", exp=exp))

    assert r.status_code == 401
    _assert_nothing_ran(downstream)


async def test_token_outliving_a_turn_is_accepted(client, agent_tokens):
    exp = int(time.time()) + settings.AGENT_TURN_TIMEOUT_S + 32

    r = await client.get(
        "/api/v1/whoami", headers=agent_tokens.headers("jsmith@access-ci.org", exp=exp)
    )

    assert r.status_code == 200
    assert r.json()["authenticated"] is True


# --- The raw token's one home ---


async def test_streamed_turn_sees_the_token(client, agent_tokens, monkeypatch):
    """The run's generator executes after the handler returns; the token set by
    the dependency must still be visible there, where tool calls run."""
    seen: list[str | None] = []

    async def _fake(**_kwargs):
        seen.append(current_agent_token())
        yield "updates", {"tool_calling_loop": {"final_answer": "hi"}}

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    token = agent_tokens.token("jsmith@access-ci.org")
    body = ENDPOINTS["run"][2]

    await client.post(
        "/api/v1/threads/t-2/runs/stream", json=body, headers={"Authorization": f"Bearer {token}"}
    )
    await client.post("/api/v1/threads/t-3/runs/stream", json=body)

    assert seen == [token, None]


async def test_token_reaches_no_checkpoint_span_or_log(client, agent_tokens, monkeypatch, caplog):
    """A full streamed turn through the real graph, with only the LLM loop
    stubbed. The stub reads the token as a tool would."""
    from src.agent import graph

    seen: list[str | None] = []

    async def _loop_node(state):
        seen.append(current_agent_token())
        return {"final_answer": "stub answer", "messages": [AIMessage("stub answer")]}

    monkeypatch.setattr(graph, "tool_calling_loop_node", _loop_node)
    registry = SimpleNamespace(catalog={"tools": [], "quick_lookup": {}})
    monkeypatch.setattr("src.api.routes.get_registry", AsyncMock(return_value=registry))
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)

    saver = InMemorySaver()
    app.state.checkpointer = saver
    spans = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(spans))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    caplog.set_level(logging.DEBUG)

    token = agent_tokens.token("jsmith@access-ci.org")
    r = await client.post(
        "/api/v1/threads/t-4/runs/stream",
        json=ENDPOINTS["run"][2],
        headers={"Authorization": f"Bearer {token}"},
    )

    assert r.status_code == 200
    assert "stub answer" in r.text
    assert seen == [token]
    assert saver.storage, "the turn wrote no checkpoint"
    assert token.encode() not in repr(saver.storage).encode()
    assert token.encode() not in repr(saver.writes).encode()
    assert spans.get_finished_spans(), "the turn recorded no span"
    for span in spans.get_finished_spans():
        assert all(token not in str(value) for value in (span.attributes or {}).values())
    for record in caplog.records:
        assert token not in record.getMessage()
