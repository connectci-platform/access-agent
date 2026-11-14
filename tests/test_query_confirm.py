"""POST /api/v1/query carries a Confirm click's ids and returns the turn's previews."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import ToolMessage

import tests.test_auth_e2e  # noqa: F401  (module-level env setup)
from src.main import app

A = "a" * 32
B = "b" * 32
END = {"final_answer": "ok", "tools_used": [], "query_analysis": None, "query_classification": None}


def _preview(preview_id=A, **extra):
    body = {"status": "preview", "executed": False, "data": {"title": "T"}, **extra}
    if preview_id:
        body["preview_id"] = preview_id
    return ToolMessage(content=json.dumps(body), tool_call_id="c1", name="register_for_event")


@pytest.fixture(autouse=True)
def _no_turnstile(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)


@pytest.fixture
def agent():
    sent: list = []

    async def fake_stream(**kwargs):
        if sent:
            yield "tool_messages", list(sent)
        yield "updates", {"__end__": END}

    with patch("src.api.routes.stream_agent", side_effect=fake_stream) as mock:
        mock.sent = sent
        yield mock


@pytest.fixture
def registry():
    reg = AsyncMock()
    reg.catalog = {"tools": [], "quick_lookup": {}}
    with patch("src.api.routes.get_registry", return_value=reg):
        yield reg


async def _done(body):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        r = await c.post("/api/v1/query", json={"query": "q", **body})
    assert r.status_code == 200
    lines = r.text.splitlines()
    i = max(n for n, line in enumerate(lines) if line == "event: done")
    return json.loads(lines[i + 1].removeprefix("data: "))


async def test_confirm_ids_reach_stream_agent_filtered(agent, registry):
    await _done({"confirm_ids": [A, "nope", A, B.upper(), B]})
    assert agent.call_args.kwargs["confirm_ids"] == [A, B]


async def test_malformed_confirm_ids_are_dropped(agent, registry):
    await _done({"confirm_ids": ["x", "A" * 32, "a" * 31]})
    assert not agent.call_args.kwargs["confirm_ids"]


async def test_done_lists_a_stamped_preview(agent, registry):
    agent.sent.extend([_preview(), _preview(None), _preview(B, no_changes=True)])
    done = await _done({})
    assert done["previews"] == [
        {
            "preview_id": A,
            "tool": "register_for_event",
            "data": {"title": "T"},
            "no_changes": False,
        },
        {"preview_id": B, "tool": "register_for_event", "data": {"title": "T"}, "no_changes": True},
    ]


async def test_done_has_no_previews_without_any(agent, registry):
    done = await _done({})
    assert "previews" not in done
