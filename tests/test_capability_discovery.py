"""Capability questions ("What can you do?") answered from the registry.

The widget's /query short-circuit and the full-screen thread runs share the
matcher and the overview builder. On a thread run the answer comes from the
capability_discovery graph node, so it must land in the checkpointed thread
like any model answer. tool_calling_loop_node is replaced with a stub that
fails the test if a capability question reaches it.
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import src.agent.domains.capabilities as caps_mod
import src.agent.graph as g
from src.agent.context import AgentContext
from src.agent.nodes.capability_discovery import (
    capability_categories,
    capability_overview,
    is_capability_question,
)
from src.api.identity import CLAIM_KEY_HEADER
from src.main import app

SIGNED_IN = AgentContext(acting_user="user@access-ci.org", scopes=frozenset({"agent:read"}))
LOCKED_EXAMPLE = '*"Create a GPU training event for next month"*'


@pytest.fixture(autouse=True)
def _fresh_capability_registry():
    caps_mod._registry = None
    yield
    caps_mod._registry = None


@pytest.fixture(autouse=True)
def _loop_must_not_run(monkeypatch):
    """A capability question must never reach the model; other turns get a stub."""
    calls: list[str] = []

    async def _stub(state: dict[str, Any]) -> dict[str, Any]:
        calls.append(state["query"])
        reply = AIMessage(content="model answer", id=f"loop-{len(calls)}")
        return {"final_answer": "model answer", "messages": [*state["messages"], reply]}

    monkeypatch.setattr(g, "tool_calling_loop_node", _stub)
    return calls


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "What can you help me with?",
        "What can you do?",
        "what can you help with",
        "show my options",
        "Show My Options",
        "  WHAT CAN YOU DO?!  ",
        "What can you do.",
        "what  can you\thelp me with ?",
        "🔒 Show my options",
    ],
)
def test_capability_question_matches(text):
    assert is_capability_question(text)


@pytest.mark.parametrize(
    "text",
    [
        "what can you do about my quota?",
        "What can you help me with on Delta?",
        "Can you show my options for storage?",
        "what can you",
        "",
        "Get help",
    ],
)
def test_other_questions_do_not_match(text):
    assert not is_capability_question(text)


# ---------------------------------------------------------------------------
# The overview
# ---------------------------------------------------------------------------


async def test_anonymous_overview_marks_sign_in_capabilities():
    answer = capability_overview(await capability_categories(False), False)

    assert answer.startswith("Here are some things you can try:")
    assert f"{LOCKED_EXAMPLE} 🔒 (login required)" in answer
    assert '*"Is Python available on Delta?"*\n' in answer
    assert "Log in to unlock all capabilities" in answer
    assert "type a question like one of these" not in answer


async def test_signed_in_overview_has_no_locks():
    answer = capability_overview(await capability_categories(True), True)

    assert LOCKED_EXAMPLE in answer
    assert "🔒" not in answer
    assert "Log in" not in answer


async def test_disabled_capability_is_absent():
    from src.config import settings

    with patch.object(settings, "DISABLED_CAPABILITIES", "search_software"):
        answer = capability_overview(await capability_categories(True), True)

    assert "Is Python available on Delta?" not in answer
    assert "Are there any system outages right now?" in answer


async def test_general_category_is_skipped():
    categories = [
        {"id": "general", "label": "General", "capabilities": [{"description": "Ask anything"}]},
        {"id": "explore", "label": "Explore", "capabilities": [{"description": "Find things"}]},
    ]

    answer = capability_overview(categories, True)

    assert "General" not in answer
    assert '**Explore**\n- *"Find things"*' in answer


async def test_resource_context_uses_scoped_categories():
    registry = caps_mod.get_capability_registry()
    scoped = {"categories": [{"id": "resource_docs", "label": "About Delta", "capabilities": []}]}

    async def _scoped(slug: str, authenticated: bool) -> dict[str, Any]:
        return scoped

    with patch.object(registry, "get_by_category_scoped", _scoped):
        assert await capability_categories(True, "delta") == scoped["categories"]


async def test_unknown_resource_context_falls_back_to_all_categories():
    registry = caps_mod.get_capability_registry()

    async def _unknown(slug: str, authenticated: bool) -> None:
        return None

    with patch.object(registry, "get_by_category_scoped", _unknown):
        assert await capability_categories(True, "nope") == registry.get_by_category(True)


# ---------------------------------------------------------------------------
# The graph: the answer is a message in the thread
# ---------------------------------------------------------------------------


async def _run(saver: InMemorySaver, thread_id: str, query: str, context: AgentContext) -> None:
    async for _ in g.stream_agent(
        query=query,
        session_id=thread_id,
        question_id="q",
        tool_catalog={},
        context=context,
        checkpointer=saver,
    ):
        pass


async def test_capability_question_lands_in_thread_state(_loop_must_not_run):
    saver = InMemorySaver()

    await _run(saver, "t1", "What can you help me with?", AgentContext.anonymous())

    state = await g.create_checkpointed_graph(saver).aget_state(
        {"configurable": {"thread_id": "t1"}}
    )
    human, reply = state.values["messages"]
    assert isinstance(human, HumanMessage)
    assert isinstance(reply, AIMessage)
    assert reply.id
    assert "🔒 (login required)" in reply.content
    assert state.values["final_answer"] == reply.content
    assert state.values["tools_used"] == []
    assert _loop_must_not_run == []


async def test_capability_answer_is_history_for_the_next_turn(_loop_must_not_run):
    saver = InMemorySaver()

    await _run(saver, "t2", "Is Delta up?", SIGNED_IN)
    await _run(saver, "t2", "what can you do", SIGNED_IN)
    await _run(saver, "t2", "what can you do about my quota?", SIGNED_IN)

    state = await g.create_checkpointed_graph(saver).aget_state(
        {"configurable": {"thread_id": "t2"}}
    )
    kinds = [m.type for m in state.values["messages"]]
    assert kinds == ["human", "ai", "human", "ai", "human", "ai"]
    assert "🔒" not in state.values["messages"][3].content
    assert _loop_must_not_run == ["Is Delta up?", "what can you do about my quota?"]


# ---------------------------------------------------------------------------
# The full-screen route: streamed, persisted, reported
# ---------------------------------------------------------------------------


@pytest.fixture
async def thread_client(monkeypatch, tmp_path):
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    import src.api.thread_runs as tr
    import src.thread_owners as owners
    import src.turn_reporter as reporter
    import src.turnstile as ts

    owners._store = None
    reporter._turn_reporter = None
    ts._guard = None
    tr._active_by_thread.clear()

    async def _registry() -> SimpleNamespace:
        return SimpleNamespace(catalog={})

    monkeypatch.setattr("src.api.routes.get_registry", _registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        app.state.checkpointer = InMemorySaver()
        yield c
        app.state.checkpointer = None
    reporter._turn_reporter = None
    owners._store = None
    ts._guard = None


def _values_event(body: str) -> dict[str, Any]:
    block = body.split("event: values\n", 1)[1]
    data: dict[str, Any] = json.loads(block.split("data: ", 1)[1].split("\n", 1)[0])
    return data


async def test_fullscreen_run_answers_and_persists(thread_client, _loop_must_not_run):
    from src.turn_reporter import TurnReport, get_turn_reporter

    content = [{"type": "text", "text": "What can you help me with?"}]
    key = {CLAIM_KEY_HEADER: "k" * 43}
    r = await thread_client.post(
        "/api/v1/threads/t-caps/runs/stream",
        headers=key,
        json={"input": {"messages": [{"type": "human", "content": content}]}},
    )
    assert r.status_code == 200

    messages = _values_event(r.text)["messages"]
    assert [m["type"] for m in messages] == ["human", "ai"]
    answer = messages[1]
    assert answer["content"].startswith("Here are some things you can try:")
    assert _loop_must_not_run == []

    history = await thread_client.post("/api/v1/threads/t-caps/history", headers=key, json={})
    assert history.json()[0]["values"]["messages"][1]["content"] == answer["content"]

    session = get_turn_reporter()._session_factory()
    try:
        row = session.query(TurnReport).filter_by(session_id="t-caps").one()
    finally:
        session.close()
    assert row.success is True
    assert row.message_id == answer["id"]


async def test_widget_query_keeps_its_closing_line():
    from src.api.routes import _check_capability_discovery

    response = await _check_capability_discovery("What can you do?", False, "s", "q")

    assert response is not None
    assert response.response.endswith(
        "\n\nJust type a question like one of these, or ask anything else!"
    )
    assert (
        await _check_capability_discovery("what can you do about my quota?", False, "s", "q")
        is None
    )


async def test_anonymous_capability_question_is_free(thread_client, _loop_must_not_run):
    from src.turnstile import get_turnstile_guard

    key = {CLAIM_KEY_HEADER: "k" * 43}

    async def _ask(text: str) -> None:
        r = await thread_client.post(
            "/api/v1/threads/t-free/runs/stream",
            headers=key,
            json={"input": {"messages": [{"type": "human", "content": text}]}},
        )
        assert r.status_code == 200
        _ = r.text

    def _count() -> int:
        session = get_turnstile_guard()._sessions.get("t-free")
        return session.query_count if session else 0

    await _ask("What can you do?")
    assert _count() == 0
    await _ask("Is Delta up?")
    assert _count() == 1
