"""Confirm ids ride from the request into the turn's HumanMessage (D8-3088)."""

from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import src.agent.graph as graph_mod
from src.agent.context import AgentContext
from src.api.thread_routes import _last_user_text, extract_confirm_ids

A = "a" * 32
B = "0123456789abcdef" * 2


def _confirm(*ids: Any) -> list[dict[str, Any]]:
    return [{"type": "text", "text": "Confirm"}, {"type": "confirm", "preview_ids": list(ids)}]


def test_string_content_has_no_ids():
    assert extract_confirm_ids("Confirm") == []


def test_non_list_content_has_no_ids():
    assert extract_confirm_ids(None) == []
    assert extract_confirm_ids({"type": "confirm", "preview_ids": [A]}) == []


def test_ids_come_from_confirm_blocks():
    assert extract_confirm_ids(_confirm(A, B)) == [A, B]


def test_cancel_has_no_ids():
    assert extract_confirm_ids([{"type": "text", "text": "Cancel"}]) == []


@pytest.mark.parametrize(
    "bad",
    ["A" * 32, "a" * 31, "a" * 33, "g" * 32, "", 5, None, ["a" * 32], {"a": 1}],
)
def test_bad_ids_are_dropped(bad):
    assert extract_confirm_ids(_confirm(bad, A)) == [A]


def test_malformed_blocks_are_dropped():
    content = [
        "junk",
        None,
        {"type": "confirm"},
        {"type": "confirm", "preview_ids": "a" * 32},
        {"type": "confirm", "preview_ids": None},
        {"type": "text", "preview_ids": [B]},
        {"type": "confirm", "preview_ids": [A]},
    ]
    assert extract_confirm_ids(content) == [A]


def test_duplicates_collapse_keeping_order():
    assert extract_confirm_ids(_confirm(B, A, B, A)) == [B, A]


def test_ids_are_capped_at_ten():
    ids = [f"{i:032x}" for i in range(15)]
    assert extract_confirm_ids(_confirm(*ids)) == ids[:10]


def test_confirm_block_does_not_disturb_the_question():
    msgs = [{"type": "human", "content": _confirm(A)}]
    assert _last_user_text(msgs) == "Confirm"


class _Graph:
    def __init__(self, prior: list[Any] | None = None):
        self.prior = prior
        self.state: dict[str, Any] = {}

    async def aget_state(self, config):
        return SimpleNamespace(values={"messages": self.prior} if self.prior else {})

    async def astream(self, state, config=None, **kwargs):
        self.state = state
        if False:
            yield

    async def ainvoke(self, state, config=None, **kwargs):
        self.state = state
        return state


def _last(graph: _Graph) -> Any:
    return graph.state["messages"][-1]


def _args(**extra: Any) -> dict[str, Any]:
    return dict(
        query="Confirm",
        session_id="s",
        question_id="q",
        tool_catalog={},
        context=AgentContext.anonymous(),
        **extra,
    )


async def _drain(**kw: Any) -> None:
    async for _ in graph_mod.stream_agent(**kw):
        pass


@pytest.mark.parametrize("prior", [None, [AIMessage(content="preview")]])
async def test_stream_agent_puts_ids_on_the_human_message(monkeypatch, prior):
    g = _Graph(prior)
    monkeypatch.setattr(graph_mod, "create_agent_graph", lambda: g)
    monkeypatch.setattr(graph_mod, "create_checkpointed_graph", lambda cp: g)
    await _drain(**_args(confirm_ids=[A], checkpointer=object() if prior else None))
    msg = _last(g)
    assert isinstance(msg, HumanMessage)
    assert msg.content == "Confirm"
    assert msg.additional_kwargs["confirm"] == [A]


@pytest.mark.parametrize("prior", [None, [AIMessage(content="preview")]])
async def test_run_agent_puts_ids_on_the_human_message(monkeypatch, prior):
    g = _Graph(prior)
    monkeypatch.setattr(graph_mod, "create_agent_graph", lambda: g)
    monkeypatch.setattr(graph_mod, "create_checkpointed_graph", lambda cp: g)
    await graph_mod.run_agent(**_args(confirm_ids=[A], checkpointer=object() if prior else None))
    assert _last(g).additional_kwargs["confirm"] == [A]


async def test_no_ids_leaves_additional_kwargs_empty(monkeypatch):
    g = _Graph()
    monkeypatch.setattr(graph_mod, "create_agent_graph", lambda: g)
    await _drain(**_args())
    assert "confirm" not in _last(g).additional_kwargs


async def test_route_passes_ids_to_stream_agent(monkeypatch, tmp_path):
    import src.thread_owners as owners
    import src.turn_reporter as reporter
    from src import turnstile
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    monkeypatch.setattr(owners, "_store", None)
    monkeypatch.setattr(reporter, "_turn_reporter", None)
    monkeypatch.setattr(turnstile, "_guard", None)
    from httpx import ASGITransport, AsyncClient

    from src.main import app

    captured: dict[str, Any] = {}

    async def _fake(**kwargs):
        captured.update(kwargs)
        if False:
            yield

    monkeypatch.setattr("src.api.thread_routes.stream_agent", _fake)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        app.state.checkpointer = None
        await c.post(
            "/api/v1/threads/t-confirm/runs/stream",
            json={
                "input": {"messages": [{"type": "human", "content": _confirm(A, "bad")}]},
                "if_not_exists": "create",
            },
        )
    assert captured["query"] == "Confirm"
    assert captured["confirm_ids"] == [A]
