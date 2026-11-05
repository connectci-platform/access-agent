"""Gated turns finish normally and are recorded, on both stream routes.

Runs through the HTTP app on the real stack: the routes, graph, checkpointer,
loop node and middleware are all real. The seams are ``get_llm`` (a scripted
model), the tool list (stub tools that record whether they ran) and the MCP
catalog. The model implements ``_astream`` as well as ``_generate``, because
both routes stream only ``AIMessageChunk`` tokens.

The routes don't currently receive token chunks from the loop's inner agent,
so a reply arrives whole in ``values`` and ``done.response``. The assertions
hold either way: the prompt follows any streamed tokens and precedes the
events that complete the turn.
"""

import json
from typing import Any
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from src.agent.nodes import tool_calling_loop as loop_mod
from src.api import routes
from src.main import app

WRITE, PRIVATE, PUBLIC = "register_for_event", "get_my_rp_accounts", "search_events"


class _ScriptedModel(BaseChatModel):
    """Plays one script entry per model call: a list of tool calls or an answer.

    Records the tools bound for each call and the messages each call received.
    Once the script runs out it answers, so a turn can't loop forever.
    """

    script: list[Any] = Field(default_factory=list)
    calls: int = 0
    bound: list[list[str]] = Field(default_factory=list)
    received: list[list[Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound.append([t.name if hasattr(t, "name") else t["name"] for t in tools])
        return self

    def _next(self, messages: list[Any]) -> AIMessage:
        self.received.append(list(messages))
        step = self.script[self.calls] if self.calls < len(self.script) else "Done."
        self.calls += 1
        if isinstance(step, str):
            return AIMessage(content=step)
        calls = [
            {"name": name, "args": {}, "id": f"chatcmpl-tool-{i}"} for i, name in enumerate(step)
        ]
        return AIMessage(content="", tool_calls=calls)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        msg = self._next(messages)
        if msg.tool_calls:
            chunks = [
                {"name": tc["name"], "args": "{}", "id": tc["id"], "index": i}
                for i, tc in enumerate(msg.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for word in str(msg.content).split(" "):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word + " "))
            # LangGraph's messages stream listens for this callback, not the yield.
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


class _ServerTool(StructuredTool):
    tool_server: str = ""


def _tools(ran: list[str]) -> list[StructuredTool]:
    def make(name: str, server: str) -> StructuredTool:
        def _run() -> str:
            ran.append(name)
            return json.dumps({"tool": name, "results": [{"title": "Python workshop"}]})

        return _ServerTool.from_function(
            func=_run, name=name, description=f"{name} stub", tool_server=server
        )

    return [make(WRITE, "events"), make(PRIVATE, "allocations"), make(PUBLIC, "events")]


async def _fake_registry():
    from types import SimpleNamespace

    return SimpleNamespace(catalog={"servers": []})


@pytest.fixture(autouse=True)
def _isolated_stores(monkeypatch, tmp_path):
    """Thread owners and turn reports on a temp sqlite file; fresh singletons."""
    from src.config import settings

    monkeypatch.setattr(settings, "DATABASE_URL", f"sqlite:///{tmp_path / 't.db'}", raising=False)
    monkeypatch.setattr(settings, "TURNSTILE_SECRET_KEY", "", raising=False)
    import src.api.thread_runs as tr
    import src.thread_owners as owners
    import src.turn_reporter as reporter_mod
    import src.turnstile as ts

    owners._store = None
    reporter_mod._turn_reporter = None
    ts._guard = None
    tr._active_by_thread.clear()
    tr._task_by_run.clear()
    yield
    reporter_mod._turn_reporter = None
    ts._guard = None


@pytest.fixture
def stack():
    """Patch the seams; yield (model, ran) for the test to script and inspect."""
    model = _ScriptedModel()
    ran: list[str] = []
    with (
        patch.object(loop_mod, "get_llm", return_value=model),
        patch.object(
            loop_mod, "_build_prompt_and_tools", return_value=("system", _tools(ran), False)
        ),
        patch.object(routes, "get_registry", _fake_registry),
    ):
        yield model, ran


@pytest.fixture
async def client():
    app.state.checkpointer = InMemorySaver()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    app.state.checkpointer = None


def _events(body: str) -> list[tuple[str, Any]]:
    events = []
    for block in body.split("\n\n"):
        name = data = None
        for line in block.split("\n"):
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: ") :])
        if name:
            events.append((name, data))
    return events


async def _run(client, thread_id: str, text: str, headers: dict | None = None):
    r = await client.post(
        f"/api/v1/threads/{thread_id}/runs/stream",
        headers=headers or {},
        json={"input": {"messages": [{"role": "user", "content": text}]}},
    )
    assert r.status_code == 200
    return _events(r.text)


async def _query(client, session_id: str, text: str) -> list[tuple[str, Any]]:
    r = await client.post("/api/v1/query", json={"query": text, "session_id": session_id})
    assert r.status_code == 200
    return _events(r.text)


def _rows(session_id: str) -> tuple[list[Any], list[Any]]:
    from src.turn_reporter import ReportToolCall, TurnReport, get_turn_reporter

    reporter = get_turn_reporter()
    assert reporter._ensure_initialized()
    session = reporter._session_factory()
    try:
        reports = session.query(TurnReport).filter_by(session_id=session_id).all()
        ids = [r.id for r in reports]
        calls = session.query(ReportToolCall).filter(ReportToolCall.report_id.in_(ids)).all()
        return reports, calls
    finally:
        session.close()


def _names(events: list[tuple[str, Any]]) -> list[str]:
    return [name for name, _ in events]


def _after_tokens_and_before(events, prompt: str, token: str, closing: list[str]) -> None:
    """The prompt comes after every streamed token and before the closing events."""
    names = _names(events)
    at = names.index(prompt)
    assert all(i < at for i, n in enumerate(names) if n == token)
    assert all(names.index(c) > at for c in closing if c in names)
    assert closing[-1] in names


def _thread_reply(events) -> str:
    (values,) = [d for n, d in events if n == "values"]
    return values["messages"][-1]["content"]


# --- runs/stream ---------------------------------------------------------------


async def test_anonymous_write_on_runs_stream_completes_with_a_sign_in_prompt(client, stack):
    model, ran = stack
    model.script = [[WRITE], "You'll need to sign in to register."]

    events = await _run(client, "t-write", "Register me for the Python workshop")

    assert ran == []
    custom = [d for n, d in events if n == "custom"]
    assert custom == [{"type": "requires_auth", "reason": "write_action"}]
    _after_tokens_and_before(events, "custom", "messages/partial", ["messages/complete", "values"])
    assert "sign in to register" in _thread_reply(events)


async def test_anonymous_private_read_is_refused_as_private_data(client, stack):
    model, ran = stack
    model.script = [[PRIVATE], "Sign in to see your accounts."]

    events = await _run(client, "t-private", "What are my RP accounts?")

    assert ran == []
    assert [d for n, d in events if n == "custom"] == [
        {"type": "requires_auth", "reason": "private_data"}
    ]


async def test_signed_in_write_runs_with_no_prompt(client, stack, agent_tokens):
    model, ran = stack
    model.script = [[WRITE], "You're registered."]

    events = await _run(
        client, "t-signed-in", "Register me", headers=agent_tokens.headers("user@x")
    )

    assert ran == [WRITE]
    assert "custom" not in _names(events)
    assert all(set(b) == {WRITE, PRIVATE, PUBLIC} for b in model.bound)
    (report,), (call,) = _rows("t-signed-in")
    assert report.permission_refusal is None
    assert report.invoked_write is True
    assert call.tool_name == WRITE


async def test_the_refused_turn_is_recorded_accurately(client, stack):
    model, _ran = stack
    model.script = [[WRITE, PUBLIC], "You'll need to sign in to register. Here are the events."]

    await _run(client, "t-record", "Register me and show me events")

    (report,), calls = _rows("t-record")
    assert report.success is True
    assert report.permission_refusal == "write_action"
    assert report.invoked_write is False
    assert report.tool_count == 1
    assert report.capabilities == ["browse_events"]
    assert [c.tool_name for c in calls] == [PUBLIC]
    assert report.payload["refusals"] == [{"tool_name": WRITE, "reason": "write_action"}]
    assert [r["tool_name"] for r in report.payload["tool_results"]] == [PUBLIC]


async def test_the_next_turn_on_the_thread_has_no_refusal(client, stack):
    model, _ran = stack
    model.script = [[WRITE], "Sign in to register.", [PUBLIC], "Here are the events."]

    await _run(client, "t-next", "Register me")
    turn_two_starts = model.calls
    events = await _run(client, "t-next", "What events are there?")

    assert "custom" not in _names(events)
    # The tools hidden after turn 1's refusal are offered again in turn 2.
    assert set(model.bound[turn_two_starts]) == {WRITE, PRIVATE, PUBLIC}
    first, second = sorted(_rows("t-next")[0], key=lambda r: r.id)
    assert first.permission_refusal == "write_action"
    assert second.permission_refusal is None
    assert second.payload["refusals"] == []


async def test_refused_only_turn_after_tool_use_reports_no_tool_calls(client, stack):
    model, _ran = stack
    model.script = [[PUBLIC], "Here are the events.", [WRITE], "Sign in to register."]

    await _run(client, "t-refused-only", "What events are there?")
    await _run(client, "t-refused-only", "Register me")

    _first, second = sorted(_rows("t-refused-only")[0], key=lambda r: r.id)
    assert second.tool_count == 0
    assert second.invoked_write is False
    assert second.capabilities == []


async def test_after_a_refusal_gated_tools_are_hidden_for_the_rest_of_the_turn(client, stack):
    """The model can't retry, and a call naming a hidden tool anyway is refused."""
    model, ran = stack
    model.script = [[WRITE], [WRITE], "You'll need to sign in to register."]

    events = await _run(client, "t-hidden", "Register me")

    assert set(model.bound[0]) == {WRITE, PRIVATE, PUBLIC}
    assert model.bound[1] == [PUBLIC]
    assert ran == []
    assert "sign in to register" in _thread_reply(events)
    (report,), _ = _rows("t-hidden")
    (trace,) = report.payload["node_trace"]
    assert trace["tool_results"] == 0
    assert report.payload["refusals"] == [
        {"tool_name": WRITE, "reason": "write_action"},
        {"tool_name": WRITE, "reason": "write_action"},
    ]


async def test_a_thread_whose_only_turn_was_refused_reaches_the_sidebar(
    client, stack, agent_tokens
):
    from src.turn_reporter import get_turn_reporter

    model, _ran = stack
    model.script = [[WRITE], "Sign in to register."]

    await _run(client, "t-sidebar", "Register me")
    get_turn_reporter().backfill_user_hash("t-sidebar", "user@x")
    r = await client.post("/api/v1/threads/search", headers=agent_tokens.headers("user@x"), json={})

    assert [t["thread_id"] for t in r.json()] == ["t-sidebar"]


# --- /query --------------------------------------------------------------------


async def test_anonymous_write_on_query_streams_the_reply_then_the_prompt(client, stack):
    from src.config import settings

    model, ran = stack
    model.script = [[WRITE], "You'll need to sign in to register."]

    events = await _query(client, "s-write", "Register me for the Python workshop")

    assert ran == []
    _after_tokens_and_before(events, "requires_auth", "token", ["done"])
    refusal = {"login_url": settings.LOGIN_URL, "reason": "write_action"}
    assert [d for n, d in events if n == "requires_auth"] == [refusal]
    (done,) = [d for n, d in events if n == "done"]
    assert done["metadata"]["refusal"] == refusal
    # A client that knows only status/token/done/error still renders the reply,
    # from tokens or, as qa-bot-core does when none arrived, from done.response.
    tokens = "".join(d["content"] for n, d in events if n == "token")
    assert "sign in to register" in (tokens or done["response"])


async def test_query_turn_with_no_refusal_has_no_prompt(client, stack):
    model, _ran = stack
    model.script = [[PUBLIC], "Here are the events."]

    events = await _query(client, "s-plain", "What events are there?")

    assert "requires_auth" not in _names(events)
    (done,) = [d for n, d in events if n == "done"]
    assert "refusal" not in done["metadata"]


async def test_query_refused_turn_writes_its_report(client, stack):
    model, _ran = stack
    model.script = [[WRITE], "Sign in to register."]

    await _query(client, "s-report", "Register me")

    (report,), calls = _rows("s-report")
    assert report.permission_refusal == "write_action"
    assert calls == []


# --- What the model sees -------------------------------------------------------


async def test_the_model_gets_prose_and_never_the_artifact(client, stack):
    from langchain_openai.chat_models.base import _convert_message_to_dict

    model, _ran = stack
    model.script = [[WRITE], "Sign in to register."]

    await _run(client, "t-prose", "Register me")

    refused = [m for m in model.received[1] if isinstance(m, ToolMessage)]
    assert len(refused) == 1
    assert refused[0].content.startswith("Not done.")
    # The provider payload carries role, content and tool_call_id, never the artifact.
    assert "artifact" not in _convert_message_to_dict(refused[0])
    assert "refusal" not in json.dumps(_convert_message_to_dict(refused[0]))
