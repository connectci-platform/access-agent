"""Who a run acts for is LangGraph runtime context, passed on every run.

Covers the AgentContext rules, that every entry point fails closed without
one, the eval's write flag, the MCP wrapper's own write check, the prompt for
a caller who can't write, and that a checkpoint written while ``acting_user``
was a state channel still resumes. Route-level gating by token scope is in
tests/test_gated_turns.py.
"""

import argparse
import logging
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.agent import graph as graph_mod
from src.agent.context import AgentContext
from src.agent.domains.tools import MCPToolWrapper
from src.agent.nodes import tool_calling_loop as loop_mod
from src.agent.prompts.system_prompt import build_system_prompt
from src.agent.state import AgentState
from src.agent_token import AgentIdentity
from src.config import settings
from src.tools.mcp_client import MCPClient, MCPToolResult

# --- The context ------------------------------------------------------------


@pytest.mark.parametrize(
    ("acting_user", "scopes", "read_only", "can_write"),
    [
        (None, {"agent:read", "agent:write"}, False, False),
        ("u@x", {"agent:read"}, False, False),
        ("u@x", {"agent:read", "agent:write"}, False, True),
        ("u@x", {"agent:read", "agent:write"}, True, False),
    ],
    ids=["anonymous", "read-only-token", "write-token", "write-token-read-only-deployment"],
)
def test_can_write(monkeypatch, acting_user, scopes, read_only, can_write):
    monkeypatch.setattr(settings, "READ_ONLY", read_only)

    assert AgentContext(acting_user, frozenset(scopes)).can_write is can_write


def test_from_identity_uses_the_tokens_subject_and_scopes():
    identity = AgentIdentity(sub="u@x", scopes=frozenset({"agent:read"}), exp=0)

    assert AgentContext.from_identity(identity) == AgentContext("u@x", frozenset({"agent:read"}))
    assert AgentContext.from_identity(None) == AgentContext.anonymous()


@pytest.mark.parametrize("allow_writes", [True, False])
def test_for_eval_says_whether_the_run_may_write(allow_writes):
    assert AgentContext.for_eval("u@x", allow_writes=allow_writes).can_write is allow_writes
    assert AgentContext.for_eval(None, allow_writes=True) == AgentContext.anonymous()


@pytest.mark.parametrize(
    "fields",
    [
        {"acting_user": "u@x", "scopes": ["agent:write"]},
        {"acting_user": "u@x", "scopes": "agent:write"},
        {"acting_user": 123, "scopes": frozenset()},
    ],
    ids=["list-scopes", "string-scopes", "non-string-user"],
)
def test_a_malformed_context_is_rejected(fields):
    """LangGraph builds the dataclass from a dict context without type checks."""
    with pytest.raises(TypeError):
        AgentContext(**fields)


@pytest.mark.asyncio
async def test_a_dict_context_with_list_scopes_cannot_write():
    from src.agent.graph import create_agent_graph
    from src.agent.state import create_initial_state

    state = create_initial_state(query="q", session_id="s", question_id="q1", tool_catalog={})
    with pytest.raises(TypeError, match="scopes"):
        await create_agent_graph().ainvoke(
            state, {}, context={"acting_user": "u@x", "scopes": ["agent:write"]}
        )


# --- Entry points fail closed -------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["run_agent", "stream_agent"])
async def test_entry_points_require_a_context(entry):
    kwargs = {"query": "q", "session_id": "s", "question_id": "q1", "tool_catalog": {}}
    with pytest.raises(TypeError, match="context"):
        if entry == "run_agent":
            await graph_mod.run_agent(**kwargs)
        else:
            async for _ in graph_mod.stream_agent(**kwargs):
                pass


class _CapturingGraph:
    """Records the context each invocation passes."""

    def __init__(self) -> None:
        self.contexts: list[Any] = []
        self.subgraphs: list[bool] = []

    async def aget_state(self, config):
        return type("Snapshot", (), {"values": {}})()

    async def ainvoke(self, state, config, *, context):
        self.contexts.append(context)
        return {}

    async def astream(self, state, config, *, stream_mode, context, subgraphs=False):
        self.contexts.append(context)
        self.subgraphs.append(subgraphs)
        return
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["run_agent", "stream_agent"])
@pytest.mark.parametrize("branch", ["checkpointer", "db_uri", "plain"])
async def test_every_invocation_site_passes_the_context(monkeypatch, entry, branch):
    fake = _CapturingGraph()

    @asynccontextmanager
    async def _checkpointer(_uri):
        yield AsyncMock()

    monkeypatch.setattr(graph_mod, "create_agent_graph", lambda: fake)
    monkeypatch.setattr(graph_mod, "create_checkpointed_graph", lambda _cp: fake)
    monkeypatch.setattr(graph_mod, "create_async_checkpointer", _checkpointer)
    context = AgentContext("u@x", frozenset({"agent:read"}))
    kwargs: dict[str, Any] = {
        "query": "q",
        "session_id": "s",
        "question_id": "q1",
        "tool_catalog": {},
        "context": context,
    }
    if branch == "checkpointer":
        kwargs["checkpointer"] = object()
    elif branch == "db_uri":
        kwargs.update(use_checkpointing=True, db_uri="postgresql://unused")

    if entry == "run_agent":
        await graph_mod.run_agent(**kwargs)
    else:
        async for _ in graph_mod.stream_agent(**kwargs):
            pass

    assert fake.contexts == [context]
    if entry == "stream_agent":
        assert fake.subgraphs == [True]


# --- The eval's write flag ----------------------------------------------------


@pytest.mark.parametrize("handler", ["_handle_run", "_handle_multiturn"])
def test_eval_rejects_acting_user_without_a_write_flag(handler):
    import src.eval.__main__ as cli

    args = argparse.Namespace(
        questions="q.json",
        threads="t.json",
        system="agent_full",
        judge_model=None,
        allow_factless=False,
        allow_draft_facts=False,
        profile_resource=None,
        resource=None,
        score=False,
        acting_user="jdoe",
        allow_writes=None,
    )

    with (
        patch("src.eval.scorer.run_eval", new_callable=AsyncMock) as run_eval,
        patch("src.eval.multiturn.run_battery", new_callable=AsyncMock) as run_battery,
        pytest.raises(cli.WriteFlagError, match="--allow-writes or --no-allow-writes"),
    ):
        getattr(cli, handler)(args)

    run_eval.assert_not_called()
    run_battery.assert_not_called()


@pytest.mark.parametrize(
    ("acting_user", "flag", "effective"),
    [("jdoe", True, True), ("jdoe", False, False), (None, True, False), (None, None, False)],
)
def test_the_recorded_write_flag_is_what_the_run_could_do(acting_user, flag, effective):
    """Without a user to act for the run is anonymous, so it records no write permission."""
    import src.eval.__main__ as cli

    args = argparse.Namespace(acting_user=acting_user, allow_writes=flag)

    assert cli._effective_allow_writes(args) is effective


def test_the_write_flag_parses_both_ways():
    import src.eval.__main__ as cli

    parser = argparse.ArgumentParser()
    cli._add_write_flag(parser)

    assert parser.parse_args(["--allow-writes"]).allow_writes is True
    assert parser.parse_args(["--no-allow-writes"]).allow_writes is False
    assert parser.parse_args([]).allow_writes is None


# --- The MCP wrapper's own check ------------------------------------------------


def _wrapper(name: str, *, acting_user: str | None, can_write: bool, client: MCPClient):
    return MCPToolWrapper(
        name=name,
        description="t",
        tool_server="events",
        mcp_client=client,
        acting_user=acting_user,
        can_write=can_write,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("acting_user", "reason"), [("u@x", "write_scope"), (None, "write_action")]
)
async def test_wrapper_refuses_a_write_without_permission(monkeypatch, caplog, acting_user, reason):
    client = MCPClient()
    call_tool = AsyncMock()
    monkeypatch.setattr(client, "call_tool", call_tool)
    tool = _wrapper("register_for_event", acting_user=acting_user, can_write=False, client=client)

    with caplog.at_level(logging.ERROR, logger="src.agent.domains.tools"):
        content, artifact = await tool._arun(event_id=1)

    call_tool.assert_not_called()
    assert artifact == {"refusal": {"reason": reason}}
    assert content.startswith("Not done.")
    assert "without write permission" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "can_write"), [("register_for_event", True), ("search_events", False)]
)
async def test_wrapper_calls_allowed_tools(monkeypatch, name, can_write):
    client = MCPClient()
    call_tool = AsyncMock(return_value=MCPToolResult(success=True, data={"ok": 1}, duration_ms=3))
    monkeypatch.setattr(client, "call_tool", call_tool)
    tool = _wrapper(name, acting_user="u@x", can_write=can_write, client=client)

    content, artifact = await tool._arun()

    call_tool.assert_awaited_once()
    assert content == '{"ok": 1}'
    assert artifact is None


def test_catalog_tools_carry_can_write():
    from src.agent.domains.tools import create_mcp_tools_from_catalog

    catalog = {"servers": [{"server": "events", "tools": [{"name": "register_for_event"}]}]}

    (default,) = create_mcp_tools_from_catalog(catalog, "u@x")
    (allowed,) = create_mcp_tools_from_catalog(catalog, "u@x", can_write=True)

    assert default.can_write is False
    assert allowed.can_write is True


@pytest.mark.asyncio
@pytest.mark.parametrize("can_write", [True, False])
async def test_the_node_builds_tools_and_prompt_from_the_context(can_write):
    from tests.agent_runtime import run_loop

    agent = AsyncMock()
    agent.ainvoke.return_value = {"messages": [HumanMessage("q"), AIMessage("a")]}
    built: dict[str, Any] = {}

    def _capture(**kwargs):
        built.update(kwargs)
        return ("system", [], False)

    with (
        patch.object(loop_mod, "create_agent", return_value=agent),
        patch.object(loop_mod, "_build_prompt_and_tools", side_effect=_capture),
    ):
        await run_loop({"messages": [HumanMessage("q")]}, "u@x", can_write=can_write)

    assert built["acting_user"] == "u@x"
    assert built["can_write"] is can_write


@pytest.mark.asyncio
@pytest.mark.parametrize("can_write", [True, False])
async def test_the_wrappers_the_node_builds_carry_the_contexts_write_permission(can_write):
    """Real catalog, real tool factory: the wrapper's own check gets the context's answer."""
    from tests.agent_runtime import run_loop

    agent = AsyncMock()
    agent.ainvoke.return_value = {"messages": [HumanMessage("q"), AIMessage("a")]}
    catalog = {"servers": [{"server": "events", "tools": [{"name": "register_for_event"}]}]}

    with patch.object(loop_mod, "create_agent", return_value=agent) as create:
        await run_loop(
            {"messages": [HumanMessage("q")], "tool_catalog": catalog}, "u@x", can_write=can_write
        )

    (wrapper,) = [t for t in create.call_args.kwargs["tools"] if isinstance(t, MCPToolWrapper)]
    assert wrapper.can_write is can_write


# --- The prompt -------------------------------------------------------------------


def test_a_signed_in_caller_who_cant_write_gets_no_write_workflows():
    prompt = build_system_prompt(acting_user="u@x", can_write=False)

    assert "## Announcements workflow" not in prompt
    # Anyone may still file a ticket (D8-3069).
    assert "## Support-ticket workflow" in prompt
    assert "can't make changes on their behalf" in prompt
    assert "ticket-creation" not in prompt


def test_the_prompt_assumes_no_write_permission_unless_told():
    assert "## Announcements workflow" not in build_system_prompt(acting_user="u@x")


@pytest.mark.parametrize(("acting_user", "can_write"), [("u@x", True), (None, False)])
def test_other_callers_keep_the_write_workflows(acting_user, can_write):
    prompt = build_system_prompt(acting_user=acting_user, can_write=can_write)

    assert "## Announcements workflow" in prompt
    assert "## Support-ticket workflow" in prompt


# --- Checkpoints from before the change -------------------------------------------


class _OldState(AgentState):
    acting_user: str | None


class _AnswerModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "answer"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Second answer."))])


@pytest.mark.asyncio
async def test_a_checkpoint_with_an_acting_user_channel_still_resumes():
    saver = InMemorySaver()
    config = {"configurable": {"thread_id": "old-thread"}}

    def _old_turn(state):
        return {"messages": [AIMessage(content="First answer.")], "final_answer": "First answer."}

    old = StateGraph(_OldState)
    old.add_node("tool_calling_loop", _old_turn)
    old.add_edge(START, "tool_calling_loop")
    old.add_edge("tool_calling_loop", END)
    await old.compile(checkpointer=saver).ainvoke(
        {"messages": [HumanMessage("First?")], "acting_user": "old@x"}, config
    )
    assert (await saver.aget(config))["channel_values"]["acting_user"] == "old@x"

    with (
        patch.object(loop_mod, "get_llm", return_value=_AnswerModel()),
        patch.object(loop_mod, "_build_prompt_and_tools", return_value=("system", [], False)),
    ):
        state = await graph_mod.run_agent(
            query="Second?",
            session_id="old-thread",
            question_id="q2",
            tool_catalog={},
            context=AgentContext.anonymous(),
            checkpointer=saver,
        )

    assert state["final_answer"] == "Second answer."
    assert [m.content for m in state["messages"]][-2:] == ["Second?", "Second answer."]
