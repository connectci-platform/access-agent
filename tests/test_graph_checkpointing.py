"""Coverage for src/agent/graph.py's checkpointer branches.

These are the paths that stitch a LangGraph checkpointer (injected at the
lifespan level, or built per-request as a transition fallback) into
run_agent/stream_agent's resume-then-invoke flow. None of this needs a real
LLM or Postgres:

- The injected-checkpointer branch is exercised against a real
  langgraph.checkpoint.memory.InMemorySaver, so the resume behavior (does a
  second call see the first call's messages?) is asserted for real rather
  than just executed.
- The per-request branch (use_checkpointing=True, db_uri=...) normally opens
  an AsyncPostgresSaver; create_async_checkpointer is patched to return an
  async context manager wrapping the same InMemorySaver so the branch's
  setup()-then-graph.ainvoke/astream wiring is real too.
- tool_calling_loop_node is swapped for a stub so the graph doesn't touch an
  LLM; the swap is done on the module-level name graph.py binds at import
  time, which _build_graph_structure looks up fresh on every graph build.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import patch

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import src.agent.graph as g

TOOL_CATALOG: dict[str, Any] = {"tools": [], "quick_lookup": {}}


class _FakeAsyncCheckpointerCM:
    """Stands in for create_async_checkpointer(db_uri)'s return value."""

    def __init__(self, saver: InMemorySaver) -> None:
        self._saver = saver

    async def __aenter__(self) -> InMemorySaver:
        return self._saver

    async def __aexit__(self, *exc_info: object) -> None:
        return None


def _stub_node(answer: str) -> Any:
    async def _node(state: dict[str, Any]) -> dict[str, Any]:
        return {"final_answer": answer, "tools_used": []}

    return _node


@pytest.fixture(autouse=True)
def _fake_loop_node(monkeypatch):
    """Swap the LLM-calling node for a stub for the whole module."""
    monkeypatch.setattr(g, "tool_calling_loop_node", _stub_node("stub answer"))


@pytest.fixture
def saver_with_async_setup():
    """A real InMemorySaver with an async setup() shim.

    InMemorySaver has no setup() (Postgres-only concept); the per-request
    branch always awaits cp.setup(), so the fake needs one.
    """
    saver = InMemorySaver()

    async def _noop_setup() -> None:
        return None

    saver.setup = _noop_setup  # type: ignore[method-assign]
    return saver


# ---------------------------------------------------------------------------
# create_pooled_checkpointer
# ---------------------------------------------------------------------------


def test_create_pooled_checkpointer_returns_async_postgres_saver_cm():
    """Body just forwards to AsyncPostgresSaver.from_conn_string; assert that."""
    with patch(
        "langgraph.checkpoint.postgres.aio.AsyncPostgresSaver.from_conn_string"
    ) as from_conn_string:
        sentinel = object()
        from_conn_string.return_value = sentinel

        result = g.create_pooled_checkpointer("postgresql://fake/db")

        from_conn_string.assert_called_once_with("postgresql://fake/db")
        assert result is sentinel


# ---------------------------------------------------------------------------
# run_agent — injected checkpointer branch
# ---------------------------------------------------------------------------


async def test_run_agent_injected_checkpointer_fresh_thread():
    """No prior state on this thread_id: the query becomes the only message,
    and the stub node's answer comes back unchanged."""
    saver = InMemorySaver()

    result = await g.run_agent(
        query="hello",
        session_id="thread-fresh",
        question_id="q1",
        tool_catalog=TOOL_CATALOG,
        checkpointer=saver,
    )

    assert result["final_answer"] == "stub answer"


async def test_run_agent_injected_checkpointer_resumes_prior_state():
    """A second run_agent call on the same thread_id sees the first call's
    checkpointed messages and appends to them (asserts real resume, not just
    that the branch executed)."""
    saver = InMemorySaver()

    await g.run_agent(
        query="first turn",
        session_id="thread-resume",
        question_id="q1",
        tool_catalog=TOOL_CATALOG,
        checkpointer=saver,
    )

    config = {"configurable": {"thread_id": "thread-resume"}}
    graph = g.create_checkpointed_graph(saver)
    state_after_first = await graph.aget_state(config)
    messages_after_first = state_after_first.values.get("messages", [])
    assert len(messages_after_first) >= 1

    await g.run_agent(
        query="second turn",
        session_id="thread-resume",
        question_id="q2",
        tool_catalog=TOOL_CATALOG,
        checkpointer=saver,
    )

    state_after_second = await graph.aget_state(config)
    messages_after_second = state_after_second.values.get("messages", [])
    # The second run's initial_state seeded [*existing_messages, HumanMessage(query)]
    # before the (stubbed) node ran, so the thread accumulated more messages.
    assert len(messages_after_second) > len(messages_after_first)


# ---------------------------------------------------------------------------
# run_agent — per-request branch (use_checkpointing + db_uri, no injected cp)
# ---------------------------------------------------------------------------


async def test_run_agent_per_request_checkpointer_branch(saver_with_async_setup):
    """use_checkpointing=True + db_uri with no injected checkpointer builds one
    per-request via create_async_checkpointer, runs setup(), and invokes."""
    with patch.object(
        g,
        "create_async_checkpointer",
        return_value=_FakeAsyncCheckpointerCM(saver_with_async_setup),
    ) as create_cp:
        result = await g.run_agent(
            query="hello",
            session_id="thread-per-request",
            question_id="q1",
            tool_catalog=TOOL_CATALOG,
            use_checkpointing=True,
            db_uri="sqlite:///fake",
        )

        create_cp.assert_called_once_with("sqlite:///fake")
        assert result["final_answer"] == "stub answer"

    # Resume behavior on the per-request path too: the checkpointed thread
    # now has state a direct aget_state can see.
    config = {"configurable": {"thread_id": "thread-per-request"}}
    graph = g.create_checkpointed_graph(saver_with_async_setup)
    state = await graph.aget_state(config)
    assert state.values.get("messages")


# ---------------------------------------------------------------------------
# stream_agent — injected checkpointer branch
# ---------------------------------------------------------------------------


async def _drain(agen: AsyncGenerator[tuple[str, Any], None]) -> list[tuple[str, Any]]:
    return [item async for item in agen]


async def test_stream_agent_injected_checkpointer_fresh_thread():
    saver = InMemorySaver()

    events = await _drain(
        g.stream_agent(
            query="hello",
            session_id="stream-thread-fresh",
            question_id="q1",
            tool_catalog=TOOL_CATALOG,
            checkpointer=saver,
        )
    )

    assert events  # the stub node produced at least one update event
    stream_types = {stream_type for stream_type, _ in events}
    assert "updates" in stream_types


async def test_stream_agent_injected_checkpointer_resumes_prior_state():
    """Same resume assertion as run_agent, via the streaming entrypoint."""
    saver = InMemorySaver()

    await _drain(
        g.stream_agent(
            query="first turn",
            session_id="stream-thread-resume",
            question_id="q1",
            tool_catalog=TOOL_CATALOG,
            checkpointer=saver,
        )
    )

    config = {"configurable": {"thread_id": "stream-thread-resume"}}
    graph = g.create_checkpointed_graph(saver)
    state_after_first = await graph.aget_state(config)
    messages_after_first = state_after_first.values.get("messages", [])
    assert len(messages_after_first) >= 1

    await _drain(
        g.stream_agent(
            query="second turn",
            session_id="stream-thread-resume",
            question_id="q2",
            tool_catalog=TOOL_CATALOG,
            checkpointer=saver,
        )
    )

    state_after_second = await graph.aget_state(config)
    messages_after_second = state_after_second.values.get("messages", [])
    assert len(messages_after_second) > len(messages_after_first)


# ---------------------------------------------------------------------------
# stream_agent — per-request branch (use_checkpointing + db_uri, no injected cp)
# ---------------------------------------------------------------------------


async def test_stream_agent_per_request_checkpointer_branch(saver_with_async_setup):
    with patch.object(
        g,
        "create_async_checkpointer",
        return_value=_FakeAsyncCheckpointerCM(saver_with_async_setup),
    ) as create_cp:
        events = await _drain(
            g.stream_agent(
                query="hello",
                session_id="stream-thread-per-request",
                question_id="q1",
                tool_catalog=TOOL_CATALOG,
                use_checkpointing=True,
                db_uri="sqlite:///fake",
            )
        )

        create_cp.assert_called_once_with("sqlite:///fake")
        assert events
        stream_types = {stream_type for stream_type, _ in events}
        assert "updates" in stream_types
