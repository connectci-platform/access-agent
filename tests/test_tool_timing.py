"""Per-tool-call timing: capture at the call site, pairing in the loop."""

import asyncio

from langchain_core.messages import AIMessage, ToolMessage

from src.agent.domains.tools import MCPToolWrapper
from src.agent.nodes.tool_calling_loop import _build_tool_results
from src.agent.turn_capture import (
    get_turn_capture,
    reset_turn_capture,
)
from src.tools.mcp_client import MCPClient, MCPToolResult


def test_mcp_wrapper_records_tool_timing(monkeypatch):
    reset_turn_capture()
    client = MCPClient()

    async def fake_call_tool(**kwargs):
        return MCPToolResult(success=True, data={"ok": 1}, duration_ms=42)

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    wrapper = MCPToolWrapper(
        name="list_things",
        description="test tool",
        tool_server="test-server",
        mcp_client=client,
    )
    asyncio.run(wrapper._arun())
    assert get_turn_capture()["tool_timings"] == [{"tool_name": "list_things", "duration_ms": 42}]


def test_doc_search_records_timing(monkeypatch):
    from src.agent.tools.access_documents import _search_access_documents

    reset_turn_capture()

    class _Retrieval:
        chunks: list = []  # noqa: RUF012

    class _Client:
        is_chatmcp_configured = True

        async def retrieve(self, query, rp_name=None):
            return _Retrieval()

    class _Registry:
        def enabled_rag_endpoints(self):
            return {"general", "xdmod"}

        def scoped_rag_enabled(self):
            return True

    monkeypatch.setattr("src.agent.tools.access_documents.get_uky_client", _Client)
    monkeypatch.setattr(
        "src.agent.tools.access_documents.get_capability_registry",
        _Registry,
    )
    asyncio.run(_search_access_documents("how do I use globus"))
    timings = get_turn_capture()["tool_timings"]
    assert len(timings) == 1
    assert timings[0]["tool_name"] == "search_access_documents"
    assert timings[0]["duration_ms"] >= 0


def test_mcp_wrapper_records_timing_on_failure(monkeypatch):
    reset_turn_capture()
    client = MCPClient()

    async def fake_call_tool(**kwargs):
        return MCPToolResult(success=False, error="boom", duration_ms=17)

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    wrapper = MCPToolWrapper(
        name="list_things",
        description="test tool",
        tool_server="test-server",
        mcp_client=client,
    )
    asyncio.run(wrapper._arun())
    assert get_turn_capture()["tool_timings"] == [{"tool_name": "list_things", "duration_ms": 17}]


def test_doc_search_records_timing_when_unavailable(monkeypatch):
    from src.agent.tools.access_documents import _search_access_documents

    reset_turn_capture()

    class _Client:
        is_chatmcp_configured = False

    class _Registry:
        def enabled_rag_endpoints(self):
            return {"general", "xdmod"}

        def scoped_rag_enabled(self):
            return True

    monkeypatch.setattr("src.agent.tools.access_documents.get_uky_client", _Client)
    monkeypatch.setattr(
        "src.agent.tools.access_documents.get_capability_registry",
        _Registry,
    )
    asyncio.run(_search_access_documents("anything"))
    timings = get_turn_capture()["tool_timings"]
    assert len(timings) == 1
    assert timings[0]["tool_name"] == "search_access_documents"


def _thread_two_calls():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "list_things", "args": {"a": 1}, "id": "c1"},
            {"name": "list_things", "args": {"a": 2}, "id": "c2"},
        ],
    )
    return [
        ai,
        ToolMessage(content="{}", tool_call_id="c1"),
        ToolMessage(content="{}", tool_call_id="c2"),
    ]


def test_build_tool_results_pairs_timings_fifo():
    results, orphans = _build_tool_results(
        _thread_two_calls(),
        [],
        [
            {"tool_name": "list_things", "duration_ms": 42},
            {"tool_name": "list_things", "duration_ms": 7},
        ],
    )
    assert orphans == 0
    assert [r.duration_ms for r in results] == [42, 7]


def test_build_tool_results_without_timings_defaults_zero():
    results, _ = _build_tool_results(_thread_two_calls(), [], [])
    assert [r.duration_ms for r in results] == [0, 0]


def test_record_current_trace_id_writes_32_hex():
    from opentelemetry.sdk.trace import TracerProvider

    from src.agent.graph import _record_current_trace_id

    reset_turn_capture()
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("agent.run") as span:
        _record_current_trace_id(span)
    tid = get_turn_capture()["trace_id"]
    assert tid is not None and len(tid) == 32
    assert int(tid, 16) != 0


def test_record_current_trace_id_noop_for_non_recording_span():
    from opentelemetry.trace import INVALID_SPAN

    from src.agent.graph import _record_current_trace_id

    reset_turn_capture()
    _record_current_trace_id(INVALID_SPAN)
    assert get_turn_capture()["trace_id"] is None


def test_build_tool_results_skips_prior_turn_messages():
    from langchain_core.messages import HumanMessage

    prior_ai = AIMessage(
        content="",
        tool_calls=[{"name": "list_things", "args": {"a": 0}, "id": "t1"}],
    )
    current_ai = AIMessage(
        content="",
        tool_calls=[{"name": "list_things", "args": {"a": 1}, "id": "t2"}],
    )
    thread = [
        HumanMessage(content="first question"),
        prior_ai,
        ToolMessage(content="{}", tool_call_id="t1"),
        AIMessage(content="first answer"),
        HumanMessage(content="second question"),
        current_ai,
        ToolMessage(content="{}", tool_call_id="t2"),
    ]
    timings = [{"tool_name": "list_things", "duration_ms": 900}]
    results, orphans = _build_tool_results(thread, [], timings)
    assert orphans == 0
    by_id = {r.step_id: r.duration_ms for r in results}
    assert by_id == {"t1": 0, "t2": 900}


def test_build_tool_results_annotates_source_message_id():
    """Each entry carries its ToolMessage's LangChain id, so multi-turn consumers can
    slice the cumulative rebuild by membership in the current turn's id set."""
    from langchain_core.messages import HumanMessage

    thread = [
        HumanMessage(content="first question", id="h1"),
        AIMessage(
            content="", tool_calls=[{"name": "list_things", "args": {}, "id": "t1"}], id="a1"
        ),
        ToolMessage(content="{}", tool_call_id="t1", id="tm1"),
        AIMessage(content="first answer", id="a2"),
        HumanMessage(content="second question", id="h2"),
        AIMessage(
            content="", tool_calls=[{"name": "list_things", "args": {}, "id": "t2"}], id="a3"
        ),
        ToolMessage(content="{}", tool_call_id="t2", id="tm2"),
    ]
    results, _ = _build_tool_results(thread, [], [])

    assert {r.step_id: r.message_id for r in results} == {"t1": "tm1", "t2": "tm2"}
    # Those ids straddle the boundary the delta slices on (last HumanMessage at 4).
    boundary = max(i for i, m in enumerate(thread) if isinstance(m, HumanMessage))
    current_ids = {m.id for m in thread[boundary + 1 :]}
    assert [r.step_id for r in results if r.message_id in current_ids] == ["t2"]


def test_build_tool_results_annotates_error_results_too():
    """The failure branch of _parse_tool_message must carry the message id as well."""
    thread = [
        AIMessage(
            content="", tool_calls=[{"name": "list_things", "args": {}, "id": "e1"}], id="a1"
        ),
        ToolMessage(content='{"error": "boom"}', tool_call_id="e1", id="tm-err"),
    ]
    results, _ = _build_tool_results(thread, [], [])
    assert len(results) == 1
    assert results[0].success is False
    assert results[0].message_id == "tm-err"


def test_build_tool_results_logs_leftover_timings(caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="src.agent.nodes.tool_calling_loop"):
        results, _ = _build_tool_results(
            _thread_two_calls(), [], [{"tool_name": "other_tool", "duration_ms": 5}]
        )
    assert [r.duration_ms for r in results] == [0, 0]
    assert "no matching current-turn ToolMessage" in caplog.text


def test_run_agent_records_trace_id(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider

    import src.agent.graph as graph_mod

    class _Graph:
        async def ainvoke(self, state, config):
            return {}

    monkeypatch.setattr(graph_mod, "create_agent_graph", _Graph)
    monkeypatch.setattr(graph_mod, "get_tracer", lambda name="t": TracerProvider().get_tracer(name))
    reset_turn_capture()
    asyncio.run(graph_mod.run_agent(query="q", session_id="s", question_id="qid", tool_catalog={}))
    assert get_turn_capture()["trace_id"] is not None


def test_stream_agent_records_trace_id(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider

    import src.agent.graph as graph_mod

    class _Graph:
        async def astream(self, state, config=None, stream_mode=None):
            return
            yield  # makes this an (empty) async generator

    monkeypatch.setattr(graph_mod, "create_agent_graph", _Graph)
    monkeypatch.setattr(graph_mod, "get_tracer", lambda name="t": TracerProvider().get_tracer(name))
    reset_turn_capture()

    async def _consume():
        async for _ in graph_mod.stream_agent(
            query="q", session_id="s", question_id="qid", tool_catalog={}
        ):
            pass

    asyncio.run(_consume())
    assert get_turn_capture()["trace_id"] is not None
