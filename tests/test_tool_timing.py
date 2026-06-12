"""Per-tool-call timing: capture at the call site, pairing in the loop."""

import asyncio

from src.agent.domains.tools import MCPToolWrapper
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
