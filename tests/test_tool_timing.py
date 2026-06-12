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
