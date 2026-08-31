from __future__ import annotations

from typing import Any

from src.probe.runner import run_probe
from src.probe.table import ProbeCase
from src.tools.mcp_client import MCPToolResult

SMALL_TABLE = [
    ProbeCase(server="allocations", tool_name="broken_tool", args={}),
    ProbeCase(server="events", tool_name="search_events", args={"date": "upcoming", "limit": 20}),
    ProbeCase(server="nsf-awards", tool_name="search_nsf_awards", args={"query": "cyberinfrastructure"}),
]

_RESULTS_BY_TOOL: dict[str, MCPToolResult] = {
    "broken_tool": MCPToolResult(success=False, error="HTTP 400"),
    "search_events": MCPToolResult(success=True, data={"events": [{"id": 1}]}),
    "search_nsf_awards": MCPToolResult(success=True, data={}),
}


class FakeMCPClient:
    async def call_tool(
        self, server: str, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        return _RESULTS_BY_TOOL[tool_name]


async def test_run_probe_flags_backend_error_and_passes_empty_result() -> None:
    results = await run_probe(FakeMCPClient(), SMALL_TABLE)

    assert [r.success for r in results] == [False, True, True]
    assert results[0].error == "HTTP 400"
    assert results[0].tool_name == "broken_tool"


class RaisingMCPClient:
    async def call_tool(
        self, server: str, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult:
        if tool_name == "search_events":
            raise RuntimeError("otel export failed")
        return _RESULTS_BY_TOOL[tool_name]


async def test_run_probe_isolates_a_raising_case_and_continues() -> None:
    results = await run_probe(RaisingMCPClient(), SMALL_TABLE)

    assert [r.tool_name for r in results] == [
        "broken_tool",
        "search_events",
        "search_nsf_awards",
    ]
    assert [r.success for r in results] == [False, False, True]
    assert results[1].tool_name == "search_events"
    assert results[1].error is not None
    assert "otel export failed" in results[1].error
