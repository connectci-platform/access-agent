"""Runs the synthetic per-tool health probe table against a live MCPClient.

Deterministic, no-LLM: each case is a direct MCPClient.call_tool with known-good
args. A case is a failure only on a transport/HTTP backend error; a clean empty
result is a pass (MCPToolResult already sets success=True on the happy path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from ..tools.mcp_client import MCPToolResult
    from .table import ProbeCase


@dataclass(frozen=True)
class ProbeResult:
    tool_name: str
    success: bool
    error: str | None


class ToolCaller(Protocol):
    async def call_tool(
        self, server: str, tool_name: str, arguments: dict[str, Any]
    ) -> MCPToolResult: ...


async def run_probe(client: ToolCaller, table: list[ProbeCase]) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for case in table:
        outcome = await client.call_tool(case.server, case.tool_name, case.args)
        results.append(
            ProbeResult(
                tool_name=case.tool_name,
                success=outcome.success,
                error=outcome.error,
            )
        )
    return results
