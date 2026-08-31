"""Runs the docs tool (`search_access_documents`) health case against UKY's
chat-mcp retrieve endpoint directly, alongside the MCP-server probe table.

`search_access_documents` does not go through an MCP server — it calls
`UKYClient.retrieve` (src/services/uky_client.py) directly. This case is
deliberately separate from `run_probe`/`ToolCaller` rather than shoehorned
into the MCPClient protocol: a clean, un-scoped query (no `rp_name`) is used
because a resource-scoped `rp_name` is exactly the agent-side bug this probe
must not reproduce — this case tests tool HEALTH with known-good args, same
as every other probe case.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from .runner import ProbeResult

if TYPE_CHECKING:
    from ..services.uky_client import UKYRetrieval

DOCS_TOOL_NAME = "search_access_documents"
_DOCS_PROBE_QUERY = "ACCESS-CI documentation"


class DocsRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        session_id: str = "",
        question_id: str = "",
        rp_name: str | None = None,
    ) -> UKYRetrieval: ...


async def probe_docs_tool(client: DocsRetriever | None = None) -> ProbeResult:
    retriever: DocsRetriever
    if client is None:
        from ..services.uky_client import get_uky_client

        retriever = get_uky_client()
    else:
        retriever = client

    try:
        await retriever.retrieve(query=_DOCS_PROBE_QUERY)
    except Exception as exc:  # a broken docs tool must not abort the whole probe
        return ProbeResult(
            tool_name=DOCS_TOOL_NAME,
            success=False,
            error=f"probe call raised: {exc}",
        )
    return ProbeResult(tool_name=DOCS_TOOL_NAME, success=True, error=None)
