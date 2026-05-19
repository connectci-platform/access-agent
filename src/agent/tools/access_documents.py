"""Doc-search tool for the tool_calling_loop.

Wraps `uky_client.ask()` as a LangChain tool so the loop can decide for
itself when to consult ACCESS-CI's documentation RAG. Two responsibilities:

  1. Picking between the general and XDMoD RAG endpoints — exposed as the
     `source` parameter; tool description guides the LLM on when each
     applies.
  2. Resource-scoping by RP slug — exposed as the optional `rp_name`
     parameter, surfaced from `state.resource_context` by the loop node
     before assembling tools.

The general corpus is served by UKY's chat-mcp endpoint, which returns
raw retrieval chunks (`top_documents`). The tool hands those chunks to
the loop so the LLM synthesizes and cites them itself, rather than
consuming UKY's own synthesis. XDMoD still uses the legacy synthesis
endpoint. The tool's outward parameter shape is unchanged.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...services.uky_client import UKYChunk, get_uky_client

logger = logging.getLogger(__name__)


class _SearchAccessDocumentsArgs(BaseModel):
    """Arguments for `search_access_documents`."""

    query: str = Field(
        description=(
            "The natural-language question or topic to look up in the "
            "ACCESS-CI documentation RAG. Pass the user's actual question, "
            "not a keyword distillation."
        ),
    )
    source: Literal["general", "xdmod"] = Field(
        default="general",
        description=(
            "Which documentation corpus to search. Use 'xdmod' for "
            "questions about XDMoD's features, what usage metrics are "
            "available, how to interpret usage data, links to XDMoD "
            "charts/dashboards, and aggregate-across-ACCESS questions "
            "(job counts, CPU hours, GPU utilization, allocations, "
            "gateways, projects, storage, capacity). Use 'general' "
            "(the default) for ACCESS-CI documentation: how-tos, "
            "policies, allocations process, hardware specs, software, "
            "Globus, MFA, identity, and similar reference material."
        ),
    )
    rp_name: str | None = Field(
        default=None,
        description=(
            "Optional resource provider slug (e.g. 'delta', 'bridges-2', "
            "'expanse', 'anvil') to scope the search to that RP's "
            "documentation set. Leave unset for cross-resource or "
            "general-process questions."
        ),
    )


_UNAVAILABLE = (
    "Documentation search is currently unavailable. "
    "Try answering from your other tools or tell the user the doc "
    "search is offline."
)


def _format_chunks(query: str, chunks: list[UKYChunk]) -> str:
    """Render retrieval chunks into text the loop's LLM can synthesize from."""
    if not chunks:
        return (
            "Documentation search returned no excerpts for that query. "
            "Try rephrasing or call a different tool."
        )
    parts = [
        f'Retrieved {len(chunks)} documentation excerpt(s) for "{query}". '
        "Synthesize an answer from these excerpts and cite the source URLs. "
        "If none are actually relevant to the question, say so plainly."
    ]
    for chunk in chunks:
        source = chunk.url or "(no source URL)"
        parts.append(f"[{chunk.rank}] source: {source}\n{chunk.text.strip()}")
    return "\n\n".join(parts)


async def _search_access_documents(
    query: str,
    source: Literal["general", "xdmod"] = "general",
    rp_name: str | None = None,
) -> str:
    """Search the ACCESS-CI documentation RAG.

    The general corpus is retrieved as raw chunks from UKY's chat-mcp
    endpoint and returned for the loop's LLM to synthesize and cite. XDMoD
    questions still use the legacy synthesis endpoint (chat-mcp is
    general-corpus only).
    """
    client = get_uky_client()

    # XDMoD: legacy synthesis endpoint.
    if source == "xdmod":
        if not client.is_configured:
            logger.warning("search_access_documents (xdmod) called but UKY RAG is not configured")
            return _UNAVAILABLE
        try:
            result = await client.ask(query=query, endpoint_type="xdmod", rp_name=rp_name)
        except Exception as exc:
            logger.warning("search_access_documents (xdmod) failed: %s", exc)
            return f"Documentation search failed: {exc}. Try another approach."
        return result.response or (
            "Documentation search returned no content for that query. "
            "Try rephrasing or call a different tool."
        )

    # General: chunk retrieval via chat-mcp; the agent synthesizes itself.
    if not client.is_chatmcp_configured:
        logger.warning(
            "search_access_documents called but chat-mcp is not configured "
            "(UKY_RAG_ENABLED=False or no UKY_CHATMCP_API_KEY)"
        )
        return _UNAVAILABLE
    try:
        retrieval = await client.retrieve(query=query, rp_name=rp_name)
    except Exception as exc:
        logger.warning("search_access_documents (chat-mcp) failed: %s", exc)
        return f"Documentation search failed: {exc}. Try another approach."

    return _format_chunks(query, retrieval.chunks)


search_access_documents = StructuredTool.from_function(
    func=None,
    coroutine=_search_access_documents,
    name="search_access_documents",
    description=(
        "Search the ACCESS-CI documentation RAG for how-tos, policies, "
        "concepts, hardware/software references, and XDMoD documentation. "
        "Call this when the user asks for documented, stable, reference-style "
        "information — process explanations ('how do I get an allocation'), "
        "policy ('what are the password requirements'), how-to guides "
        "('how do I use Globus'), or anything that lives in ACCESS-CI's "
        "documentation. Use 'source=xdmod' for XDMoD-specific or "
        "aggregate-metrics questions, otherwise leave 'source' as 'general'. "
        "Pass 'rp_name' to scope to a specific resource provider when the "
        "user asks about one."
    ),
    args_schema=_SearchAccessDocumentsArgs,
)
