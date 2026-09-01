"""Doc-search tool for the tool_calling_loop.

Wraps `uky_client.ask()` as a LangChain tool so the loop can decide for
itself when to consult ACCESS-CI's documentation RAG. Two responsibilities:

  1. Picking between the general and XDMoD RAG endpoints — exposed as the
     `source` parameter; tool description guides the LLM on when each
     applies.
  2. Resource-scoping by RP slug — exposed as the optional `rp_name`
     parameter, surfaced from `state.resource_context` by the loop node
     before assembling tools.

The general corpus is served by UKY's chat-mcp `/api/retrieve-docs`
endpoint, which returns ranked chunks (`documents`) with no UKY-side
synthesis spent. The tool hands those chunks to the loop so the LLM
synthesizes and cites them itself. XDMoD still uses the legacy
synthesis endpoint. The tool's outward parameter shape is unchanged.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Literal

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...services.uky_client import UKYChunk, get_uky_client
from ..domains.capabilities import get_capability_registry
from ..turn_capture import record_retrieved_chunks, record_tool_timing

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
            "Optional resource provider slug to scope the search to that RP's "
            "documentation set. The slug is the resource name lowercased with "
            "spaces and punctuation removed (e.g. 'Bridges-2' -> 'bridges2', "
            "'Delta' -> 'delta', 'Stampede3' -> 'stampede3'). Leave unset for "
            "cross-resource or general-process questions."
        ),
    )


_TOOL_NAME = "search_access_documents"

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


def _normalize_rp_name(s: str) -> str:
    # UKY accepts short lowercase slugs (bridges2, delta). The LLM may produce a
    # display-name variant (Bridges-2); normalize to the slug shape. No-op on
    # already-valid slugs (verified against UKY's set). Residual mismatches are
    # caught by the unscoped retry (a later task).
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _error_payload(exc: Exception) -> dict[str, Any]:
    """Build the structured error dict returned on a backend failure.

    Returning a dict (vs. the old prose string) lets ``_parse_tool_message``
    downstream flag ``success=False`` — see module docstring context in
    task-4-brief.md. ``status_code`` is populated for HTTP errors raised by
    ``uky_client`` and ``None`` for transport-level errors (no response to
    read a status from).
    """
    status_code = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
    return {"error": f"Documentation search failed: {exc}", "status_code": status_code}


async def _search_access_documents_inner(
    query: str,
    source: Literal["general", "xdmod"] = "general",
    rp_name: str | None = None,
) -> str | dict[str, Any]:
    """Search the ACCESS-CI documentation RAG.

    The general corpus is retrieved as raw chunks from UKY's chat-mcp
    endpoint and returned for the loop's LLM to synthesize and cite. XDMoD
    questions still use the legacy synthesis endpoint (chat-mcp is
    general-corpus only).

    The capability registry (driven by ``DISABLED_CAPABILITIES``) decides
    which RAG endpoints are reachable. An operator can disable the XDMoD
    or general endpoints via env without having to remove the tool — the
    LLM still sees the tool but disabled sources surface _UNAVAILABLE.
    Resource-scoping (``rp_name``) is similarly gated by
    ``scoped_rag_enabled``.
    """
    client = get_uky_client()
    registry = get_capability_registry()
    enabled_endpoints = registry.enabled_rag_endpoints()

    # Honor capability-registry gating: drop rp_name if scoped RAG is
    # disabled, even though the endpoint itself may still be enabled.
    if rp_name and not registry.scoped_rag_enabled():
        logger.info("search_access_documents: dropping rp_name=%s (scoped RAG disabled)", rp_name)
        rp_name = None

    if rp_name:
        rp_name = _normalize_rp_name(rp_name)

    # XDMoD: legacy synthesis endpoint.
    if source == "xdmod":
        if "xdmod" not in enabled_endpoints:
            logger.info("search_access_documents (xdmod) disabled by capability registry")
            return _UNAVAILABLE
        if not client.is_configured:
            logger.warning("search_access_documents (xdmod) called but UKY RAG is not configured")
            return _UNAVAILABLE
        try:
            result = await client.ask(query=query, endpoint_type="xdmod", rp_name=rp_name)
        except Exception as exc:
            logger.warning("search_access_documents (xdmod) failed: %s", exc)
            return _error_payload(exc)
        return result.response or (
            "Documentation search returned no content for that query. "
            "Try rephrasing or call a different tool."
        )

    # General: chunk retrieval via chat-mcp; the agent synthesizes itself.
    if "general" not in enabled_endpoints:
        logger.info("search_access_documents (general) disabled by capability registry")
        return _UNAVAILABLE
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
        return _error_payload(exc)

    record_retrieved_chunks(retrieval.chunks)
    return _format_chunks(query, retrieval.chunks)


async def _search_access_documents(
    query: str,
    source: Literal["general", "xdmod"] = "general",
    rp_name: str | None = None,
) -> str | dict[str, Any]:
    """Timing wrapper: guarantees exactly one timing record per call (1:1 invariant)."""
    start = time.monotonic()
    try:
        return await _search_access_documents_inner(query, source, rp_name)
    finally:
        record_tool_timing(_TOOL_NAME, int((time.monotonic() - start) * 1000))


search_access_documents = StructuredTool.from_function(
    func=None,
    coroutine=_search_access_documents,
    name=_TOOL_NAME,
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
