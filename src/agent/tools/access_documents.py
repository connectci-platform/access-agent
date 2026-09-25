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
from ..turn_capture import record_retrieved_chunks, record_scoped_search, record_tool_timing

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
            "available, how to interpret usage data, and links to XDMoD "
            "charts/dashboards. (For actual current usage numbers — job "
            "counts, CPU hours, utilization — prefer the get_chart_data "
            "tool over doc search.) Use 'general' "
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


_SCOPE_DROPPED_NOTE = (
    "(Note: documentation could not be scoped to the requested resource; "
    "showing general ACCESS results.)\n\n"
)

# Distinct from _SCOPE_DROPPED_NOTE (backend rejected the rp_name outright):
# this fires when the backend accepted the scoped rp_name but returned zero
# chunks — a scoped search with nothing that answers the question, not an
# invalid slug. See docs/superpowers/specs/2026-09-23-profile-ab-results.md
# mechanism 3 ("no answer at all when the scoped resource lacks the thing
# asked about").
_ZERO_CHUNKS_SCOPED_NOTE = (
    "(No documentation matched within the `<rp>` scope; showing general ACCESS results.)\n\n"
)


def _repeat_scoped_search_note(rp_name: str) -> str:
    """Distinct note for the repeat-scoped-search widening (see
    _search_access_documents_inner's general-path repeat check) — a
    different failure than either _SCOPE_DROPPED_NOTE (backend rejected the
    slug) or _ZERO_CHUNKS_SCOPED_NOTE (accepted but zero chunks): this is a
    second scoped call this turn to a slug that already returned something,
    just not something that answered the question.
    """
    return (
        f"(A second scoped search for `{rp_name}` this turn; showing general "
        "ACCESS results instead.)\n\n"
    )


def _is_invalid_rp_name(exc: Exception) -> bool:
    """True iff ``exc`` is UKY's 400 for an rp_name it doesn't recognize."""
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    if exc.response.status_code != 400:
        return False
    return "invalid rp_name" in exc.response.text.lower()


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


async def _retry_xdmod_unscoped(
    query: str, rp_name: str | None, exc: Exception
) -> dict[str, Any] | str | None:
    """Handle the xdmod except-block's invalid-rp_name retry.

    Returns ``None`` if ``exc`` doesn't qualify for a retry (caller should
    fall through to its own ``_error_payload(exc)``); otherwise returns the
    final str/dict result of the retry attempt.
    """
    if not rp_name or not _is_invalid_rp_name(exc):
        return None
    logger.warning(
        "search_access_documents (xdmod): UKY rejected rp_name=%r; retrying unscoped", rp_name
    )
    client = get_uky_client()
    try:
        result = await client.ask(query=query, endpoint_type="xdmod", rp_name=None)
    except Exception as exc2:
        logger.warning("search_access_documents unscoped retry failed: %s", exc2)
        return _error_payload(exc2)
    return _SCOPE_DROPPED_NOTE + (
        result.response
        or (
            "Documentation search returned no content for that query. "
            "Try rephrasing or call a different tool."
        )
    )


async def _retry_general_unscoped(
    query: str, rp_name: str | None, exc: Exception
) -> dict[str, Any] | str | None:
    """Handle the general except-block's invalid-rp_name retry.

    Returns ``None`` if ``exc`` doesn't qualify for a retry (caller should
    fall through to its own ``_error_payload(exc)``); otherwise returns the
    final str/dict result of the retry attempt.
    """
    if not rp_name or not _is_invalid_rp_name(exc):
        return None
    logger.warning(
        "search_access_documents (general): UKY rejected rp_name=%r; retrying unscoped", rp_name
    )
    client = get_uky_client()
    try:
        retrieval = await client.retrieve(query=query, rp_name=None)
    except Exception as exc2:
        logger.warning("search_access_documents unscoped retry failed: %s", exc2)
        return _error_payload(exc2)
    record_retrieved_chunks(retrieval.chunks)
    return _SCOPE_DROPPED_NOTE + _format_chunks(query, retrieval.chunks)


async def _retry_general_unscoped_on_zero_chunks(query: str, rp_name: str) -> str:
    """Retry a scoped general search unscoped after a zero-chunk result.

    Called only when the backend *accepted* ``rp_name`` and returned zero
    chunks — distinct from ``_retry_general_unscoped``'s invalid-rp_name 400
    path. Always returns the final (unscoped) text, since a second empty
    result is still the best answer available ("no excerpts" guidance).
    """
    logger.info(
        "search_access_documents (general): scoped rp_name=%r returned zero chunks; "
        "retrying unscoped",
        rp_name,
    )
    client = get_uky_client()
    retrieval = await client.retrieve(query=query, rp_name=None)
    record_retrieved_chunks(retrieval.chunks)
    return _ZERO_CHUNKS_SCOPED_NOTE + _format_chunks(query, retrieval.chunks)


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

    if source == "xdmod":
        return await _search_xdmod(client, enabled_endpoints, query, rp_name)
    return await _search_general(client, enabled_endpoints, query, rp_name)


async def _search_xdmod(
    client: Any,
    enabled_endpoints: set[str],
    query: str,
    rp_name: str | None,
) -> str | dict[str, Any]:
    """XDMoD path: legacy synthesis endpoint."""
    if "xdmod" not in enabled_endpoints:
        logger.info("search_access_documents (xdmod) disabled by capability registry")
        return _UNAVAILABLE
    if not client.is_configured:
        logger.warning("search_access_documents (xdmod) called but UKY RAG is not configured")
        return _UNAVAILABLE
    try:
        result = await client.ask(query=query, endpoint_type="xdmod", rp_name=rp_name)
    except Exception as exc:
        retried = await _retry_xdmod_unscoped(query, rp_name, exc)
        if retried is not None:
            return retried
        logger.warning("search_access_documents (xdmod) failed: %s", exc)
        return _error_payload(exc)
    return result.response or (
        "Documentation search returned no content for that query. "
        "Try rephrasing or call a different tool."
    )


async def _search_general(
    client: Any,
    enabled_endpoints: set[str],
    query: str,
    rp_name: str | None,
) -> str | dict[str, Any]:
    """General path: chunk retrieval via chat-mcp; the agent synthesizes itself."""
    if "general" not in enabled_endpoints:
        logger.info("search_access_documents (general) disabled by capability registry")
        return _UNAVAILABLE
    if not client.is_chatmcp_configured:
        logger.warning(
            "search_access_documents called but chat-mcp is not configured "
            "(UKY_RAG_ENABLED=False or no UKY_CHATMCP_API_KEY)"
        )
        return _UNAVAILABLE

    # A second scoped call to the SAME normalized rp_name this turn means the
    # first scoped search already returned something that didn't answer the
    # question (otherwise the model wouldn't be asking again) — widen to
    # unscoped instead of repeating the scoped search, which is what let the
    # loop spin until the recursion limit in production. Checked before the
    # call so the repeat itself is never made scoped.
    if rp_name and record_scoped_search(rp_name):
        logger.info(
            "search_access_documents (general): repeat scoped search for "
            "rp_name=%r this turn; widening to unscoped",
            rp_name,
        )
        retrieval = await client.retrieve(query=query, rp_name=None)
        record_retrieved_chunks(retrieval.chunks)
        return _repeat_scoped_search_note(rp_name) + _format_chunks(query, retrieval.chunks)

    try:
        retrieval = await client.retrieve(query=query, rp_name=rp_name)
    except Exception as exc:
        retried = await _retry_general_unscoped(query, rp_name, exc)
        if retried is not None:
            return retried
        logger.warning("search_access_documents (chat-mcp) failed: %s", exc)
        return _error_payload(exc)

    if rp_name and not retrieval.chunks:
        return await _retry_general_unscoped_on_zero_chunks(query, rp_name)

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
