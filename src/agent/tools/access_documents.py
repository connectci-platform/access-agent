"""Doc-search tool for the no-classify tool_calling_loop.

Wraps `uky_client.ask()` as a LangChain tool so the loop can decide for
itself when to consult ACCESS-CI's documentation RAG. Subsumes two jobs
the legacy `classify` node used to do up-front:

  1. Picking between the general and XDMoD RAG endpoints — exposed as the
     `source` parameter; tool description guides the LLM on when each
     applies.
  2. Resource-scoping by RP slug — exposed as the optional `rp_name`
     parameter, surfaced from `state.resource_context` by the loop node
     before assembling tools.

When `/retrieve` ships, the body of `_search_access_documents` swaps to
that endpoint; the tool's outward parameter shape stays close to the
same so the loop's prompt and any downstream consumers don't churn.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ...services.uky_client import get_uky_client

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


async def _search_access_documents(
    query: str,
    source: Literal["general", "xdmod"] = "general",
    rp_name: str | None = None,
) -> str:
    """Search the ACCESS-CI documentation RAG and return the response text."""
    client = get_uky_client()
    if not client.is_configured:
        logger.warning(
            "search_access_documents called but UKY RAG client is not "
            "configured (UKY_RAG_ENABLED=False or no API key)"
        )
        return (
            "Documentation search is currently unavailable. "
            "Try answering from your other tools or tell the user the doc "
            "search is offline."
        )

    try:
        result = await client.ask(
            query=query,
            endpoint_type=source,
            rp_name=rp_name,
        )
    except Exception as exc:
        logger.warning("search_access_documents failed: %s", exc)
        return f"Documentation search failed: {exc}. Try another approach."

    return result.response or (
        "Documentation search returned no content for that query. "
        "Try rephrasing or call a different tool."
    )


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
