"""Answer "what can you do?" from the capability registry, without the model.

The widget's /query route and the full-screen client's thread runs both use
this. On /query it short-circuits the route; on a thread run it is a graph node
that START routes to, so the answer is an ordinary AIMessage in the thread and
streams, checkpoints, reports and rates like any other.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime  # noqa: TC002 - LangGraph reads node annotations at runtime

from ..context import AgentContext, require_context
from ..domains.capabilities import get_capability_registry

# Whole-message phrasings only. "What can you do about my quota?" is a real
# question for the model, so this is an exact match after normalizing.
_CAPABILITY_QUESTIONS = frozenset(
    {
        "show my options",
        "what can you do",
        "what can you help with",
        "what can you help me with",
    }
)

_TRAILING_PUNCTUATION = re.compile(r"[\s?!.]+$")
_WHITESPACE = re.compile(r"\s+")


def is_capability_question(text: str) -> bool:
    """Whether a message asks what the assistant can do, as a whole message.

    Ignores case, surrounding whitespace, trailing ``?``/``!``/``.``, and the
    lock emoji the widget prefixes to auth-required buttons.
    """
    normalized = text.strip().removeprefix("🔒").strip().lower()
    normalized = _TRAILING_PUNCTUATION.sub("", normalized)
    return _WHITESPACE.sub(" ", normalized) in _CAPABILITY_QUESTIONS


async def capability_categories(
    authenticated: bool, resource_context: str | None = None
) -> list[dict[str, Any]]:
    """The caller's capability categories, RP-scoped when ``resource_context``
    names a known resource. Disabled capabilities are never in the registry."""
    registry = get_capability_registry()
    if resource_context:
        scoped = await registry.get_by_category_scoped(resource_context, authenticated)
        if scoped:
            categories: list[dict[str, Any]] = scoped["categories"]
            return categories
    return registry.get_by_category(authenticated)


def capability_overview(categories: list[dict[str, Any]], authenticated: bool) -> str:
    """Markdown list of categories with an example question per capability.

    Sign-in-required capabilities are marked with 🔒 for an anonymous caller.
    The "general" category is skipped, since asking anything is the default.
    """
    lines = ["Here are some things you can try:\n"]
    for cat in categories:
        if cat["id"] == "general":
            continue
        lines.append(f"**{cat['label']}**")
        for cap in cat["capabilities"]:
            example = cap.get("example_query") or cap["description"]
            locked = " 🔒 (login required)" if cap.get("locked") else ""
            lines.append(f'- *"{example}"*{locked}')
        lines.append("")
    if not authenticated:
        lines.append("*Some features require logging in. Log in to unlock all capabilities.*")
    return "\n".join(lines).rstrip()


async def capability_discovery_node(
    state: dict[str, Any], runtime: Runtime[AgentContext]
) -> dict[str, Any]:
    """Reply to a capability question with the registry's overview.

    Returns the whole thread plus the reply, as tool_calling_loop does, because
    the run stream sends the node's messages to the client as the thread. The
    per-turn fields are reset so a checkpointed thread doesn't carry the
    previous turn's tools into this turn's report.
    """
    authenticated = require_context(runtime.context).acting_user is not None
    categories = await capability_categories(authenticated, state.get("resource_context"))
    answer = capability_overview(categories, authenticated)
    return {
        "final_answer": answer,
        "messages": [*state.get("messages", []), AIMessage(content=answer, id=str(uuid.uuid4()))],
        "tools_used": [],
        "tool_results": [],
        "tool_refusals": [],
        "tool_call_count": 0,
        "answer_unavailable": False,
        "total_tokens": None,
        "model_calls": None,
        "node_trace": [{"node": "capability_discovery", "answer_length": len(answer)}],
    }
