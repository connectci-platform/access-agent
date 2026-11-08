"""Answer "what can you do?" without the model.

The widget's /query route and the full-screen client's thread runs share the
matcher. /query short-circuits with a per-capability list
(``widget_capability_overview``). A thread run goes to
``capability_discovery_node``, which START routes to, and answers with the
hand-written areas in ``domains/capability_areas.py``. That answer is an
ordinary AIMessage in the thread, so it checkpoints, reports and rates like any
other, and it carries its example questions as ``suggestions`` in
``additional_kwargs`` for the client to show as buttons.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage
from langgraph.runtime import Runtime  # noqa: TC002 - LangGraph reads node annotations at runtime

from ..context import AgentContext, require_context
from ..domains.capabilities import get_capability_registry
from ..domains.capability_areas import AREA_BY_CAPABILITY, CAPABILITY_AREAS

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


def widget_capability_overview(categories: list[dict[str, Any]], authenticated: bool) -> str:
    """The widget's answer: categories with an example question per capability.

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


OVERVIEW_INTRO = "I can help you find and use ACCESS resources and services."
SIGN_IN_NOTE = "Sign in to see your own allocations and usage, and to register for events."


def area_overview(
    enabled: set[str], authenticated: bool, *, your_resources: str | None = None
) -> tuple[str, list[str]]:
    """The full-screen answer's markdown and its suggested questions, in area order.

    ``enabled`` is the registry's enabled capability ids. ``your_resources`` is
    the seam for D8-3028: a markdown section about the signed-in user's own
    resources, placed above the areas. Nothing passes it yet.
    """
    shown = {AREA_BY_CAPABILITY.get(cap_id) for cap_id in enabled}
    blocks = [OVERVIEW_INTRO]
    if your_resources:
        blocks.append(your_resources)
    suggestions: list[str] = []
    for area in CAPABILITY_AREAS:
        if area.id not in shown:
            continue
        # Two trailing spaces are a hard line break; GFM joins plain newlines.
        blocks.append(f"**{area.title}**  \n{area.description}")
        suggestions.extend(ex.text for ex in area.examples if ex.capability in enabled)
    if not authenticated:
        blocks.append(SIGN_IN_NOTE)
    return "\n\n".join(blocks), suggestions


async def capability_discovery_node(
    state: dict[str, Any], runtime: Runtime[AgentContext]
) -> dict[str, Any]:
    """Reply to a capability question with the area overview.

    Returns the whole thread plus the reply, as tool_calling_loop does, because
    the run stream sends the node's messages to the client as the thread. The
    per-turn fields are reset so a checkpointed thread doesn't carry the
    previous turn's tools into this turn's report.
    """
    authenticated = require_context(runtime.context).acting_user is not None
    enabled = {cap.id for cap in get_capability_registry().get_all()}
    answer, suggestions = area_overview(enabled, authenticated)
    reply = AIMessage(
        content=answer,
        id=str(uuid.uuid4()),
        additional_kwargs={"capability_overview": True, "suggestions": suggestions},
    )
    return {
        "final_answer": answer,
        "messages": [*state.get("messages", []), reply],
        "tools_used": [],
        "tool_results": [],
        "tool_refusals": [],
        "tool_call_count": 0,
        "answer_unavailable": False,
        "total_tokens": None,
        "model_calls": None,
        "node_trace": [{"node": "capability_discovery", "answer_length": len(answer)}],
    }
