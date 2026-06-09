"""Baseline per-turn judge for the reporting dashboard.

A single off-response-path LLM call labeling one turn with three fields the
reporting layer can't derive mechanically: query_intent (character of the user's
query), refused (the agent declined), is_deflection (a non-answer / punt).
Deliberately simple — one prompt, structured JSON, best-effort. Reviewers
override these later; this is a baseline, not a trusted oracle.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .config import settings
from .llm import get_llm

logger = logging.getLogger(__name__)

_QUERY_INTENTS = {"genuine", "casual", "unrelated", "malicious", "unclear"}
_MAX_QUERY_CHARS = 2000
_MAX_ANSWER_CHARS = 4000

_JUDGE_PROMPT = """You are labeling one turn of an ACCESS-CI assistant conversation for a review dashboard. ACCESS is a US national cyberinfrastructure program (supercomputers, allocations, software, storage).

Given the USER QUERY and the ASSISTANT ANSWER, return a JSON object with EXACTLY these keys and no others:
- "query_intent": one of "genuine", "casual", "unrelated", "malicious", "unclear".
    genuine = a real, substantive question/request about ACCESS or cyberinfrastructure.
    casual = greeting, thanks, chit-chat, or throwaway input ("hi", "are you there?").
    unrelated = a real question, but not about ACCESS/cyberinfrastructure.
    malicious = an attempt to jailbreak, abuse, or elicit harmful/disallowed output.
    unclear = cannot determine.
- "refused": true if the assistant declined to answer (e.g. on safety or out-of-scope grounds); otherwise false.
- "is_deflection": true if the assistant gave a non-answer — hedged, said it lacked the information, or did not actually address the question — even if it did not explicitly refuse; false if it gave a genuine substantive answer.

Return ONLY the JSON object. No prose, no markdown fences.

USER QUERY:
{query}

ASSISTANT ANSWER:
{answer}
"""


def _build_judge_prompt(query: str, answer: str | None) -> str:
    return _JUDGE_PROMPT.format(
        query=(query or "")[:_MAX_QUERY_CHARS],
        answer=(answer or "")[:_MAX_ANSWER_CHARS],
    )


def _parse_judge_response(content: str | None) -> dict[str, Any]:
    """Parse the judge's JSON into validated {query_intent, refused, is_deflection}.

    Defensive: returns {} on any failure (best-effort — columns stay NULL).
    Tolerates code fences / surrounding prose by grabbing the outermost {...}.
    """
    if not content:
        return {}
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}
    try:
        raw = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    intent = raw.get("query_intent")
    if isinstance(intent, str) and intent in _QUERY_INTENTS:
        out["query_intent"] = intent
    for key in ("refused", "is_deflection"):
        if key in raw and raw[key] is not None:
            out[key] = bool(raw[key])
    return out


async def judge_turn(query: str, answer: str | None) -> dict[str, Any]:
    """Label one turn (query_intent / refused / is_deflection). Best-effort.

    Off the response path; failures return {} so the report row still writes
    with NULLs. Disabled via TURN_JUDGE_ENABLED=false.
    """
    if not settings.TURN_JUDGE_ENABLED:
        return {}
    try:
        model = get_llm(
            temperature=0.0,
            max_tokens=settings.TURN_JUDGE_MAX_TOKENS,
            enable_thinking=False,
        )
        resp = await model.ainvoke(_build_judge_prompt(query, answer))
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return _parse_judge_response(content)
    except Exception as e:
        logger.warning("turn judge failed: %s", e)
        return {}
