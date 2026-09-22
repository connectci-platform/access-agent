"""Scoring rubric for agent answer evaluation (v2).

Five scored dimensions on a 3-point ordinal scale (hedging is 2-point),
plus a pre-score answerability screen that is NOT a scored dimension.
Higher integer = better. See docs/collaboration/2026-07-06-eval-rubric-v2.md.
"""

from dataclasses import dataclass
from typing import Any

from src.agent.profile import UserProfile


@dataclass(frozen=True)
class Dimension:
    name: str
    description: str
    anchors: list[str]  # best -> worst, one label per ordinal value (highest first)


DIMENSIONS = [
    Dimension(
        name="correctness",
        description=(
            "Are the answer's claims true, judged against the question's authored "
            "required_facts (the ground truth), NOT surface plausibility?"
        ),
        anchors=["Correct", "Partial", "Incorrect"],
    ),
    Dimension(
        name="specificity",
        description=(
            "Does the answer give concrete, actionable info (named resources, versions, "
            "dates, counts, live values) vs. generic guidance? Scored independently of "
            "correctness. 'N/A' when the question does not call for specifics."
        ),
        anchors=["Actionable", "Mixed", "Generic"],  # N/A handled separately (see parser)
    ),
    Dimension(
        name="relevance",
        description="Does the answer address the question without padding?",
        anchors=["On-target", "Partial", "Off"],
    ),
    Dimension(
        name="citation_quality",
        description="Are source URLs present, valid, and drawn from the actual source material?",
        anchors=["Good", "Fair", "Poor"],
    ),
    Dimension(
        name="hedging",
        description=(
            "Does the answer honestly signal how much to trust it, to a user who cannot "
            "verify it? Over-confident on shaky claims OR needless hedging = miscalibrated."
        ),
        anchors=["Calibrated", "Miscalibrated"],
    ),
]

DIMENSION_NAMES = [d.name for d in DIMENSIONS]

# Highest ordinal integer per dimension (values run max..0). hedging is 2-point.
DIMENSION_MAX: dict[str, int] = {d.name: len(d.anchors) - 1 for d in DIMENSIONS}

# Map each dimension's anchor labels to their ordinal integers (best label -> max).
DIMENSION_LABELS: dict[str, dict[str, int]] = {
    d.name: {label: DIMENSION_MAX[d.name] - i for i, label in enumerate(d.anchors)}
    for d in DIMENSIONS
}

# Configurable/overridable DEFAULT — NOT a hardcoded constant. compute_composite accepts a
# `weights` override so the split can be tuned per-comparison as we learn (spec §5.3, RESOLVED).
# Correctness-dominant (fact-grounded correctness matters most, guards confident-wrong);
# specificity second as the actionable-vs-generic discriminator.
DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.40,
    "specificity": 0.25,
    "relevance": 0.15,
    "citation_quality": 0.12,
    "hedging": 0.08,
}


def compute_composite(
    scores: dict[str, int | None],
    weights: dict[str, float] | None = None,
) -> float:
    """Weighted composite on [0,1].

    Each dimension is normalized to [0,1] by its own max before weighting, so the
    2-point hedging axis is not out-weighted by the 3-point axes. A None value
    (e.g. specificity N/A) is skipped and its weight is renormalized away.
    """
    w = weights or DEFAULT_WEIGHTS
    active = {name: w[name] for name in DIMENSION_NAMES if scores.get(name) is not None}
    total_w = sum(active.values())
    if total_w == 0:
        return 0.0
    acc = 0.0
    for name, weight in active.items():
        norm = scores[name] / DIMENSION_MAX[name]  # type: ignore[operator]
        acc += norm * (weight / total_w)
    return acc


def flatten_required_facts(
    facts: list[str | dict[str, Any]],
) -> list[tuple[str, str]]:
    """Flatten required_facts into [(id, text), ...] for prompt rendering.

    A fact dict carrying a stable ``fact_id`` (e.g. loaded from
    reporting.question_facts) keeps that id, so fact verdicts stay joinable
    across runs even when the authored fact list is edited. Its text comes from
    ``fact_text`` (the reporting column) or ``text`` (defensive fallback).

    Legacy shapes keep positional ids: plain string facts and ``{heading, items}``
    dicts are numbered F1, F2, ... in document order. Stable-id facts do NOT
    consume a positional slot.
    """
    out: list[tuple[str, str]] = []
    counter = 1
    for fact in facts:
        if isinstance(fact, dict) and "fact_id" in fact and ("fact_text" in fact or "text" in fact):
            text_val = fact.get("fact_text", fact.get("text"))
            out.append((str(fact["fact_id"]), str(text_val)))
        elif isinstance(fact, str):
            out.append((f"F{counter}", fact))
            counter += 1
        elif isinstance(fact, dict) and "heading" in fact and "items" in fact:
            heading = str(fact["heading"]).rstrip(":")
            for item in fact["items"]:
                out.append((f"F{counter}", f"{heading}: {item}"))
                counter += 1
    return out


def _render_request_profile_section(profile: UserProfile | None) -> str:
    """Render the judge-prompt "## Request profile" section, or "" when absent.

    See docs/superpowers/specs/2026-09-21-profile-ab-grading-decisions.md
    ("The mechanism"): the no-profile arm must render nothing so its prompt
    stays byte-identical to the pre-profile prompt.
    """
    if profile is None or not profile.allocated_resources:
        return ""

    listing = ", ".join(
        f"{r.name} (rp_name='{r.rp_slug}')" if r.rp_slug else r.name
        for r in profile.allocated_resources
    )
    return f"""

## Request profile

The request supplied a user profile naming these allocated resources: {listing}.
Where a required fact asks the answer to ask which resource the user means, or to make explicit
that the answer is resource-dependent, an answer that answers for one of the supplied resources
and says so satisfies that fact. Facts about not presenting one resource's specifics as true of
all ACCESS resources are unchanged: an answer that is scoped to a supplied resource but still
states that resource's values as ACCESS-wide fails them.
Facts for questions that are not about any particular resource, such as ACCESS account or
password matters, are also unchanged: answering such a question with resource-specific detail
does not satisfy them.
"""


def build_judge_prompt(
    query: str,
    answer: str,
    rag_context: str | None = None,
    tool_results: str | None = None,
    node_trace: str | None = None,
    required_facts: list[str | dict[str, Any]] | None = None,
    conversation_history: list[tuple[str, str]] | None = None,
    profile: UserProfile | None = None,
) -> str:
    """Build the LLM judge prompt with the rubric and context."""
    rubric_text = "\n".join(
        f"- **{d.name}**: {d.description}\n  Values (best to worst): "
        + " / ".join(d.anchors)
        + ("" if d.name != "specificity" else " / N/A (question doesn't call for specifics)")
        for d in DIMENSIONS
    )

    context_sections = []
    if rag_context:
        context_sections.append(f"## RAG Documents Retrieved\n{rag_context}")
    if tool_results:
        context_sections.append(f"## Tool Results\n{tool_results}")
    if node_trace:
        context_sections.append(f"## Agent Decision Trace\n{node_trace}")

    context_text = "\n\n".join(context_sections) if context_sections else "No context available."

    has_history = bool(conversation_history)

    history_source_clause = (
        " by the prior turns shown in 'Conversation so far'," if has_history else ""
    )

    facts_support_enum = (
        "the Tool Results, RAG Documents, the prior turns shown in 'Conversation so far', or "
        "well-known ACCESS-CI facts"
        if has_history
        else "the Tool Results, RAG Documents, or well-known ACCESS-CI facts"
    )
    facts_history_clause = (
        " A value the agent correctly restates from an earlier turn (visible in "
        "'Conversation so far') IS supported."
        if has_history
        else ""
    )

    profile_section = _render_request_profile_section(profile)

    facts_section = ""
    facts_response_schema = ""
    if required_facts:
        flat = flatten_required_facts(required_facts)
        facts_lines = "\n".join(f"{fid}. {text}" for fid, text in flat)
        facts_section = f"""

## Required Facts

A correct answer for this question must support each of the following claims. For each one, rate whether the agent's answer supports it:
- "yes" — the answer states the claim AND the claim is supported by {facts_support_enum}. A specific name or number stated only in the answer (not in the context) is not evidence — that is a hallucination, score it accordingly on the relevant fact.{facts_history_clause}
- "partial" — the answer touches on the claim but is incomplete, vague, imprecise, OR mixes a correct statement with hallucinated specifics
- "no" — the answer omits or contradicts the claim. If the answer is silent on the claim, that is "no", not "yes" — do not infer support from absence of contradiction

{facts_lines}
"""
        verdict_lines = ",\n    ".join(
            f'{{"id": "{fid}", "verdict": "<yes|partial|no>", "justification": "<brief>"}}'
            for fid, _ in flat
        )
        facts_response_schema = f""",
  "required_facts": [
    {verdict_lines}
  ]"""

    history_section = ""
    if conversation_history:
        turns = []
        for i, (q, a) in enumerate(conversation_history, 1):
            turns.append(f"Turn {i} — User: {q}\nTurn {i} — Assistant: {a}")
        history_section = (
            "## Conversation so far\n\n"
            "This is a multi-turn conversation. Earlier turns, oldest first:\n\n"
            + "\n\n".join(turns)
            + "\n\nThe answer you are evaluating responds to the latest user question "
            "below, in the context of this conversation. Judge reference resolution "
            '("that", "those", "the first one") against these earlier turns.\n\n'
        )

    return f"""You are evaluating the quality of an AI agent's answer to a user question.

## Mission

This agent supports researchers using ACCESS-CI (the US national cyberinfrastructure allocation system). Users ask about compute resources, software availability, allocations, system status, events, and how to get help. They need specific, current, actionable information — named resources, software versions, event dates, ticket confirmations, exact counts. Generic how-to guidance is less valuable than concrete data, because the user's goal is usually to DO something (run a job, request help, find a resource) not to read documentation.

When the system takes an action on the user's behalf (creates a support ticket, looks up live allocation data), that is a meaningfully better outcome than pointing the user at a URL and asking them to do it themselves.

## How to read the context sections

You will see up to three kinds of context the agent had:

- **RAG Documents Retrieved**: curated Q&A snippets from documentation. Static; not live. May be stale.
- **Tool Results**: structured records of live tool calls, one record per call. Each record includes the tool name, the arguments the agent passed, whether the call succeeded, how long it took, an explicit `result_count` and `empty` flag, and the raw data. Use these to judge whether the agent called the right tool with the right arguments, whether the tool returned useful data, and whether the agent represented that data faithfully in its answer. An `empty: true` record means the tool returned no data on its own terms — not that there is no data on the topic anywhere.
- **Agent Decision Trace**: a record from the agent's `tool_calling_loop` node, with the number of tool calls it made, which tools it called, and how many tool results it received.

**Treat the context as your source of truth, not the answer.** The agent's answer is what you are grading. When the answer makes a specific factual claim — names a resource, group, person, software version, count, date, URL, ticket number — that claim must be supported either by the Tool Results, by the RAG Documents,{history_source_clause} or by widely known ACCESS-CI facts you are confident about. A specific name or number that appears only in the answer and nowhere in the context is unsupported, and should be treated as a hallucination. Penalize unsupported specifics in the relevant rubric dimensions and in the per-fact verdicts below.

## Answerability screen (do this FIRST, before scoring)

Decide whether this is a fair, answerable question:
- "Fair" — a reasonable question the system could answer.
- "Unfair" — off-topic, adversarial, or unanswerable without context the system lacks.
If "Unfair", still return the screen but the dimension scores will be ignored.

## Scoring Rubric

Choose exactly one labeled value per dimension (use the anchor label strings verbatim):

{rubric_text}

For `correctness`, base your label on the per-fact required_facts verdicts below:
mostly-supported -> "Correct", mixed -> "Partial", missing/contradicted key facts -> "Incorrect".
For `specificity`, use "N/A" only when the question is a process/how-to that does not call
for concrete named specifics.

## Important

- Judge whether the agent accurately represented the information it HAD ACCESS TO.
- If the source documents contain outdated information and the agent faithfully reported it, that is Correct on correctness. Data quality is not the agent's fault.
- If the agent added information not in the sources, that is a hallucination (Incorrect on correctness).
- CRITICAL: Tool Results are LIVE DATA from real-time APIs and are MORE CURRENT than RAG Documents. When tool results return POSITIVE DATA that conflicts with RAG documents (e.g., tool says "Delta has 4 GPU nodes" but RAG says 8), the agent is CORRECT to trust the tool results. Score the agent based on whether it accurately represented the tool results.
- HOWEVER: Tool results returning 0 items or empty results represent ABSENCE of data, not contradiction of other sources. Do not penalize an answer for relying on RAG documents just because a tool search returned no results — the search may not have matched, or the data may not be in that tool's scope. Only treat tool results as overriding RAG when the tool returns positive data that conflicts with the RAG answer.
- If tool results show 0 items AND the RAG documents have relevant content, the agent is CORRECT to use the RAG content. Do not penalize this.

{history_section}## User Question

{query}

## Agent Answer

{answer}

## Context the Agent Had Access To

{context_text}{profile_section}{facts_section}

## Your Response

Return a JSON object with this exact structure (no other text):
```json
{{
  "answerable": "<Fair|Unfair>",
  "correctness": {{"value": "<Correct|Partial|Incorrect>", "justification": "<brief>"}},
  "specificity": {{"value": "<Actionable|Mixed|Generic|N/A>", "justification": "<brief>"}},
  "relevance": {{"value": "<On-target|Partial|Off>", "justification": "<brief>"}},
  "citation_quality": {{"value": "<Good|Fair|Poor>", "justification": "<brief>"}},
  "hedging": {{"value": "<Calibrated|Miscalibrated>", "justification": "<brief>"}}{facts_response_schema}
}}
```"""
