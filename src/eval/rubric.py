"""Scoring rubric for agent answer evaluation.

Five dimensions, each scored 1-5. Weights are configurable.
The same rubric is used by both the LLM judge and human reviewers in Argilla.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Dimension:
    name: str
    description: str
    low: str
    high: str


DIMENSIONS = [
    Dimension(
        name="correctness",
        description="Does the answer accurately represent its sources? Tool results (live API data) take precedence over RAG docs when they conflict.",
        low="Contradicts tool results or hallucinates facts not in any source",
        high="Faithfully represents tool results and relevant RAG content",
    ),
    Dimension(
        name="completeness",
        description="Does the answer address all parts of the question with the most appropriate type of information? Specific data (resource names, version numbers, event dates, ticket confirmations) is more complete than general guidance when the question calls for specifics.",
        low="Misses the main point, or gives only general guidance when specific data was needed",
        high="Thoroughly covers the question with concrete, specific information",
    ),
    Dimension(
        name="relevance",
        description="Does the answer stay on topic without padding?",
        low="Mostly irrelevant content",
        high="Focused and directly addresses the query",
    ),
    Dimension(
        name="citation_quality",
        description="Are URLs present, valid, and from source docs?",
        low="No URLs or hallucinated URLs",
        high="All relevant URLs preserved from sources",
    ),
    Dimension(
        name="hedging",
        description="Does the answer admit uncertainty when sources are thin, avoid over-hedging when strong?",
        low="Confidently wrong or hedges everything",
        high="Calibrated confidence matching source quality",
    ),
]

DIMENSION_NAMES = [d.name for d in DIMENSIONS]

DEFAULT_WEIGHTS: dict[str, float] = {
    "correctness": 0.30,
    "completeness": 0.25,
    "relevance": 0.20,
    "citation_quality": 0.15,
    "hedging": 0.10,
}


def compute_composite(
    scores: dict[str, int],
    weights: dict[str, float] | None = None,
) -> float:
    """Compute weighted composite score from dimension scores."""
    w = weights or DEFAULT_WEIGHTS
    return sum(scores[name] * w[name] for name in DIMENSION_NAMES)


def build_judge_prompt(
    query: str,
    answer: str,
    rag_context: str | None = None,
    tool_results: str | None = None,
    node_trace: str | None = None,
) -> str:
    """Build the LLM judge prompt with the rubric and context."""
    rubric_text = "\n".join(
        f"- **{d.name}** (1-5): {d.description}\n  1 = {d.low}\n  5 = {d.high}" for d in DIMENSIONS
    )

    context_sections = []
    if rag_context:
        context_sections.append(f"## RAG Documents Retrieved\n{rag_context}")
    if tool_results:
        context_sections.append(f"## Tool Results\n{tool_results}")
    if node_trace:
        context_sections.append(f"## Agent Decision Trace\n{node_trace}")

    context_text = "\n\n".join(context_sections) if context_sections else "No context available."

    return f"""You are evaluating the quality of an AI agent's answer to a user question.

## Mission

This agent supports researchers using ACCESS-CI (the US national cyberinfrastructure allocation system). Users ask about compute resources, software availability, allocations, system status, events, and how to get help. They need specific, current, actionable information — named resources, software versions, event dates, ticket confirmations, exact counts. Generic how-to guidance is less valuable than concrete data, because the user's goal is usually to DO something (run a job, request help, find a resource) not to read documentation.

When the system takes an action on the user's behalf (creates a support ticket, looks up live allocation data), that is a meaningfully better outcome than pointing the user at a URL and asking them to do it themselves.

## How to read the context sections

You will see up to three kinds of context the agent had:

- **RAG Documents Retrieved**: curated Q&A snippets from documentation. Static; not live. May be stale.
- **Tool Results**: structured records of live tool calls, one record per call. Each record includes the tool name, the arguments the agent passed, whether the call succeeded, how long it took, an explicit `result_count` and `empty` flag, and the raw data. Use these to judge whether the agent called the right tool with the right arguments, whether the tool returned useful data, and whether the agent represented that data faithfully in its answer. An `empty: true` record means the tool returned no data on its own terms — not that there is no data on the topic anywhere.
- **Agent Decision Trace**: the ordered list of graph nodes the agent went through (classify, plan, execute, evaluate, synthesize, etc.), with each node's key decisions.

## Scoring Rubric

Score each dimension from 1 (worst) to 5 (best):

{rubric_text}

## Important

- Judge whether the agent accurately represented the information it HAD ACCESS TO.
- If the source documents contain outdated information and the agent faithfully reported it, that is CORRECT (score 5 on correctness). Data quality is not the agent's fault.
- If the agent added information not in the sources, that is a hallucination (score 1-2 on correctness).
- CRITICAL: Tool Results are LIVE DATA from real-time APIs and are MORE CURRENT than RAG Documents. When tool results return POSITIVE DATA that conflicts with RAG documents (e.g., tool says "Delta has 4 GPU nodes" but RAG says 8), the agent is CORRECT to trust the tool results. Score the agent based on whether it accurately represented the tool results.
- HOWEVER: Tool results returning 0 items or empty results represent ABSENCE of data, not contradiction of other sources. Do not penalize an answer for relying on RAG documents just because a tool search returned no results — the search may not have matched, or the data may not be in that tool's scope. Only treat tool results as overriding RAG when the tool returns positive data that conflicts with the RAG answer.
- If tool results show 0 items AND the RAG documents have relevant content, the agent is CORRECT to use the RAG content. Do not penalize this.

## Completeness: Specificity and Action

In judging completeness, you are looking for specific examples. Specific examples are concrete items like named resources (e.g., "Anvil", "Expanse", "Bridges-2"), specific software with versions (e.g., "Anaconda3 version 2020.11"), named events with dates, or exact counts and statistics. An answer that describes a category ("several resources support GPUs") without naming them is NOT specific. An answer that names them ("Anvil has NVIDIA A100s, Expanse has V100s, ACES has H100s") IS specific.

You may find that an answer contains a list of examples. Count the specific, individually named items (not categories). If the answer lists 6 or more specific named items, completeness may be scored 5. If the answer lists fewer than 6 specific named items, score completeness no higher than 4. Do not count general categories or types of things (e.g., "workshops on AI, cybersecurity, and data management") — those are categories, not specific items.

Apply these rules:
- When the question asks for CURRENT or SPECIFIC information (e.g., "what events are coming up", "which resources have X installed", "show me allocation statistics") and the answer provides only general/static guidance without specific names, versions, dates, or counts, score completeness 3 or lower. A correct general answer to a specific question is incomplete.
- When the question asks "which resources" or "where can I" and the answer does NOT include a list of specifically named resources, score completeness no higher than 3 — even if the general advice is correct.
- When the system TAKES AN ACTION on behalf of the user (e.g., creates a support ticket, files a report) rather than merely suggesting the user take that action themselves, that is more complete. An answer that says "a ticket has been created (ticket ATS-12345)" is more complete than "you should open a ticket at this URL."
- When the answer includes real-time data (live event listings, current software versions, system status) alongside documentation, it is more complete than documentation alone — the user gets both the how-to and the current state.

## User Question

{query}

## Agent Answer

{answer}

## Context the Agent Had Access To

{context_text}

## Your Response

Return a JSON object with this exact structure (no other text):
```json
{{
  "correctness": {{"score": <1-5>, "justification": "<brief explanation>"}},
  "completeness": {{"score": <1-5>, "justification": "<brief explanation>"}},
  "relevance": {{"score": <1-5>, "justification": "<brief explanation>"}},
  "citation_quality": {{"score": <1-5>, "justification": "<brief explanation>"}},
  "hedging": {{"score": <1-5>, "justification": "<brief explanation>"}}
}}
```"""
