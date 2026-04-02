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
        description="Does the answer accurately represent its sources (RAG docs + tool results)?",
        low="Contradicts sources or hallucinates facts",
        high="Faithfully represents all source material",
    ),
    Dimension(
        name="completeness",
        description="Does the answer address all parts of the question?",
        low="Misses the main point",
        high="Thoroughly covers the question",
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

## Scoring Rubric

Score each dimension from 1 (worst) to 5 (best):

{rubric_text}

## Important

- Judge whether the agent accurately represented the information it HAD ACCESS TO.
- If the source documents contain outdated information and the agent faithfully reported it, that is CORRECT (score 5 on correctness). Data quality is not the agent's fault.
- If the agent added information not in the sources, that is a hallucination (score 1-2 on correctness).

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
