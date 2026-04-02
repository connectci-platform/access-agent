"""LLM judge for scoring agent answers against the rubric."""

import json
import logging
import re
from dataclasses import dataclass

from openai import AsyncOpenAI

from .rubric import DIMENSION_NAMES, build_judge_prompt, compute_composite

logger = logging.getLogger(__name__)


@dataclass
class JudgeResult:
    scores: dict[str, int]
    justifications: dict[str, str]
    composite: float


def parse_judge_response(raw: str) -> JudgeResult | None:
    """Parse judge LLM response into JudgeResult. Returns None if malformed."""
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Judge returned invalid JSON")
        return None

    scores: dict[str, int] = {}
    justifications: dict[str, str] = {}

    for name in DIMENSION_NAMES:
        if name not in data:
            logger.warning(f"Judge response missing dimension: {name}")
            return None
        entry = data[name]
        if not isinstance(entry, dict) or "score" not in entry:
            logger.warning(f"Judge response malformed for dimension: {name}")
            return None
        score = entry["score"]
        if not isinstance(score, int) or score < 1 or score > 5:
            logger.warning(f"Judge score out of range for {name}: {score}")
            return None
        scores[name] = score
        justifications[name] = entry.get("justification", "")

    return JudgeResult(
        scores=scores,
        justifications=justifications,
        composite=compute_composite(scores),
    )


class Judge:
    """LLM-based answer quality judge. Configurable endpoint for cloud or on-premise."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
        )

    async def score(
        self,
        query: str,
        answer: str,
        rag_context: str | None = None,
        tool_results: str | None = None,
        node_trace: str | None = None,
    ) -> JudgeResult | None:
        prompt = build_judge_prompt(
            query=query,
            answer=answer,
            rag_context=rag_context,
            tool_results=tool_results,
            node_trace=node_trace,
        )

        for attempt in range(2):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=500,
                )
                raw = response.choices[0].message.content or ""
                result = parse_judge_response(raw)
                if result is not None:
                    return result
                if attempt == 0:
                    logger.warning("Judge parse failed, retrying with stricter prompt")
                    prompt += "\n\nIMPORTANT: Return ONLY the JSON object. No other text."
            except Exception as e:
                logger.error(f"Judge LLM call failed (attempt {attempt + 1}): {e}")

        logger.error("Judge failed after 2 attempts")
        return None
