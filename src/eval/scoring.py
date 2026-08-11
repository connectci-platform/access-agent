"""Shared judge-then-persist write path for eval scores.

Both the single-turn scorer (run_eval) and the scored multi-turn runner call
these, so the eval_scores write contract lives in exactly one place.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .db import EvalDB
    from .judge import Judge, JudgeResult

logger = logging.getLogger(__name__)

# The rubric revision stamped on every judged row (formerly a literal in run_eval).
RUBRIC_VERSION = 2


async def score_and_persist_turn(
    db: "EvalDB",
    judge: "Judge",
    *,
    run_id: str,
    question_id: str,
    question_text: str,
    answer: str,
    rag_context: str | None = None,
    tool_results: str | None = None,
    node_trace: str | None = None,
    required_facts: list[Any] | None = None,
    conversation_history: list[tuple[str, str]] | None = None,
    extra_context: dict[str, Any] | None = None,
    duration_ms: float = 0.0,
) -> "JudgeResult | None":
    judge_result = await judge.score(
        query=question_text,
        answer=answer,
        rag_context=rag_context,
        tool_results=tool_results,
        node_trace=node_trace,
        required_facts=required_facts,
        conversation_history=conversation_history,
    )

    # A judge_error row carries the SAME context keys as a success row (only
    # fact_verdicts, which need a verdict, are absent), so a transient judge outage
    # leaves a row that can be manually re-scored with its full transcript.
    context: dict[str, Any] = {
        "rag_context": rag_context,
        "tool_results": tool_results,
        "node_trace": node_trace,
        "required_facts": required_facts,
    }
    if conversation_history:
        context["conversation_history"] = [list(pair) for pair in conversation_history]

    if judge_result is None:
        if extra_context:
            context.update(extra_context)
        db.add_score(
            run_id=run_id,
            question_id=question_id,
            source="judge_error",
            question_text=question_text,
            answer_text=answer,
            duration_ms=duration_ms,
            context=context,
            justifications={"error": "Judge failed to produce valid scores"},
        )
        return None

    context["fact_verdicts"] = judge_result.fact_verdicts
    if extra_context:
        context.update(extra_context)

    db.add_score(
        run_id=run_id,
        question_id=question_id,
        source="judge",
        question_text=question_text,
        answer_text=answer,
        context=context,
        context_completeness="full" if rag_context or tool_results else "partial",
        correctness=judge_result.scores["correctness"],
        specificity=judge_result.scores["specificity"],
        specificity_na=judge_result.specificity_na,
        answerable=judge_result.answerable,
        rubric_version=RUBRIC_VERSION,
        relevance=judge_result.scores["relevance"],
        citation_quality=judge_result.scores["citation_quality"],
        hedging=judge_result.scores["hedging"],
        composite_score=judge_result.composite,
        duration_ms=duration_ms,
        justifications=judge_result.justifications,
    )
    return judge_result


def persist_skipped_turn(
    db: "EvalDB",
    *,
    run_id: str,
    question_id: str,
    question_text: str,
    error: str | None,
    duration_ms: float = 0.0,
    extra_context: dict[str, Any] | None = None,
) -> None:
    db.add_score(
        run_id=run_id,
        question_id=question_id,
        source="skipped",
        question_text=question_text,
        answer_text=error or "Agent failed",
        duration_ms=duration_ms,
        context=extra_context or None,
        justifications={"error": error},
    )
