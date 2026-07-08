"""Re-judge an existing run with the current rubric and judge prompt.

Replays frozen question/answer/context rows through the current judge, writing
a new eval_run linked back to the original via metadata.rejudged_from. Does
not re-call the system — the answer_text and context blobs come from Postgres.

Limitation: the stored `tool_results` field is pre-formatted text produced by
the formatter that was current at original-run time. Rubric prompt changes
(e.g. mission preamble) fully apply; tool-context-format changes only apply
to questions re-run from scratch, since the raw tool call objects aren't
preserved in storage.
"""

from __future__ import annotations

import logging
from typing import Any

from src.config import settings

from .db import EvalDB
from .judge import Judge
from .rubric import DIMENSION_NAMES, compute_composite
from .runner import get_git_info

logger = logging.getLogger(__name__)


async def rejudge_run(
    original_run_id: str,
    database_url: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    db_url = database_url or settings.DATABASE_URL
    j_base = judge_base_url or settings.EVAL_JUDGE_BASE_URL or None
    j_key = judge_api_key or settings.EVAL_JUDGE_API_KEY or settings.OPENAI_API_KEY
    j_model = judge_model or settings.EVAL_JUDGE_MODEL

    db = EvalDB(db_url)
    original = db.get_run(original_run_id)
    if original is None:
        raise ValueError(f"Run {original_run_id} not found")

    judge = Judge(
        base_url=j_base, api_key=j_key, model=j_model, thinking=settings.EVAL_JUDGE_THINKING
    )
    git_info = get_git_info()
    original_meta: dict[str, Any] = original.metadata_ or {}  # type: ignore[assignment]

    new_run = db.create_run(
        run_type="rejudge",
        agent_commit=original.agent_commit,
        agent_branch=original.agent_branch,
        tool_catalog=original.tool_catalog,
        llm_model=original.llm_model,
        judge_model=j_model,
        question_set=original.question_set,
        question_count=original.question_count,
        metadata_={
            "system": original_meta.get("system"),
            "rejudged_from": original_run_id,
            "rejudge_commit": git_info.get("commit"),
            "rejudge_branch": git_info.get("branch"),
        },
    )
    logger.info(
        f"Rejudge run {new_run.id} created (source run {original_run_id}, "
        f"system={original_meta.get('system')}, judge={j_model})"
    )

    original_scores = db.get_scores_for_run(original_run_id)
    all_scores: list[dict[str, int | None]] = []
    rescored = 0
    skipped = 0
    errors = 0

    for i, score in enumerate(original_scores, 1):
        if score.source != "judge":
            skipped += 1
            continue

        context: dict[str, Any] = score.context or {}  # type: ignore[assignment]
        logger.info(f"[{i}/{len(original_scores)}] rejudging {score.question_id}")

        judge_result = await judge.score(
            query=str(score.question_text or ""),
            answer=str(score.answer_text or ""),
            rag_context=context.get("rag_context"),
            tool_results=context.get("tool_results"),
            node_trace=context.get("node_trace"),
            # The scorer freezes required_facts in context; replay them so a
            # rejudged run keeps fact-grounded verdicts (and the fact-sized
            # token budget — omitting them starved verbose judges at 500 tokens).
            required_facts=context.get("required_facts"),
        )

        if judge_result is None:
            errors += 1
            db.add_score(
                run_id=new_run.id,
                question_id=score.question_id,
                source="judge_error",
                question_text=score.question_text,
                answer_text=score.answer_text,
                context=context,
                context_completeness=score.context_completeness,
                duration_ms=score.duration_ms,
                justifications={"error": "Rejudge failed to produce valid scores"},
            )
            continue

        db.add_score(
            run_id=new_run.id,
            question_id=score.question_id,
            source="judge",
            question_text=score.question_text,
            answer_text=score.answer_text,
            context=context,
            context_completeness=score.context_completeness,
            correctness=judge_result.scores["correctness"],
            specificity=judge_result.scores["specificity"],
            specificity_na=judge_result.specificity_na,
            answerable=judge_result.answerable,
            rubric_version=2,
            relevance=judge_result.scores["relevance"],
            citation_quality=judge_result.scores["citation_quality"],
            hedging=judge_result.scores["hedging"],
            composite_score=judge_result.composite,
            duration_ms=score.duration_ms,
            justifications=judge_result.justifications,
        )
        all_scores.append(judge_result.scores)
        rescored += 1

    if all_scores:
        avg_scores = {
            name: (
                sum(v for s in all_scores if (v := s.get(name)) is not None)
                / max(1, sum(1 for s in all_scores if s.get(name) is not None))
            )
            for name in DIMENSION_NAMES
        }
        avg_composite = sum(compute_composite(s) for s in all_scores) / len(all_scores)
    else:
        avg_scores = {}
        avg_composite = 0.0

    db.update_run_summary(str(new_run.id), avg_scores, avg_composite)

    summary = {
        "new_run_id": new_run.id,
        "original_run_id": original_run_id,
        "system": original_meta.get("system"),
        "question_set": original.question_set,
        "rescored": rescored,
        "skipped": skipped,
        "errors": errors,
        "original_composite": round(original.composite_score or 0.0, 2),
        "new_composite": round(avg_composite, 2),
        "delta": round(avg_composite - (original.composite_score or 0.0), 2),
        "per_dimension": {k: round(v, 2) for k, v in avg_scores.items()},
        "judge_model": j_model,
        "rejudge_commit": git_info.get("commit", "")[:8],
    }

    logger.info(
        f"Rejudge run {new_run.id} complete: {rescored} rescored, delta={summary['delta']:+.2f}"
    )
    return summary
