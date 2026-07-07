"""Orchestrate an eval run: load questions, run agent, judge answers, store results."""

import logging
from pathlib import Path
from typing import Any

from src.config import settings
from src.tools import ToolRegistry, get_catalog_aggregator

from .db import EvalDB
from .judge import Judge
from .question_facts import resolve_required_facts
from .questions import load_questions
from .rubric import DIMENSION_NAMES, compute_composite
from .runner import SystemMode, gen_semantic_run_id, get_git_info, run_question

logger = logging.getLogger(__name__)


async def run_eval(
    question_set_path: str,
    system: SystemMode = "agent_full",
    database_url: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    db_url = database_url or settings.DATABASE_URL
    j_base = judge_base_url or settings.EVAL_JUDGE_BASE_URL or None
    j_key = judge_api_key or settings.EVAL_JUDGE_API_KEY or settings.OPENAI_API_KEY
    j_model = judge_model or settings.EVAL_JUDGE_MODEL

    questions = load_questions(question_set_path)
    logger.info(f"Loaded {len(questions)} questions from {question_set_path}")

    logger.info("Loading tool catalog...")
    aggregator = get_catalog_aggregator()
    catalog = await aggregator.fetch_catalog()
    registry = ToolRegistry(catalog=catalog)
    logger.info(f"Loaded {registry.tool_count} tools")

    git_info = get_git_info()
    tool_catalog_snapshot = {
        "tool_count": registry.tool_count,
        "tools": [t.get("name", "unknown") for t in (registry.catalog or {}).get("tools", [])],
    }

    db = EvalDB(db_url)
    judge = Judge(base_url=j_base, api_key=j_key, model=j_model)

    run = db.create_run(
        id=gen_semantic_run_id(system),
        run_type="pre_production",
        agent_commit=git_info.get("commit"),
        agent_branch=git_info.get("branch"),
        tool_catalog=tool_catalog_snapshot,
        llm_model=settings.OPENAI_MODEL,
        judge_model=j_model,
        question_set=question_set_path,
        question_count=len(questions),
        metadata_={"system": system},
    )
    logger.info(f"Eval run {run.id} started ({len(questions)} questions, system={system})")

    all_scores: list[dict[str, int | None]] = []
    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {q.question[:60]}...")

        result = await run_question(
            q.id,
            q.question,
            registry.catalog,
            system=system,
            resource_context=q.metadata.get("resource"),
            battery_id=Path(question_set_path).stem,
            battery_run_id=str(run.id),
        )

        # Prefer stable-id required facts from reporting.question_facts; fall back to
        # the YAML battery's required_facts when the reporting table is unavailable.
        required_facts = resolve_required_facts(db, q.id, q.metadata.get("required_facts"))

        if not result.success:
            db.add_score(
                run_id=run.id,
                question_id=q.id,
                source="skipped",
                question_text=q.question,
                answer_text=result.error or "Agent failed",
                duration_ms=result.duration_ms,
                justifications={"error": result.error},
            )
            continue

        judge_result = await judge.score(
            query=q.question,
            answer=result.answer,
            rag_context=result.rag_context,
            tool_results=result.tool_results,
            node_trace=result.node_trace,
            required_facts=required_facts,
        )

        if judge_result is None:
            db.add_score(
                run_id=run.id,
                question_id=q.id,
                source="judge_error",
                question_text=q.question,
                answer_text=result.answer,
                duration_ms=result.duration_ms,
                context={
                    "rag_context": result.rag_context,
                    "tool_results": result.tool_results,
                },
                justifications={"error": "Judge failed to produce valid scores"},
            )
            continue

        db.add_score(
            run_id=run.id,
            question_id=q.id,
            source="judge",
            question_text=q.question,
            answer_text=result.answer,
            context={
                "rag_context": result.rag_context,
                "tool_results": result.tool_results,
                "node_trace": result.node_trace,
                "required_facts": required_facts,
                "fact_verdicts": judge_result.fact_verdicts,
                "ground_truth_stability": q.metadata.get("ground_truth_stability"),
            },
            context_completeness="full" if result.rag_context or result.tool_results else "partial",
            correctness=judge_result.scores["correctness"],
            specificity=judge_result.scores["specificity"],
            specificity_na=judge_result.specificity_na,
            answerable=judge_result.answerable,
            rubric_version=2,
            relevance=judge_result.scores["relevance"],
            citation_quality=judge_result.scores["citation_quality"],
            hedging=judge_result.scores["hedging"],
            composite_score=judge_result.composite,
            duration_ms=result.duration_ms,
            justifications=judge_result.justifications,
        )
        all_scores.append(judge_result.scores)

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

    db.update_run_summary(str(run.id), avg_scores, avg_composite)

    summary = {
        "run_id": run.id,
        "system": system,
        "questions": len(questions),
        "scored": len(all_scores),
        "skipped": len(questions) - len(all_scores),
        "composite_score": round(avg_composite, 2),
        "per_dimension": {k: round(v, 2) for k, v in avg_scores.items()},
        "agent_commit": git_info.get("commit", "")[:8],
        "agent_branch": git_info.get("branch", ""),
    }

    logger.info(f"Eval run {run.id} complete: composite={avg_composite:.2f}")
    return summary
