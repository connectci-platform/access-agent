"""Orchestrate an eval run: load questions, run agent, judge answers, store results."""

import logging
from typing import Any

from src.config import settings
from src.tools import ToolRegistry, get_catalog_aggregator

from .db import EvalDB
from .judge import Judge
from .questions import load_questions
from .rubric import DIMENSION_NAMES, compute_composite
from .runner import SystemMode, gen_semantic_run_id, get_git_info, run_question

logger = logging.getLogger(__name__)


async def run_eval(  # noqa: PLR0915
    question_set_path: str,
    system: SystemMode = "agent_full",
    database_url: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
    push_argilla: bool = False,
) -> dict[str, Any]:
    # Override USE_TOOL_CALLING_LOOP based on --system choice (CLI is authoritative for eval runs).
    # agent_full → loop (new default); agent_full_legacy → legacy plan→execute chain.
    # Other systems (agent_rag_only, raw_rag) don't exercise the flag, so we leave the state
    # as whatever it resolved to but force a predictable value for log clarity.
    flag_value = system == "agent_full"
    if flag_value != settings.USE_TOOL_CALLING_LOOP:
        logger.info(
            f"Overriding USE_TOOL_CALLING_LOOP from {settings.USE_TOOL_CALLING_LOOP} "
            f"to {flag_value} for --system {system}"
        )
    settings.USE_TOOL_CALLING_LOOP = flag_value

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

    all_scores: list[dict[str, int]] = []
    for i, q in enumerate(questions, 1):
        logger.info(f"[{i}/{len(questions)}] {q.question[:60]}...")

        result = await run_question(
            q.id,
            q.question,
            registry.catalog,
            system=system,
            resource_context=q.metadata.get("resource"),
        )

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
            },
            context_completeness="full" if result.rag_context or result.tool_results else "partial",
            correctness=judge_result.scores["correctness"],
            completeness=judge_result.scores["completeness"],
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
            name: sum(s[name] for s in all_scores) / len(all_scores) for name in DIMENSION_NAMES
        }
        avg_composite = sum(compute_composite(s) for s in all_scores) / len(all_scores)
    else:
        avg_scores = {}
        avg_composite = 0.0

    db.update_run_summary(str(run.id), avg_scores, avg_composite)

    if push_argilla:
        from .argilla_push import (
            build_argilla_record,
            dataset_name_for_branch,
            push_scores_to_argilla,
        )

        ds_name = dataset_name_for_branch(git_info.get("branch"))
        # Push low-scoring answers for review (composite < 4.5), plus all errors/skips
        argilla_score_threshold = 4.5
        argilla_records = []
        for score in db.get_scores_for_run(str(run.id)):
            if score.source not in ("judge", "judge_error", "skipped"):
                continue
            if score.source == "judge" and (score.composite_score or 0) >= argilla_score_threshold:
                continue
            score_context: dict[str, Any] = score.context or {}  # type: ignore[assignment]
            argilla_records.append(
                build_argilla_record(
                    question_id=str(score.question_id),
                    question_text=str(score.question_text or ""),
                    answer_text=str(score.answer_text or ""),
                    judge_scores={
                        "correctness": int(score.correctness or 0),
                        "completeness": int(score.completeness or 0),
                        "relevance": int(score.relevance or 0),
                        "citation_quality": int(score.citation_quality or 0),
                        "hedging": int(score.hedging or 0),
                    },
                    composite_score=float(score.composite_score or 0.0),
                    rag_context=score_context.get("rag_context"),
                    tool_results=score_context.get("tool_results"),
                    node_trace=score_context.get("node_trace"),
                    run_id=str(run.id),
                    agent_branch=git_info.get("branch"),
                    agent_commit=git_info.get("commit"),
                    judge_model=j_model,
                    duration_ms=float(score.duration_ms) if score.duration_ms else None,
                )
            )

        if argilla_records:
            pushed = push_scores_to_argilla(
                argilla_records,
                settings.ARGILLA_URL,
                settings.ARGILLA_API_KEY,
                ds_name,
            )
            logger.info(f"Pushed {pushed} records to Argilla dataset '{ds_name}'")

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
