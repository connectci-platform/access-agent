"""Orchestrate an eval run: load questions, run agent, judge answers, store results."""

import logging
from pathlib import Path
from typing import Any

from src.agent.profile import UserProfile
from src.config import settings
from src.llm.providers import active_model_name
from src.tools import ToolRegistry, get_catalog_aggregator

from .db import EvalDB
from .judge import Judge
from .question_facts import resolve_required_facts
from .questions import load_questions
from .rubric import DIMENSION_NAMES, compute_composite
from .runner import SYSTEM_SHORTCODES, SystemMode, gen_semantic_run_id, get_git_info, run_question
from .scoring import persist_skipped_turn, score_and_persist_turn

logger = logging.getLogger(__name__)


def preflight_fact_coverage(
    db: EvalDB,
    questions: list[Any],
    allow_factless: bool,
) -> None:
    """Refuse to score questions that resolve to no facts, before the first call.

    A fact-less question does not score zero — ``rubric.py`` renders no Required
    Facts block while still instructing the judge to grade correctness from
    per-fact verdicts, so the judge falls back to surface plausibility and the
    question scores *higher* than a graded one. Silent, and in the wrong
    direction.

    Two kinds of factless question are expected, and neither is a gap. A battery
    where NO question has facts is a smoke or coverage set that does not do fact
    scoring (friendly, real_user, mcp_coverage, ...). And a battery assembled
    from several sources — loop_smoke draws from tool_coverage, combined,
    friendly and real_user — carries questions from both kinds; those tag
    themselves with ``source_battery``, so a factless question is only a gap when
    some OTHER question from the same source has facts.
    """
    resolved = {
        q.id: resolve_required_facts(db, q.id, q.metadata.get("required_facts")) for q in questions
    }
    if not any(resolved.values()):
        return

    # Sources that grade at least one question; a factless question from any
    # other source came from a battery that does no fact scoring.
    scored_sources = {q.metadata.get("source_battery") for q in questions if resolved.get(q.id)}
    factless = sorted(
        q.id
        for q in questions
        if not resolved.get(q.id) and q.metadata.get("source_battery") in scored_sources
    )
    if not factless:
        return
    if not allow_factless:
        raise ValueError(
            f"{len(factless)} question(s) in a fact-scored battery resolve to no "
            f"required facts: {', '.join(factless)} — add facts, or pass "
            f"--allow-factless to score them ungrounded"
        )
    logger.warning(
        f"scoring {len(factless)} question(s) with no required facts: {', '.join(factless)}"
    )


async def run_eval(
    question_set_path: str,
    system: SystemMode = "agent_full",
    database_url: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
    allow_factless: bool = False,
    profile: UserProfile | None = None,
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
        # registry.tools is the flat name->definition dict tool_count is derived
        # from; the raw catalog nests tools under servers[*].tools, so a
        # top-level catalog["tools"] read comes back empty. Persist the names so
        # the coverage audit can diff served-vs-exercised (was silently []).
        "tools": sorted(registry.tools.keys()),
    }

    db = EvalDB(db_url)
    # Before the first agent call, so a coverage gap is a refusal rather than a
    # run's worth of quietly ungrounded scores.
    preflight_fact_coverage(db, questions, allow_factless)

    judge = Judge(
        base_url=j_base, api_key=j_key, model=j_model, thinking=settings.EVAL_JUDGE_THINKING
    )

    run = db.create_run(
        id=gen_semantic_run_id(SYSTEM_SHORTCODES[system]),
        run_type="pre_production",
        agent_commit=git_info.get("commit"),
        agent_branch=git_info.get("branch"),
        tool_catalog=tool_catalog_snapshot,
        # Provenance: the model that actually answers. agent_full runs the
        # tool-calling loop on the configured provider; raw_rag never touches
        # the loop LLM (answers come from the UKY /ask endpoint).
        llm_model=active_model_name() if system == "agent_full" else "uky-rag-ask",
        judge_model=j_model,
        question_set=question_set_path,
        question_count=len(questions),
        metadata_={"system": system, "profile": profile.model_dump() if profile else None},
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
            profile=profile,
        )

        # Prefer stable-id required facts from reporting.question_facts; fall back to
        # the YAML battery's required_facts when the reporting table is unavailable.
        required_facts = resolve_required_facts(db, q.id, q.metadata.get("required_facts"))

        if not result.success:
            persist_skipped_turn(
                db,
                run_id=str(run.id),
                question_id=q.id,
                question_text=q.question,
                error=result.error,
                duration_ms=result.duration_ms,
            )
            continue

        judge_result = await score_and_persist_turn(
            db,
            judge,
            run_id=str(run.id),
            question_id=q.id,
            question_text=q.question,
            answer=result.answer,
            rag_context=result.rag_context,
            tool_results=result.tool_results,
            node_trace=result.node_trace,
            required_facts=required_facts,
            extra_context={"ground_truth_stability": q.metadata.get("ground_truth_stability")},
            duration_ms=result.duration_ms,
        )
        if judge_result is None:
            continue
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
