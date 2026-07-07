"""Fetch data from eval tables for report generation."""

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from .db import EvalDB

logger = logging.getLogger(__name__)


def _parse_since(since: str) -> datetime:
    """Parse '7d', '30d', '24h' into a datetime."""
    match = re.match(r"(\d+)([dhm])", since)
    if not match:
        return datetime.now(UTC) - timedelta(days=7)
    value, unit = int(match.group(1)), match.group(2)
    delta = {"d": timedelta(days=value), "h": timedelta(hours=value), "m": timedelta(minutes=value)}
    return datetime.now(UTC) - delta.get(unit, timedelta(days=7))


def build_report_data(  # noqa: PLR0912
    db: EvalDB,
    since: str = "7d",
    resource: str | None = None,
) -> dict[str, Any]:
    """Build report data dict from eval_scores."""
    since_dt = _parse_since(since)
    all_scores = db.query_scores(since=since_dt, resource=resource)

    # Build a map of run_id -> run created_at for ordering.
    # This avoids using score row timestamps (which can be skewed by late Argilla syncs).
    run_created: dict[str, datetime] = {}
    for s in all_scores:
        rid = str(s.run_id)
        if rid not in run_created:
            run = db.get_run(rid)
            if run and run.created_at:
                run_created[rid] = run.created_at  # type: ignore[assignment]

    # Group by (question_id, run_id) to avoid mixing scores across runs.
    by_question_run: dict[tuple[str, str], list[Any]] = {}
    for s in all_scores:
        key = (str(s.question_id), str(s.run_id))
        by_question_run.setdefault(key, []).append(s)

    # For each question_id, pick the latest run (by eval_runs.created_at, not score timestamps).
    latest_per_question: dict[str, tuple[str, list[Any]]] = {}  # qid -> (run_id, scores)
    for (qid, run_id), group in by_question_run.items():
        run_ts = run_created.get(run_id, datetime.min.replace(tzinfo=UTC))
        if qid not in latest_per_question:
            latest_per_question[qid] = (run_id, group)
        else:
            existing_run_id = latest_per_question[qid][0]
            existing_ts = run_created.get(existing_run_id, datetime.min.replace(tzinfo=UTC))
            if run_ts > existing_ts:
                latest_per_question[qid] = (run_id, group)

    # Within each group, prefer human scores over judge
    scores = []
    for _run_id, group in latest_per_question.values():
        human = [s for s in group if s.source == "human"]
        judge = [s for s in group if s.source == "judge"]
        if human:
            scores.extend(human)
        elif judge:
            scores.extend(judge)

    if not scores:
        return {
            "period": f"Since {since_dt.strftime('%Y-%m-%d')}",
            "resource": resource,
            "total_scored": 0,
            "human_coverage": 0.0,
            "composite_score": 0.0,
            "per_dimension": {},
            "worst_answers": [],
            "capability_gaps": [],
            "judge_human_agreement": None,
            "total_queries": 0,
            "human_reviewed": 0,
            "previous_composite": None,
            "capability_breakdown": [],
        }

    human_count = sum(1 for s in scores if s.source == "human")
    composites = [s.composite_score for s in scores if s.composite_score is not None]
    avg_composite = sum(composites) / len(composites) if composites else 0.0

    per_dimension = {}
    for dim in ["correctness", "specificity", "relevance", "citation_quality", "hedging"]:
        vals = [getattr(s, dim) for s in scores if getattr(s, dim) is not None]
        per_dimension[dim] = sum(vals) / len(vals) if vals else 0.0

    worst = sorted(scores, key=lambda s: s.composite_score or 0)[:10]
    worst_answers = [
        {
            "question": s.question_text or "",
            "composite": s.composite_score or 0,
            "question_id": s.question_id,
        }
        for s in worst
    ]

    return {
        "period": f"Since {since_dt.strftime('%Y-%m-%d')}",
        "resource": resource,
        "total_scored": len(scores),
        "total_queries": len(scores),
        "human_coverage": human_count / len(scores) if scores else 0.0,
        "human_reviewed": human_count,
        "composite_score": round(avg_composite, 2),
        "previous_composite": None,
        "per_dimension": {k: round(v, 2) for k, v in per_dimension.items()},
        "worst_answers": worst_answers,
        "capability_gaps": [],
        "judge_human_agreement": None,
        "capability_breakdown": [],
    }
