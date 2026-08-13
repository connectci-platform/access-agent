"""Fetch data from eval tables for report generation."""

import logging
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from .db import EvalDB
from .rubric import DIMENSION_NAMES

logger = logging.getLogger(__name__)


def _per_dimension_means(scores: list[Any]) -> dict[str, float]:
    """Mean per v2 dimension, excluding None (e.g. specificity N/A) from each mean."""
    out: dict[str, float] = {}
    for dim in DIMENSION_NAMES:
        vals = [getattr(s, dim) for s in scores if getattr(s, dim, None) is not None]
        out[dim] = sum(vals) / len(vals) if vals else 0.0
    return out


def _parse_since(since: str) -> datetime:
    """Parse '7d', '30d', '24h' into a datetime."""
    match = re.match(r"(\d+)([dhm])", since)
    if not match:
        return datetime.now(UTC) - timedelta(days=7)
    value, unit = int(match.group(1)), match.group(2)
    delta = {"d": timedelta(days=value), "h": timedelta(hours=value), "m": timedelta(minutes=value)}
    return datetime.now(UTC) - delta.get(unit, timedelta(days=7))


def _index_runs(db: EvalDB, scores: list[Any]) -> tuple[dict[str, datetime], dict[str, str | None]]:
    """Memoized run lookup: run_id -> created_at (ordering) and run_id -> mode (fail-closed filter).

    One db.get_run per unique run_id — including run_ids whose lookup comes back
    missing, tracked in ``seen`` so a second score referencing the same orphaned
    run_id doesn't trigger another db.get_run call or another warning. A run_id
    present in run_mode means the run row was FOUND — that's the "found" test the
    fail-closed filter relies on. A missing run is logged (once) and simply absent
    from both maps.
    """
    run_created: dict[str, datetime] = {}
    run_mode: dict[str, str | None] = {}
    seen: set[str] = set()
    for s in scores:
        rid = str(s.run_id)
        if rid in seen:
            continue
        seen.add(rid)
        run = db.get_run(rid)
        if run is None:
            logger.warning(f"Score {s.question_id} references missing run {rid}; excluded")
            continue
        if run.created_at:
            run_created[rid] = run.created_at  # type: ignore[assignment]
        run_meta: dict[str, Any] = run.metadata_ or {}  # type: ignore[assignment]
        run_mode[rid] = run_meta.get("mode")
    return run_created, run_mode


def build_report_data(
    db: EvalDB,
    since: str = "7d",
    resource: str | None = None,
) -> dict[str, Any]:
    """Build report data dict from eval_scores."""
    since_dt = _parse_since(since)
    all_scores = db.query_scores(since=since_dt, resource=resource)

    # run_created orders "latest per question" below by eval_runs.created_at (not score
    # row timestamps, which can be skewed by late Argilla syncs). run_mode backs the
    # fail-closed filter directly below.
    run_created, run_mode = _index_runs(db, all_scores)

    # Fail-closed filter: keep a score only if its run row was FOUND (rid in run_mode)
    # and is not a multiturn run. A score whose run lookup fails is excluded, not
    # silently included — otherwise multiturn rows leak into the dashboard exactly
    # when this lookup breaks.
    kept = [
        s for s in all_scores if (rid := str(s.run_id)) in run_mode and run_mode[rid] != "multiturn"
    ]
    dropped = len(all_scores) - len(kept)
    if dropped:
        logger.info(f"Report excludes {dropped} score(s) from multiturn/missing runs")
    all_scores = kept

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

    # Answerability screen: excluded items never enter aggregates (spec §6).
    scores = [s for s in scores if getattr(s, "answerable", None) is not False]

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

    per_dimension = _per_dimension_means(scores)

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
