"""Grand-prix subcommand: runs the full Production Baseline Comparison in-process.

For each battery, runs raw_rag + agent_full, then runs compare-judge on the
pair, then renders one HTML report from all the per-battery JSON artifacts.

Uses the existing run_eval, compare_runs, and build_report_from_json
functions — no duplicated orchestration logic. Run IDs are captured in
memory as each eval completes, so no Postgres polling is needed.

Defaults to the four canonical batteries (friendly, real_user, combined,
mcp_coverage). Pass --batteries to restrict to a subset.

--skip-runs mode: skip phase 1 and use the most-recent raw_rag + agent_full
runs for each battery on the current branch. Useful for iterating on the
compare-judge + HTML phases without burning an hour on fresh runs.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import settings

from .compare_judge import compare_runs
from .db import EvalDB
from .html_report.builder import build_report_from_json
from .scorer import run_eval

logger = logging.getLogger(__name__)

DEFAULT_BATTERIES = [
    "friendly_battery",
    "real_user_battery",
    "combined_battery",
    "mcp_coverage_battery",
]

SYSTEMS = ("raw_rag", "agent_full")


def _question_set_path(battery: str) -> str:
    return f"eval/questions/{battery}.json"


def _short_battery(battery: str) -> str:
    return battery[: -len("_battery")] if battery.endswith("_battery") else battery


def _find_recent_run_id(
    db: EvalDB,
    *,
    system: str,
    question_set: str,
    branch: str | None = None,
) -> str | None:
    """Find the most recent raw_rag/agent_full run for a given battery.

    Used by --skip-runs to pick up where an earlier grand prix left off.
    Filters by branch when provided, so runs from other branches don't bleed in.
    """
    from sqlalchemy import desc

    from .models import EvalRun

    with db._session_factory() as session:  # noqa: SLF001  # intentional access to grand-prix internals
        q = session.query(EvalRun).filter(
            EvalRun.question_set == question_set,
        )
        if branch is not None:
            q = q.filter(EvalRun.agent_branch == branch)
        rows = q.order_by(desc(EvalRun.created_at)).limit(50).all()
        for row in rows:
            md: Any = row.metadata_ or {}
            if isinstance(md, dict) and md.get("system") == system:
                return str(row.id)
    return None


async def run_grand_prix(
    batteries: list[str] | None = None,
    output_html: str | None = None,
    comparisons_dir: str = "comparisons",
    judge_model: str | None = None,
    skip_runs: bool = False,
    branch_filter: str | None = None,
) -> dict[str, Any]:
    """Run a full grand prix (8 eval runs + 4 compare-judges + 1 HTML).

    Args:
        batteries: list of battery keys (e.g. ["friendly_battery", ...]).
            Defaults to DEFAULT_BATTERIES (all four).
        output_html: where to write the final HTML. Defaults to
            ~/.agent/diagrams/grand-prix-<timestamp>.html.
        comparisons_dir: directory for per-battery compare-judge JSON
            artifacts. Defaults to "comparisons/".
        judge_model: override judge model for eval + compare-judge.
        skip_runs: if True, skip phase 1 and pick up the newest existing
            runs per (system, battery) for compare-judge + HTML.
        branch_filter: when skip_runs=True, only consider runs from this
            agent branch. Defaults to the current git branch.
    """
    batteries = batteries or DEFAULT_BATTERIES
    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")

    comparisons_path = Path(comparisons_dir)
    comparisons_path.mkdir(parents=True, exist_ok=True)

    if output_html is None:
        default_dir = Path.home() / ".agent" / "diagrams"
        default_dir.mkdir(parents=True, exist_ok=True)
        output_html = str(default_dir / f"grand-prix-{ts}.html")

    run_ids: dict[tuple[str, str], str] = {}  # (system, battery) -> run_id

    # Phase 1: eval runs (unless skipped)
    if skip_runs:
        if branch_filter is None:
            from .runner import get_git_info

            branch_filter = get_git_info().get("branch")
        logger.info(
            f"--skip-runs: picking up newest runs per (system, battery) on branch={branch_filter}"
        )
        db = EvalDB(settings.DATABASE_URL)
        for battery in batteries:
            qs_path = _question_set_path(battery)
            for system in SYSTEMS:
                rid = _find_recent_run_id(
                    db, system=system, question_set=qs_path, branch=branch_filter
                )
                if rid is None:
                    raise RuntimeError(
                        f"--skip-runs set but no {system} run found for {battery} "
                        f"on branch {branch_filter}"
                    )
                run_ids[(system, battery)] = rid
                logger.info(f"  {system}/{battery}: {rid}")
    else:
        total = len(batteries) * len(SYSTEMS)
        i = 0
        for battery in batteries:
            for system in SYSTEMS:
                i += 1
                logger.info(f"=== Phase 1 [{i}/{total}]: {system} / {battery} ===")
                summary = await run_eval(
                    question_set_path=_question_set_path(battery),
                    system=system,  # type: ignore[arg-type]
                    judge_model=judge_model,
                )
                rid = summary.get("run_id")
                if not rid:
                    raise RuntimeError(f"{system}/{battery}: run_eval returned no run_id")
                run_ids[(system, battery)] = rid
                logger.info(
                    f"  done {system}/{battery} run_id={rid} "
                    f"composite={summary.get('composite_score')}"
                )

    # Phase 2: compare-judge per battery
    json_paths: list[str] = []
    for i, battery in enumerate(batteries, 1):
        baseline = run_ids[("raw_rag", battery)]
        candidate = run_ids[("agent_full", battery)]
        out = str(comparisons_path / f"grand_prix_{ts}_{_short_battery(battery)}.json")
        logger.info(f"=== Phase 2 [{i}/{len(batteries)}]: compare-judge {battery} ===")
        logger.info(f"  baseline={baseline[:8]} candidate={candidate[:8]} -> {out}")
        artifact = await compare_runs(
            baseline_run_id=baseline,
            candidate_run_id=candidate,
            output_path=out,
            judge_model=judge_model,
        )
        summary = artifact.get("run_summary") or {}
        logger.info(
            f"  done {battery}: winner={summary.get('winner')} margin={summary.get('margin')}"
        )
        json_paths.append(out)

    # Phase 3: render HTML from all per-battery JSONs.
    # grand-prix is explicitly a raw_rag-vs-agent_full comparison, so always
    # use the matching narrative preset regardless of any future default change.
    logger.info(f"=== Phase 3: render HTML to {output_html} ===")
    build_report_from_json(
        json_paths=[Path(p) for p in json_paths],
        output_path=Path(output_html),
        preset="grand-prix",
    )

    return {
        "timestamp": ts,
        "batteries": batteries,
        "run_ids": {f"{s}/{b}": rid for (s, b), rid in run_ids.items()},
        "comparison_jsons": json_paths,
        "html": output_html,
        "skip_runs": skip_runs,
    }
