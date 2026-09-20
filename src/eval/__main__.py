"""CLI entry point for the eval pipeline.

Usage:
    python -m src.eval run [--questions path.json]
    python -m src.eval compare --run-a ID --run-b ID
"""

import argparse
import asyncio
import logging
import sys
from typing import Any

from ..telemetry import init_telemetry, shutdown_telemetry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _handle_run(args: argparse.Namespace) -> None:
    from .report import print_run_summary
    from .scorer import run_eval

    summary = asyncio.run(
        run_eval(
            question_set_path=args.questions,
            system=args.system,
            judge_model=args.judge_model,
            allow_factless=args.allow_factless,
        )
    )
    print_run_summary(summary)


def _handle_multiturn(args: argparse.Namespace) -> None:
    from . import multiturn

    results, summary = asyncio.run(
        multiturn.run_battery(
            battery_path=args.threads,
            acting_user=args.acting_user,
            resource_context=args.resource,
            score=args.score,
            judge_model=args.judge_model,
            allow_draft_facts=args.allow_draft_facts,
        )
    )
    multiturn.print_summary(results, summary)


def _handle_coverage(args: argparse.Namespace) -> None:
    from src.config import settings

    from .coverage import build_coverage, reconcile_warnings
    from .coverage_report import render_coverage
    from .db import EvalDB

    db = EvalDB(settings.DATABASE_URL)

    # Resolve the run: explicit --run-id, else newest agent_full run.
    # The newest-run query uses execute_readonly_sql (Postgres metadata->> JSON
    # operators + SET TRANSACTION READ ONLY), which does not run on the SQLite
    # test DB; the explicit --run-id path is exercised in tests instead.
    run_id = args.run_id
    if not run_id:  # pragma: no cover - Postgres-only newest-run resolution
        clause = "AND question_set = %(qs)s" if args.battery else ""
        rows, _ = db.execute_readonly_sql(
            "SELECT id FROM eval_runs WHERE metadata->>'system'='agent_full' "
            f"AND tool_catalog IS NOT NULL {clause} ORDER BY created_at DESC LIMIT 1",
        )
        if not rows:
            print("No agent_full run with a tool_catalog found.")
            return
        run_id = rows[0][0]

    run = db.get_run(run_id)
    if run is None:
        print(f"Run {run_id} not found.")
        return
    meta: dict[str, Any] = run.metadata_ or {}  # type: ignore[assignment]
    system = meta.get("system")
    if system != "agent_full":
        print(
            f"Run {run_id} is system={system!r}, not 'agent_full'. "
            "Coverage needs the tool-calling loop's node_trace/tool_results; a "
            "raw_rag run has none — this would be no data, not thin coverage."
        )
        return

    catalog: dict[str, Any] = run.tool_catalog or {}  # type: ignore[assignment]
    served: list[str] = catalog.get("tools", []) or []
    scores: list[dict[str, Any]] = [
        {"question_id": s.question_id, "context": s.context or {}}
        for s in db.get_scores_for_run(run_id)
        if s.source == "judge"
    ]

    coverage = build_coverage(served, scores)
    warnings = reconcile_warnings(scores)
    print(
        render_coverage(
            coverage,
            run_id=run_id,
            battery=str(run.question_set) if run.question_set else None,
            model=str(run.llm_model) if run.llm_model else None,
            warnings=warnings,
            fmt=args.format,
        )
    )


def _handle_compare(args: argparse.Namespace) -> None:
    from src.config import settings

    from .db import EvalDB
    from .report import print_comparison

    db = EvalDB(settings.DATABASE_URL)
    run_a = db.get_run(args.run_a)
    run_b = db.get_run(args.run_b)
    if not run_a or not run_b:
        print("Error: One or both run IDs not found")
        sys.exit(1)
    summary_a: dict[str, Any] = {
        "run_id": run_a.id,
        "agent_branch": run_a.agent_branch,
        "composite_score": run_a.composite_score or 0.0,
        "per_dimension": _per_dimension(run_a.scores_summary),
    }
    summary_b: dict[str, Any] = {
        "run_id": run_b.id,
        "agent_branch": run_b.agent_branch,
        "composite_score": run_b.composite_score or 0.0,
        "per_dimension": _per_dimension(run_b.scores_summary),
    }
    # Scoring semantics are per-run: multiturn is a macro mean over Fair-only
    # thread composites, single-turn a micro average over questions — and the
    # per-dimension means differ in population the same way. Comparing across the
    # two is not like-for-like, so print_comparison annotates BOTH sections.
    commensurable = _is_macro_composite(run_a.scores_summary) == _is_macro_composite(
        run_b.scores_summary
    )
    print_comparison(summary_a, summary_b, composite_commensurable=commensurable)


def _is_macro_composite(scores_summary: Any) -> bool:
    """Whether a run's composite is a multiturn macro (Fair-only thread) mean.

    Feature-detected from the stored summary shape rather than run metadata: the
    nested per_dimension / thread_composites contract IS the multiturn summary,
    and detecting it keeps rejudged multiturn runs annotated too.
    """
    if not isinstance(scores_summary, dict):
        return False
    return isinstance(scores_summary.get("per_dimension"), dict) or isinstance(
        scores_summary.get("thread_composites"), dict
    )


def _per_dimension(scores_summary: Any) -> dict[str, Any]:
    """Read the per-dimension means out of a run's scores_summary.

    Single-turn runs store the dimension means as scores_summary itself;
    multiturn runs store a richer contract with the means under
    ``per_dimension`` alongside thread composites and turn counts. Feature-detect
    by key so both shapes render real values.
    """
    if not isinstance(scores_summary, dict):
        return {}
    nested = scores_summary.get("per_dimension")
    if isinstance(nested, dict):
        return nested
    return scores_summary


def _handle_report(args: argparse.Namespace) -> None:
    from src.config import settings

    from .db import EvalDB
    from .report import (
        generate_leadership_report,
        generate_resource_report,
        generate_team_report,
    )
    from .report_data import build_report_data

    db = EvalDB(settings.DATABASE_URL)
    data = build_report_data(db, args.since, resource=args.resource)
    if args.format == "team":
        print(generate_team_report(data))
    elif args.format == "leadership":
        print(generate_leadership_report(data))
    elif args.format == "resource":
        if not args.resource:
            print("Error: --resource required for resource format")
            sys.exit(1)
        print(generate_resource_report(data))


def _handle_ask(args: argparse.Namespace) -> None:
    from src.config import settings

    from .ask import ask as eval_ask

    result = eval_ask(
        question=args.question,
        database_url=settings.DATABASE_URL,
        api_key=settings.EVAL_JUDGE_API_KEY or settings.OPENAI_API_KEY,
        model=settings.EVAL_JUDGE_MODEL,
        base_url=settings.EVAL_JUDGE_BASE_URL or None,
    )
    print(result)


def _handle_comparison(args: argparse.Namespace) -> None:
    from pathlib import Path

    from src.config import settings

    from .db import EvalDB
    from .report import generate_comparison_report

    db = EvalDB(settings.DATABASE_URL)

    # Parse run ID pairs: "baseline_id:candidate_id,baseline_id:candidate_id,..."
    run_pairs = []
    for pair_str in args.pairs:
        parts = pair_str.split(":")
        if len(parts) != 2:
            print(f"Error: invalid pair '{pair_str}', expected 'baseline_id:candidate_id'")
            sys.exit(1)
        b_run = db.get_run(parts[0])
        c_run = db.get_run(parts[1])
        if not b_run or not c_run:
            print(f"Error: run not found in pair '{pair_str}'")
            sys.exit(1)

        b_meta: dict[str, Any] = b_run.metadata_ or {}  # type: ignore[assignment]
        c_meta: dict[str, Any] = c_run.metadata_ or {}  # type: ignore[assignment]

        b_scores = db.get_scores_for_run(parts[0])
        c_scores = db.get_scores_for_run(parts[1])

        def score_to_dict(s: Any) -> dict[str, Any]:
            return {
                "question_id": s.question_id,
                "question_text": s.question_text,
                "answer_text": s.answer_text,
                "composite_score": s.composite_score,
                "correctness": s.correctness,
                "specificity": s.specificity,
                "relevance": s.relevance,
                "citation_quality": s.citation_quality,
                "hedging": s.hedging,
                "duration_ms": s.duration_ms,
                "context": s.context,
                "source": s.source,
            }

        battery = (
            (c_run.question_set or b_run.question_set or "unknown")
            .split("/")[-1]
            .replace(".json", "")
        )
        run_pairs.append(
            {
                "battery": battery,
                "baseline": {
                    "run_id": parts[0],
                    "system": b_meta.get("system", "baseline"),
                    "composite": b_run.composite_score or 0,
                    "scores_summary": b_run.scores_summary or {},
                    "scores": [score_to_dict(s) for s in b_scores],
                },
                "candidate": {
                    "run_id": parts[1],
                    "system": c_meta.get("system", "candidate"),
                    "composite": c_run.composite_score or 0,
                    "scores_summary": c_run.scores_summary or {},
                    "scores": [score_to_dict(s) for s in c_scores],
                },
            }
        )

    report = generate_comparison_report(run_pairs, title=args.title)

    if args.output:
        with Path(args.output).open("w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


def _handle_rejudge(args: argparse.Namespace) -> None:
    import json as _json

    from .rejudge import rejudge_run

    summary = asyncio.run(
        rejudge_run(
            original_run_id=args.run_id,
            judge_model=args.judge_model,
        )
    )
    print(_json.dumps(summary, indent=2, default=str))


def _handle_grand_prix(args: argparse.Namespace) -> None:
    import json as _json

    from .grand_prix import DEFAULT_BATTERIES, run_grand_prix

    batteries = None
    if args.batteries:
        batteries = [b.strip() for b in args.batteries.split(",") if b.strip()]

    summary = asyncio.run(
        run_grand_prix(
            batteries=batteries,
            output_html=args.output,
            comparisons_dir=args.comparisons_dir,
            judge_model=args.judge_model,
            skip_runs=args.skip_runs,
        )
    )
    print(_json.dumps(summary, indent=2, default=str))
    print()
    print(f"Default batteries: {', '.join(DEFAULT_BATTERIES)}")


def _handle_compare_judge(args: argparse.Namespace) -> None:
    import json as _json

    from .compare_judge import compare_runs

    artifact = asyncio.run(
        compare_runs(
            baseline_run_id=args.baseline,
            candidate_run_id=args.candidate,
            output_path=args.output,
            judge_model=args.judge_model,
        )
    )
    preview = {
        "baseline_run_id": artifact["baseline_run_id"],
        "candidate_run_id": artifact["candidate_run_id"],
        "baseline_system": artifact["baseline_system"],
        "candidate_system": artifact["candidate_system"],
        "questions_compared": artifact["questions_compared"],
        "baseline_composite": artifact["baseline_composite"],
        "candidate_composite": artifact["candidate_composite"],
        "run_summary": artifact.get("run_summary"),
        "output_path": args.output,
    }
    print(_json.dumps(preview, indent=2, default=str))


def _handle_html(args: argparse.Namespace) -> None:
    from datetime import date as _date
    from pathlib import Path

    from src.config import settings

    from .html_report.builder import build_report, build_report_from_json

    output_path = Path(args.output)

    if args.from_json:
        json_paths = [Path(p) for p in args.from_json]
        missing = [p for p in json_paths if not p.exists()]
        if missing:
            print(f"Error: missing JSON file(s): {', '.join(str(p) for p in missing)}")
            sys.exit(1)
        bundle = build_report_from_json(
            json_paths=json_paths,
            output_path=output_path,
            preset=args.preset,
            title=args.title,
            subtitle=args.subtitle,
            label_a=args.label_a,
            label_b=args.label_b,
        )
        print(
            f"Wrote {output_path} ({len(bundle['all_pairs'])} pairs "
            f"from {len(json_paths)} compare-judge JSON(s))"
        )
        return

    on_date: _date | None = None
    if args.date:
        on_date = _date.fromisoformat(args.date)

    question_sets: list[str] | None = None
    if args.question_sets:
        question_sets = [q.strip() for q in args.question_sets.split(",") if q.strip()]

    bundle = build_report(
        database_url=settings.DATABASE_URL,
        output_path=output_path,
        on_date=on_date,
        question_sets=question_sets,
        preset=args.preset,
        title=args.title,
        subtitle=args.subtitle,
        label_a=args.label_a,
        label_b=args.label_b,
    )
    print(f"Wrote {output_path} ({len(bundle['all_pairs'])} question pairs)")


def _handle_score_production(_args: argparse.Namespace) -> None:
    print("Production scoring is not yet implemented.")
    print()
    print("Requires one of:")
    print("  - On-premise LLM at UKY (set EVAL_JUDGE_BASE_URL)")
    print("  - Updated privacy policy for external LLM processing of user queries")
    print("  - PII redaction pipeline")
    print()
    print("See: docs/superpowers/specs/2026-03-31-eval-pipeline-design.md")
    sys.exit(1)


def build_parser() -> argparse.ArgumentParser:  # noqa: PLR0915  # all subcommand wiring
    """Construct the eval CLI parser (kept separate from main() so it is testable)."""
    parser = argparse.ArgumentParser(
        prog="python -m src.eval",
        description="Agent answer evaluation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run pre-production eval")
    run_parser.add_argument(
        "--allow-factless",
        action="store_true",
        dest="allow_factless",
        help=(
            "Score questions that resolve to no required facts. Without facts the "
            "judge grades on plausibility alone and scores HIGHER than a graded "
            "question, so this is for smoke runs only."
        ),
    )
    run_parser.add_argument(
        "--system",
        choices=["agent_full", "raw_rag"],
        default="agent_full",
        help=(
            "System to evaluate: "
            "agent_full (default, tool-calling loop — the production path), "
            "raw_rag (UKY /ask, no agent)."
        ),
    )
    run_parser.add_argument(
        "--questions",
        default="eval/questions/friendly_battery.json",
        help="Path to question set JSON (default: friendly_battery.json)",
    )
    run_parser.add_argument(
        "--judge-model",
        default=None,
        help="Override judge model (default: from config)",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare two eval runs")
    compare_parser.add_argument("--run-a", required=True, help="First run ID")
    compare_parser.add_argument("--run-b", required=True, help="Second run ID")

    coverage_parser = subparsers.add_parser(
        "coverage", help="Per-tool evaluation-depth audit for a run"
    )
    coverage_parser.add_argument("--run-id", help="Run to audit (default: newest agent_full run)")
    coverage_parser.add_argument("--battery", help="Restrict the default run to this question_set")
    coverage_parser.add_argument(
        "--format", choices=["table", "md", "csv", "json"], default="table"
    )

    report_parser = subparsers.add_parser("report", help="Generate eval report")
    report_parser.add_argument(
        "--format", choices=["team", "leadership", "resource"], required=True
    )
    report_parser.add_argument("--since", default="7d", help="Time period (e.g., 7d, 30d)")
    report_parser.add_argument("--resource", help="Resource name (for resource format)")

    ask_parser = subparsers.add_parser("ask", help="Ask a question about eval data")
    ask_parser.add_argument("question", help="Natural language question")

    comparison_parser = subparsers.add_parser(
        "comparison", help="Generate A/B comparison report from paired runs"
    )
    comparison_parser.add_argument(
        "pairs",
        nargs="+",
        help="Run ID pairs as baseline_id:candidate_id (one per battery)",
    )
    comparison_parser.add_argument(
        "--title",
        default="Production Baseline Comparison",
        help="Report title",
    )
    comparison_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file path (default: stdout)",
    )

    rejudge_parser = subparsers.add_parser(
        "rejudge",
        help="Re-judge an existing run's answers with the current rubric (no system re-call)",
    )
    rejudge_parser.add_argument("--run-id", required=True, help="Run ID to re-judge")
    rejudge_parser.add_argument("--judge-model", default=None, help="Override judge model")

    gp_parser = subparsers.add_parser(
        "grand-prix",
        help=(
            "Run the full Production Baseline Comparison: 8 eval runs "
            "+ 4 compare-judges + 1 HTML report"
        ),
    )
    gp_parser.add_argument(
        "--batteries",
        default=None,
        help=(
            "Comma-separated list of battery keys (e.g. "
            "'friendly_battery,mcp_coverage_battery'). Default: all four."
        ),
    )
    gp_parser.add_argument(
        "--output",
        "-o",
        default=None,
        help=("Output HTML path. Default: ~/.agent/diagrams/grand-prix-<timestamp>.html"),
    )
    gp_parser.add_argument(
        "--comparisons-dir",
        default="comparisons",
        help="Directory for per-battery compare-judge JSON artifacts",
    )
    gp_parser.add_argument(
        "--judge-model",
        default=None,
        help="Override judge model (default: from config)",
    )
    gp_parser.add_argument(
        "--skip-runs",
        action="store_true",
        help=(
            "Skip phase 1 (eval runs); use newest existing runs per "
            "(system, battery) on the current branch. For iterating on the "
            "compare-judge + HTML phases without burning fresh runs."
        ),
    )

    compare_judge_parser = subparsers.add_parser(
        "compare-judge",
        help="Compare two runs head-to-head with an LLM; writes JSON artifact (no DB writes)",
    )
    compare_judge_parser.add_argument(
        "--baseline", required=True, help="Baseline run ID (e.g. raw_rag)"
    )
    compare_judge_parser.add_argument(
        "--candidate", required=True, help="Candidate run ID (e.g. agent_full)"
    )
    compare_judge_parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    compare_judge_parser.add_argument(
        "--judge-model", default=None, help="Override judge model (default: from config)"
    )

    html_parser = subparsers.add_parser(
        "html", help="Generate the HTML comparison report (raw_rag vs agent_full)"
    )
    html_parser.add_argument(
        "-o",
        "--output",
        default="comparison-report.html",
        help="Output path (default: ./comparison-report.html)",
    )
    html_parser.add_argument(
        "--date",
        default=None,
        help="YYYY-MM-DD — restrict to runs created on this date (UTC). Default: newest available.",
    )
    html_parser.add_argument(
        "--question-sets",
        default=None,
        help=(
            "Comma-separated list of battery keys (e.g. "
            "'friendly_battery,combined_battery'). Default: all four."
        ),
    )
    html_parser.add_argument(
        "--from-json",
        nargs="+",
        default=None,
        help=(
            "Render from one or more compare-judge JSON artifacts instead of "
            "querying Postgres. Each JSON is one battery-pair; multiple JSONs "
            "are merged into a multi-battery report. Overrides --date and "
            "--question-sets."
        ),
    )
    html_parser.add_argument(
        "--preset",
        choices=["grand-prix"],
        default="grand-prix",
        help=(
            "Narrative prose preset: 'grand-prix' (raw_rag vs agent_full "
            "production-baseline comparison)."
        ),
    )
    html_parser.add_argument(
        "--title",
        default=None,
        help=(
            "Override the report's main heading (the <h1>). Default: "
            "'Production Baseline Comparison'. Use for ad-hoc comparisons "
            "that don't match an existing preset."
        ),
    )
    html_parser.add_argument(
        "--subtitle",
        default=None,
        help=(
            "Override the subtitle under the main heading. Default: taken "
            "from the active preset (e.g., 'Raw UKY RAG vs full ACCESS "
            "agent' for grand-prix)."
        ),
    )
    html_parser.add_argument(
        "--label-a",
        default=None,
        help=(
            "Override the column label for the baseline (A) system. "
            "Default: derived from the system ID (e.g., 'Raw RAG', "
            "'Agent'). Useful when both runs share a system ID (e.g., "
            "two agent_full runs in an agent-vs-agent comparison)."
        ),
    )
    html_parser.add_argument(
        "--label-b",
        default=None,
        help=(
            "Override the column label for the candidate (B) system. "
            "Default: derived from the system ID."
        ),
    )

    multiturn_parser = subparsers.add_parser(
        "multiturn",
        help=(
            "Run a multi-turn thread battery (exercises context-management / "
            "SummarizationMiddleware; not exercised by single-turn eval)"
        ),
    )
    multiturn_parser.add_argument(
        "--threads",
        required=True,
        help="Path to multi-turn battery JSON",
    )
    multiturn_parser.add_argument(
        "--acting-user",
        default=None,
        help="Optional ACCESS ID for authenticated calls during the thread",
    )
    multiturn_parser.add_argument(
        "--resource",
        default=None,
        help="Optional RP slug applied as resource_context for the thread",
    )
    multiturn_parser.add_argument(
        "--score",
        action="store_true",
        help="Judge each turn and persist to eval_runs/eval_scores (requires DATABASE_URL)",
    )
    multiturn_parser.add_argument(
        "--judge-model",
        default=None,
        dest="judge_model",
        help="Override judge model (default: from config)",
    )
    multiturn_parser.add_argument(
        "--allow-draft-facts",
        action="store_true",
        dest="allow_draft_facts",
        help=(
            "Score against facts that still contain 'AUTHOR:' placeholders "
            "(smoke tests only — placeholders make correctness verdicts arbitrary)"
        ),
    )

    # Production scoring is deferred — requires on-premise LLM or updated privacy policy
    # to send real user queries to a judge. See spec: docs/superpowers/specs/2026-03-31-eval-pipeline-design.md
    subparsers.add_parser(
        "score-production",
        help="[NOT YET IMPLEMENTED] Score recent production answers (requires on-premise LLM)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    handlers = {
        "run": _handle_run,
        "compare": _handle_compare,
        "comparison": _handle_comparison,
        "rejudge": _handle_rejudge,
        "compare-judge": _handle_compare_judge,
        "grand-prix": _handle_grand_prix,
        "report": _handle_report,
        "ask": _handle_ask,
        "html": _handle_html,
        "score-production": _handle_score_production,
        "multiturn": _handle_multiturn,
        "coverage": _handle_coverage,
    }

    handler = handlers.get(args.command)
    if not handler:
        parser.print_help()
        sys.exit(1)

    # Separate process from the API server, so src/main.py's lifespan hooks
    # never run and eval spans would go to a no-op provider. Tagged
    # service.component=access-agent-eval (service.name is the Honeycomb
    # dataset) to tell an eval run from live traffic. The flush must be in a
    # finally: a crashing run is the one worth tracing, and a CLI exits before
    # the batch processor would send anything.
    init_telemetry(service_name="access-agent-eval")
    try:
        handler(args)
    finally:
        shutdown_telemetry()


if __name__ == "__main__":
    main()
