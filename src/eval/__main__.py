"""CLI entry point for the eval pipeline.

Usage:
    python -m src.eval run [--questions path.json]
    python -m src.eval compare --run-a ID --run-b ID
"""

import argparse
import asyncio
import logging
import sys
from typing import Any, cast

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
            push_argilla=args.push_argilla,
        )
    )
    print_run_summary(summary)


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
        "per_dimension": run_a.scores_summary or {},
    }
    summary_b: dict[str, Any] = {
        "run_id": run_b.id,
        "agent_branch": run_b.agent_branch,
        "composite_score": run_b.composite_score or 0.0,
        "per_dimension": run_b.scores_summary or {},
    }
    print_comparison(summary_a, summary_b)


def _handle_argilla_sync(args: argparse.Namespace) -> None:
    from src.config import settings

    from .argilla_pull import sync_from_argilla
    from .db import EvalDB

    db = EvalDB(settings.DATABASE_URL)
    stats = sync_from_argilla(
        db=db,
        argilla_url=settings.ARGILLA_URL,
        argilla_api_key=settings.ARGILLA_API_KEY,
        dataset_name=args.dataset,
        run_id=args.run_id,
    )
    print(f"Synced: {stats['synced']}, Skipped: {stats['skipped']}, Errors: {stats['errors']}")


def _handle_cleanup(args: argparse.Namespace) -> None:
    from src.config import settings

    from .argilla_push import dataset_name_for_branch, delete_dataset

    ds_name = dataset_name_for_branch(args.branch)
    if delete_dataset(settings.ARGILLA_URL, settings.ARGILLA_API_KEY, ds_name):
        print(f"Deleted dataset '{ds_name}'")
    else:
        print(f"Failed to delete dataset '{ds_name}'")
        sys.exit(1)


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
                "completeness": s.completeness,
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


def _handle_argilla_push(args: argparse.Namespace) -> None:
    from src.config import settings

    from .argilla_push import (
        build_argilla_record,
        dataset_name_for_branch,
        push_scores_to_argilla,
    )
    from .db import EvalDB

    argilla_url = args.argilla_url or settings.ARGILLA_URL
    argilla_key = args.argilla_key or settings.ARGILLA_API_KEY

    if not argilla_url or not argilla_key:
        print("Error: ARGILLA_URL and ARGILLA_API_KEY required (via args or env)")
        sys.exit(1)

    db = EvalDB(settings.DATABASE_URL)
    run = db.get_run(args.run_id)
    if not run:
        print(f"Error: Run {args.run_id} not found")
        sys.exit(1)

    if run.run_type == "rejudge" and not args.force:
        print(f"Error: Run {args.run_id} is a rejudge run.")
        print(
            "Argilla records are keyed by question_id and would OVERWRITE the "
            "originals on push, silently replacing the prior judge's suggestions."
        )
        print("If you really want to push a rejudge run, re-run with --force.")
        sys.exit(1)

    meta: dict[str, Any] = run.metadata_ or {}  # type: ignore[assignment]
    ds_name = args.dataset or dataset_name_for_branch(cast("str | None", run.agent_branch))

    scores = db.get_scores_for_run(args.run_id)
    records = []
    for score in scores:
        if score.source not in ("judge", "judge_error", "skipped"):
            continue
        score_context: dict[str, Any] = score.context or {}  # type: ignore[assignment]
        records.append(
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
                run_id=args.run_id,
                agent_branch=cast("str | None", run.agent_branch),
                agent_commit=cast("str | None", run.agent_commit),
                judge_model=cast("str | None", run.judge_model),
                duration_ms=float(score.duration_ms) if score.duration_ms else None,
            )
        )

    if not records:
        print("No scores to push")
        sys.exit(0)

    pushed = push_scores_to_argilla(records, argilla_url, argilla_key, ds_name)
    print(f"Pushed {pushed} records to Argilla dataset '{ds_name}'")
    print(f"  System: {meta.get('system', '?')}")
    print(f"  Battery: {run.question_set}")
    print(f"  Composite: {run.composite_score:.2f}")


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


def main() -> None:  # noqa: PLR0915  # CLI dispatcher, statements not meaningfully extractable
    parser = argparse.ArgumentParser(
        prog="python -m src.eval",
        description="Agent answer evaluation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run pre-production eval")
    run_parser.add_argument(
        "--system",
        choices=["agent_full", "agent_full_legacy", "agent_rag_only", "raw_rag"],
        default="agent_full",
        help=(
            "System to evaluate: "
            "agent_full (default, tool-calling loop — the new Phase-3 path), "
            "agent_full_legacy (old plan→execute→evaluate→recover→synthesize chain, "
            "for parity comparison), "
            "agent_rag_only (skip the agent, serve RAG matches directly), "
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
    run_parser.add_argument(
        "--push-argilla",
        action="store_true",
        help="Push scored answers to Argilla for human review",
    )

    compare_parser = subparsers.add_parser("compare", help="Compare two eval runs")
    compare_parser.add_argument("--run-a", required=True, help="First run ID")
    compare_parser.add_argument("--run-b", required=True, help="Second run ID")

    sync_parser = subparsers.add_parser("argilla-sync", help="Sync human annotations from Argilla")
    sync_parser.add_argument("--dataset", required=True, help="Argilla dataset name")
    sync_parser.add_argument("--run-id", required=True, help="Eval run to associate scores with")

    cleanup_parser = subparsers.add_parser("cleanup", help="Delete Argilla branch dataset")
    cleanup_parser.add_argument("--branch", required=True, help="Branch name")

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

    push_parser = subparsers.add_parser(
        "argilla-push", help="Push a completed run to Argilla (all scores, no threshold)"
    )
    push_parser.add_argument("--run-id", required=True, help="Eval run ID to push")
    push_parser.add_argument(
        "--dataset", default=None, help="Argilla dataset name (default: eval-{branch})"
    )
    push_parser.add_argument(
        "--argilla-url", default=None, help="Argilla URL (default: from config)"
    )
    push_parser.add_argument(
        "--argilla-key", default=None, help="Argilla API key (default: from config)"
    )
    push_parser.add_argument(
        "--force",
        action="store_true",
        help="Allow pushing a rejudge run (overwrites original's Argilla suggestions)",
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
        choices=["grand-prix", "phase3-parity"],
        default="grand-prix",
        help=(
            "Narrative prose preset: 'grand-prix' (default — raw_rag vs agent_full "
            "production-baseline comparison) or 'phase3-parity' (loop vs legacy chain)."
        ),
    )

    # Production scoring is deferred — requires on-premise LLM or updated privacy policy
    # to send real user queries to a judge. See spec: docs/superpowers/specs/2026-03-31-eval-pipeline-design.md
    subparsers.add_parser(
        "score-production",
        help="[NOT YET IMPLEMENTED] Score recent production answers (requires on-premise LLM)",
    )

    args = parser.parse_args()

    handlers = {
        "run": _handle_run,
        "compare": _handle_compare,
        "comparison": _handle_comparison,
        "argilla-sync": _handle_argilla_sync,
        "argilla-push": _handle_argilla_push,
        "rejudge": _handle_rejudge,
        "compare-judge": _handle_compare_judge,
        "grand-prix": _handle_grand_prix,
        "cleanup": _handle_cleanup,
        "report": _handle_report,
        "ask": _handle_ask,
        "html": _handle_html,
        "score-production": _handle_score_production,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
