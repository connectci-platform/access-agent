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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.eval",
        description="Agent answer evaluation pipeline",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run pre-production eval")
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
        "argilla-sync": _handle_argilla_sync,
        "cleanup": _handle_cleanup,
        "report": _handle_report,
        "ask": _handle_ask,
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
