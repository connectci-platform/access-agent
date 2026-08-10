"""Red-team gate CLI. Phase 1: nightly-only, flags via issue, never fails the job."""

from __future__ import annotations

import argparse
import asyncio
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import httpx

from src.config import settings  # module-level singleton, not a factory (src/config.py:264)
from src.eval.judge import Judge

from .gate import GateResult, run_gate
from .replayer import redteam_headers
from .report import flag_line, issue_body, write_artifact
from .suite import assert_versions_match, join_suite, load_baseline, load_prompts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

BASELINE = Path(__file__).parent.parent.parent / "tests" / "redteam" / "suite-v1" / "baseline.json"


def read_only_enabled() -> bool:
    return os.getenv("READ_ONLY", "").lower() in ("1", "true")


def _build_judge() -> Judge:
    return Judge(
        base_url=settings.EVAL_JUDGE_BASE_URL or None,
        api_key=settings.EVAL_JUDGE_API_KEY or None,
        model=settings.EVAL_JUDGE_MODEL,
        thinking=settings.EVAL_JUDGE_THINKING,
    )


async def run_from_env(
    _gate: Callable[..., Awaitable[GateResult]] = run_gate,
) -> GateResult:
    # Hard preconditions checked BEFORE any network/replay activity.
    if not read_only_enabled():
        raise RuntimeError("READ_ONLY must be true for the red-team gate")
    prompts_path = Path(os.environ["REDTEAM_PROMPTS_PATH"])
    base_url = os.environ.get("REDTEAM_BASE_URL", "http://localhost:8000")
    n = int(os.environ.get("REDTEAM_N", "5"))
    concurrency = int(os.environ.get("REDTEAM_CONCURRENCY", "6"))
    baseline = load_baseline(BASELINE)
    prompts = load_prompts(prompts_path)
    assert_versions_match(baseline, prompts)  # abort on drift (raises SuiteVersionMismatch)
    items = join_suite(baseline, prompts)
    run_id = f"redteam-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    headers = redteam_headers(baseline.suite_version, run_id)
    judge = _build_judge()
    async with httpx.AsyncClient() as client:
        result = await _gate(
            items,
            base_url=base_url,
            n=n,
            concurrency=concurrency,
            judge=judge,
            headers=headers,
            http_client=client,
        )
    # Write the full on-prem artifact BEFORE returning, so a later print/crash
    # can never lose evidence (ordering is load-bearing — see test).
    artifact = Path(os.environ.get("REDTEAM_ARTIFACT_DIR", "/tmp/redteam")) / f"{run_id}.json"
    write_artifact(artifact, result.artifact_records)  # on-prem only, gitignored dir
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-issue-body", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(run_from_env())
    for f in result.flags:
        print(flag_line(f))  # uses the Flag's precomputed hash — never recomputes from ""
    if args.emit_issue_body and result.flags:
        Path(os.environ.get("REDTEAM_ISSUE_BODY", "issue_body.md")).write_text(
            issue_body(result.flags)
        )


if __name__ == "__main__":
    main()
