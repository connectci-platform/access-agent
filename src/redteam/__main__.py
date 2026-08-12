"""Red-team gate CLI. Phase 1: nightly-only, flags via issue, never fails the job."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from httpx import ASGITransport

from src.config import settings  # module-level singleton, not a factory (src/config.py:264)
from src.eval.judge import Judge

from .gate import GateResult, JudgeMismatch, RedteamOutage, run_gate
from .replayer import redteam_headers
from .report import flag_line, issue_body, outage_body, write_artifact
from .suite import (
    assert_suite_complete,
    assert_versions_match,
    join_suite,
    load_baseline,
    load_prompts,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

BASELINE = Path(__file__).parent.parent.parent / "tests" / "redteam" / "suite-v1" / "baseline.json"

# The running scorer's identity — matches cascade.py's scoring logic. Bumped
# whenever the cascade's scoring behavior changes in a way that could shift
# verdicts, so a stale baseline can't silently be judged by new logic.
SCORER_VERSION = "cascade-v1"


def _reject_in_repo(path: Path, var: str) -> Path:
    resolved = path.resolve()
    repo_root = Path(__file__).resolve().parents[2]
    if repo_root == resolved or repo_root in resolved.parents:
        raise RuntimeError(f"{var} must be outside the repo tree (got {resolved})")
    return resolved


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
    n = int(os.environ.get("REDTEAM_N", "5"))
    concurrency = int(os.environ.get("REDTEAM_CONCURRENCY", "6"))
    baseline = load_baseline(BASELINE)
    prompts = load_prompts(prompts_path)
    assert_versions_match(baseline, prompts)  # abort on drift (raises SuiteVersionMismatch)
    items = join_suite(baseline, prompts)
    assert_suite_complete(baseline, prompts)  # raises SuiteIncomplete on drift
    if baseline.judge_model != settings.EVAL_JUDGE_MODEL:
        raise JudgeMismatch(
            f"baseline judge_model {baseline.judge_model!r} != running "
            f"{settings.EVAL_JUDGE_MODEL!r}"
        )
    # scorer_version is a static pin the running scorer must match (cascade-v1).
    if baseline.scorer_version != SCORER_VERSION:
        raise JudgeMismatch(
            f"baseline scorer_version {baseline.scorer_version!r} != {SCORER_VERSION!r}"
        )
    run_id = f"redteam-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    headers = redteam_headers(baseline.suite_version, run_id)
    judge = _build_judge()
    base_url_env = os.environ.get("REDTEAM_BASE_URL")
    lifespan_cm: AbstractAsyncContextManager[Any]
    if base_url_env:
        base_url = base_url_env
        client_cm = httpx.AsyncClient()
        lifespan_cm = contextlib.nullcontext()
    else:
        base_url = "http://redteam-asgi"
        from src.main import app  # imported lazily so READ_ONLY precheck runs first

        client_cm = httpx.AsyncClient(transport=ASGITransport(app=app), base_url=base_url)
        lifespan_cm = app.router.lifespan_context(app)
    async with lifespan_cm, client_cm as client:
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
    artifact_dir = _reject_in_repo(
        Path(os.environ.get("REDTEAM_ARTIFACT_DIR", "/tmp/redteam")), "REDTEAM_ARTIFACT_DIR"
    )
    artifact = artifact_dir / f"{run_id}.json"
    write_artifact(artifact, result.artifact_records)  # on-prem only, gitignored dir
    return result


def _write_status(disposition: str, reason: str | None, flag_count: int) -> None:
    path = os.environ.get("REDTEAM_STATUS_PATH")
    if not path:
        return
    p = _reject_in_repo(Path(path), "REDTEAM_STATUS_PATH")
    p.write_text(
        json.dumps({"disposition": disposition, "reason": reason, "flag_count": flag_count})
    )


def main_argv(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--emit-issue-body", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = asyncio.run(run_from_env())
    except RedteamOutage as e:
        # An operational failure, NOT a redacted-regression finding: distinct message
        # and a distinct issue body so it never reads as "the agent regressed." One
        # handler covers every outage subclass (JudgeOutage/JudgeMismatch and, later,
        # AgentOutage/SurfaceOutage) — they're told apart by e.reason.
        _write_status("outage", e.reason, 0)
        print(f"REDTEAM OUTAGE ({e.reason}): {e}")
        if args.emit_issue_body:
            body = _reject_in_repo(
                Path(os.environ.get("REDTEAM_ISSUE_BODY", "issue_body.md")),
                "REDTEAM_ISSUE_BODY",
            )
            body.write_text(outage_body(e.reason, str(e)))  # redaction: counts/reason only
        raise SystemExit(2) from e
    regressions = [f for f in result.flags if f.kind == "candidate-regression"]
    for f in result.flags:
        print(flag_line(f))  # uses the Flag's precomputed hash — never recomputes from ""
    if regressions:
        _write_status("regression", None, len(regressions))
        if args.emit_issue_body:
            body = _reject_in_repo(
                Path(os.environ.get("REDTEAM_ISSUE_BODY", "issue_body.md")),
                "REDTEAM_ISSUE_BODY",
            )
            body.write_text(issue_body(regressions))
        raise SystemExit(3)
    _write_status("clean", None, 0)
    raise SystemExit(0)


def main() -> None:  # entrypoint keeps its name for `python -m src.redteam`
    main_argv()


if __name__ == "__main__":
    main()
