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

from .gate import GateResult, JudgeMismatch, RedteamOutage, SurfaceOutage, run_gate
from .replayer import redteam_headers
from .report import findings_file_lines, outage_body, write_artifact
from .suite import (
    SuiteIncomplete,
    SuiteKeyMismatch,
    SuiteVersionMismatch,
    assert_suite_complete,
    assert_versions_match,
    join_suite,
    load_baseline,
    load_prompts,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

BASELINE = Path(__file__).parent.parent.parent / "tests" / "redteam" / "suite-v1" / "baseline.json"

# The running scorer's identity — matches cascade.py's scoring logic. Bump this
# whenever src/redteam/cascade.py's scoring logic changes in a way that could
# shift verdicts, so a stale baseline can't silently be judged by new logic.
SCORER_VERSION = "cascade-v1"

# The 9 READ-capable MCP servers (production mcp_server_urls minus the two
# write-only servers announcements/jsm) — a preflight only cares that the
# READ surface an attack could actually exercise is up; write-only outages
# are irrelevant to whether an attack would trivially refuse.
REDTEAM_REQUIRED_READ_SERVERS = frozenset(
    {
        "allocations",
        "software-discovery",
        "system-status",
        "events",
        "affinity-groups",
        "xdmod",
        "xdmod-data",
        "nsf-awards",
        "compute-resources",
    }
)
REDTEAM_MIN_TOOLS = 10


async def _surface_preflight(client: Any, base_url: str) -> None:
    """GET /api/v1/health and refuse to replay against a thin READ tool surface.

    A thin/degraded surface would make attacks refuse trivially -> false
    green. Missing 'unavailable_servers' means all servers available
    (routes.py omits the key entirely when servers_available == servers_total).

    The health route is mounted under the /api/v1 prefix (main.py includes the
    router with prefix="/api/v1"); hitting a bare /health 404s and would be
    misread as "no catalog" -> a false surface outage.
    """
    resp = await client.get(f"{base_url}/api/v1/health", timeout=30.0)
    tools = (resp.json() or {}).get("tools")
    if not tools:
        raise SurfaceOutage("no tool catalog in /api/v1/health — agent surface not warm")
    total = tools.get("total", 0)
    if total < REDTEAM_MIN_TOOLS:
        raise SurfaceOutage(f"tool surface thin: {total} < {REDTEAM_MIN_TOOLS}")
    unavailable = set(tools.get("unavailable_servers", []))
    down = REDTEAM_REQUIRED_READ_SERVERS & unavailable
    if down:
        raise SurfaceOutage(f"required READ servers unavailable: {sorted(down)}")


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
        await _surface_preflight(client, base_url)
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
    except (RedteamOutage, SuiteVersionMismatch, SuiteKeyMismatch, SuiteIncomplete) as e:
        # An operational failure, NOT a redacted-regression finding: distinct message
        # and a distinct issue body so it never reads as "the agent regressed." One
        # handler covers every outage subclass (JudgeOutage/JudgeMismatch,
        # AgentOutage/SurfaceOutage) plus the suite-drift exceptions from suite.py
        # (Suite* aren't RedteamOutage subclasses — suite.py has no runtime import
        # of gate.py, so broadening this handler avoids an import cycle) — they're
        # told apart by e.reason. CRITICAL: a suite drift is routine (id-set churn)
        # and str(e) may embed detail; routing it through this same redacted path
        # (never str(e) to stdout) is what keeps prompt ids out of public Actions
        # logs — see FIX 1 in the 2026-08-26 whole-branch review.
        _write_status("outage", e.reason, 0)
        if args.emit_issue_body:
            body = _reject_in_repo(
                Path(os.environ.get("REDTEAM_ISSUE_BODY", "issue_body.md")),
                "REDTEAM_ISSUE_BODY",
            )
            body.write_text(outage_body(e.reason, str(e)))  # redaction: counts/reason only
        raise SystemExit(2) from e
    regressions = [f for f in result.flags if f.tier == "defended"]
    scope_findings = [f for f in result.flags if f.tier == "soft"]
    fixes = [
        f for f in result.flags if f.tier == "known-jailbreak"
    ]  # candidate-fix: a tracked hole closed
    all_findings = regressions + scope_findings + fixes
    if args.emit_issue_body and all_findings:
        body = _reject_in_repo(
            Path(os.environ.get("REDTEAM_ISSUE_BODY", "issue_body.md")), "REDTEAM_ISSUE_BODY"
        )
        body.write_text("\n".join(findings_file_lines(all_findings)) + "\n")
    if regressions:
        _write_status("regression", None, len(regressions))
        raise SystemExit(3)
    if scope_findings:
        _write_status("scope", None, len(scope_findings))
        raise SystemExit(0)  # scope finding does not fail the job
    if fixes:
        _write_status("fix", None, len(fixes))
        raise SystemExit(0)  # good news, does not fail the job
    _write_status("clean", None, 0)
    raise SystemExit(0)


def main() -> None:  # entrypoint keeps its name for `python -m src.redteam`
    main_argv()


if __name__ == "__main__":
    main()
