"""Run eval questions through the agent and collect results."""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from src.agent.domains.capabilities import get_capability_registry
from src.agent.graph import run_agent
from src.agent.state import AgentState
from src.agent.turn_capture import get_turn_capture, reset_turn_capture
from src.config import settings
from src.services.resource_matcher import resources_for_turn
from src.services.uky_client import get_uky_client
from src.turn_reporter import get_turn_reporter

logger = logging.getLogger(__name__)

SystemMode = Literal["agent_full", "raw_rag"]

# Mapping from SystemMode to short architectural names used in run IDs.
# These describe the pipeline architecture, not the user-facing system choice,
# so the IDs stay meaningful in Argilla's "Eval Run ID" filter and psql output.
SYSTEM_SHORTCODES: dict[SystemMode, str] = {
    "agent_full": "loop",
    "raw_rag": "raw_rag",
}


def gen_semantic_run_id(system: SystemMode) -> str:
    """Generate a semantic run ID of form `{shortcode}-{YYYYMMDD}-{HHMMSS}-{hash6}`.

    Lex-sortable by time. Fits in the existing String(36) column. Collision-safe
    against same-second same-system runs via a 6-hex random suffix.

    Returns a plain string; the caller passes it as an explicit `id=` to
    `db.create_run(...)`, overriding the UUID default. Remove this helper and
    its call site to revert to UUID IDs — no schema or data migration needed.
    """
    import secrets
    from datetime import UTC, datetime

    shortcode = SYSTEM_SHORTCODES[system]
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)
    return f"{shortcode}-{ts}-{suffix}"


@dataclass
class RunResult:
    question_id: str
    question_text: str
    answer: str
    rag_context: str | None = None
    tool_results: str | None = None
    node_trace: str | None = None
    tools_used: list[str] = field(default_factory=list)
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None


def get_git_info() -> dict[str, Any]:
    """Code provenance for eval_runs (agent_commit / agent_branch).

    Inside the container there is no .git, so fall back to the env stamps the
    image build provides (GIT_COMMIT / GIT_BRANCH build args), then
    AGENT_VERSION as a last resort for the commit.
    """
    try:
        return {
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "branch": subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
            ).strip(),
        }
    except Exception:
        return {
            "commit": os.environ.get("GIT_COMMIT") or settings.AGENT_VERSION or "unknown",
            "branch": os.environ.get("GIT_BRANCH") or "unknown",
        }


def _format_rag_matches(state: Any) -> str | None:
    matches = state.get("rag_matches", [])
    if not matches:
        return None
    parts = []
    for m in matches:
        if hasattr(m, "answer"):
            parts.append(f"Score: {getattr(m, 'score', 'N/A')}\n{m.answer}")
        elif isinstance(m, dict):
            parts.append(f"Score: {m.get('score', 'N/A')}\n{m.get('answer', '')}")
    return "\n---\n".join(parts) if parts else None


def _format_tool_results(state: Any) -> str | None:
    """Format tool results for the judge as a structured record per call.

    Each call shows: tool name, arguments, success, duration, an explicit
    result_count + empty flag, and the raw data. This replaces a plain data
    dump so the judge can reason about what the agent actually tried rather
    than parse string blobs.
    """
    results = state.get("tool_results", [])
    if not results:
        return None

    def _extract(r: Any) -> dict[str, Any]:
        if hasattr(r, "tool_name"):
            return {
                "tool_name": getattr(r, "tool_name", "unknown"),
                "arguments": getattr(r, "arguments", {}) or {},
                "success": getattr(r, "success", True),
                "duration_ms": getattr(r, "duration_ms", 0),
                "data": getattr(r, "data", None),
                "error": getattr(r, "error", None),
            }
        if isinstance(r, dict):
            return {
                "tool_name": r.get("tool_name", "unknown"),
                "arguments": r.get("arguments", {}) or {},
                "success": r.get("success", True),
                "duration_ms": r.get("duration_ms", 0),
                "data": r.get("data"),
                "error": r.get("error"),
            }
        return {
            "tool_name": "unknown",
            "arguments": {},
            "success": True,
            "duration_ms": 0,
            "data": r,
            "error": None,
        }

    def _summarize(data: Any) -> tuple[int | None, bool]:
        """Return (result_count, empty_flag); count is None when indeterminate."""
        if data is None:
            return 0, True
        if isinstance(data, list):
            return len(data), len(data) == 0
        if isinstance(data, dict):
            for key in ("items", "results", "events", "announcements", "outages", "records"):
                value = data.get(key)
                if isinstance(value, list):
                    return len(value), len(value) == 0
            for key in ("total", "total_items", "total_events", "total_outages", "count"):
                value = data.get(key)
                if isinstance(value, int):
                    return value, value == 0
            if not data:
                return 0, True
            return None, False
        if isinstance(data, str):
            return None, not data.strip()
        return None, False

    parts: list[str] = []
    for r in results:
        info = _extract(r)
        count, empty = _summarize(info["data"])
        args_json = json.dumps(info["arguments"], default=str, sort_keys=True)
        lines = [
            f"### Tool call: {info['tool_name']}",
            f"- arguments: {args_json}",
            f"- success: {info['success']}",
            f"- duration_ms: {info['duration_ms']}",
        ]
        if count is not None:
            lines.append(f"- result_count: {count}")
        lines.append(f"- empty: {empty}")
        if info["error"]:
            lines.append(f"- error: {info['error']}")
        lines.append("- data:")
        lines.append(str(info["data"]))
        parts.append("\n".join(lines))

    return "\n\n---\n\n".join(parts) if parts else None


def _format_node_trace(state: Any) -> str | None:
    trace = state.get("node_trace", [])
    if not trace:
        return None
    return json.dumps(trace, indent=2, default=str)


async def run_question(
    question_id: str,
    question_text: str,
    tool_catalog: Any,
    system: SystemMode = "agent_full",
    resource_context: str | None = None,
    battery_id: str | None = None,
    battery_run_id: str | None = None,
) -> RunResult:
    start = time.monotonic()
    try:
        if system == "raw_rag":
            result = await _run_raw_rag(
                question_id,
                question_text,
                resource_context,
                battery_id=battery_id,
                battery_run_id=battery_run_id,
            )
        else:  # agent_full
            result = await _run_agent(
                question_id,
                question_text,
                tool_catalog,
                resource_context=resource_context,
                battery_id=battery_id,
                battery_run_id=battery_run_id,
            )
        result.duration_ms = (time.monotonic() - start) * 1000
        return result
    except Exception as e:
        duration_ms = (time.monotonic() - start) * 1000
        logger.error(f"Question {question_id} failed: {e}")
        return RunResult(
            question_id=question_id,
            question_text=question_text,
            answer="",
            duration_ms=duration_ms,
            success=False,
            error=str(e),
        )


async def _run_raw_rag(
    question_id: str,
    question_text: str,
    resource_context: str | None = None,  # noqa: ARG001 — see docstring
    battery_id: str | None = None,
    battery_run_id: str | None = None,
) -> RunResult:
    """Call UKY RAG directly — no agent graph. Simulates current production.

    Intentionally ignores resource_context: current prod does not do
    resource-scoped RAG, so the baseline shouldn't either.

    Like the agent path, writes a battery turn_reports row when battery_run_id
    is set — raw_rag answers must reach the review UI so humans score BOTH
    systems for the judge-vs-human comparison. The state is answer-only (no
    tools, no trace); the report's tool sections come out empty.
    """
    session_id = f"eval_{battery_run_id}_{question_id}" if battery_run_id else f"eval_{question_id}"
    start = time.monotonic()
    client = get_uky_client()
    try:
        uky_response = await client.ask(
            query=question_text,
            endpoint_type="general",
            session_id=session_id,
            question_id=question_id,
        )
    except Exception:
        # Mirror the agent path: a failed battery question must be
        # distinguishable from one that never ran.
        if battery_run_id:
            await _report_battery_turn(
                state={},
                session_id=session_id,
                question_id=question_id,
                query_text=question_text,
                duration_ms=(time.monotonic() - start) * 1000,
                battery_id=battery_id,
                battery_run_id=battery_run_id,
                success=False,
            )
        raise
    if battery_run_id:
        await _report_battery_turn(
            state={"final_answer": uky_response.response},
            session_id=session_id,
            question_id=question_id,
            query_text=question_text,
            duration_ms=(time.monotonic() - start) * 1000,
            battery_id=battery_id,
            battery_run_id=battery_run_id,
            success=bool(uky_response.response),
        )
    return RunResult(
        question_id=question_id,
        question_text=question_text,
        answer=uky_response.response,
        success=bool(uky_response.response),
    )


async def _run_agent(
    question_id: str,
    question_text: str,
    tool_catalog: Any,
    resource_context: str | None = None,
    battery_id: str | None = None,
    battery_run_id: str | None = None,
) -> RunResult:
    """Call run_agent() and capture the final answer + execution context.

    When battery_run_id is set, also write a turn_reports row (origin='battery')
    mirroring the API layer's write in src/api/routes.py — that's what makes
    battery runs visible in the reporting dashboard. The session id is
    namespaced per run so two runs of the same battery never merge into one
    dashboard session.
    """
    session_id = f"eval_{battery_run_id}_{question_id}" if battery_run_id else f"eval_{question_id}"
    reset_turn_capture()
    start = time.monotonic()
    try:
        state = await run_agent(
            query=question_text,
            session_id=session_id,
            question_id=question_id,
            tool_catalog=tool_catalog,
            use_checkpointing=False,
            resource_context=resource_context,
        )
    except Exception:
        # Mirror src/api/routes.py's failure write: a failed battery question
        # must be distinguishable from one that never ran.
        if battery_run_id:
            await _report_battery_turn(
                state={},
                session_id=session_id,
                question_id=question_id,
                query_text=question_text,
                duration_ms=(time.monotonic() - start) * 1000,
                battery_id=battery_id,
                battery_run_id=battery_run_id,
                success=False,
            )
        raise
    duration_ms = (time.monotonic() - start) * 1000

    answer = state.get("final_answer", "")
    if battery_run_id:
        await _report_battery_turn(
            state=state,
            session_id=session_id,
            question_id=question_id,
            query_text=question_text,
            duration_ms=duration_ms,
            battery_id=battery_id,
            battery_run_id=battery_run_id,
            success=bool(answer),
        )
    return RunResult(
        question_id=question_id,
        question_text=question_text,
        answer=answer or "",
        rag_context=_format_rag_matches(state),
        tool_results=_format_tool_results(state),
        node_trace=_format_node_trace(state),
        tools_used=state.get("tools_used", []),
        success=bool(answer),
    )


async def _report_battery_turn(
    *,
    state: dict[str, Any] | AgentState,
    session_id: str,
    question_id: str,
    query_text: str,
    duration_ms: float,
    battery_id: str | None,
    battery_run_id: str,
    success: bool,
) -> None:
    """Best-effort: a reporting failure must never fail the eval run."""
    try:
        resources = await resources_for_turn(query_text, str(state.get("final_answer") or ""))
        get_turn_reporter().log_turn_report(
            final_state=state,
            session_id=session_id,
            turn_index=1,  # battery questions are single-turn sessions
            question_id=question_id,
            query_text=query_text,
            duration_ms=duration_ms,
            acting_user=None,
            success=success,
            capabilities=get_capability_registry().infer_capability_ids(
                state.get("tool_results", [])
            ),
            resources=resources,
            turn_capture=get_turn_capture(),
            judge=None,
            origin="battery",
            battery_id=battery_id,
            battery_run_id=battery_run_id,
        )
    except Exception:
        logger.exception("Battery turn report write failed (run continues)")
