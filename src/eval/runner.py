"""Run eval questions through the agent and collect results."""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.graph import run_agent

logger = logging.getLogger(__name__)


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
    info = {}
    try:
        info["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    except Exception:
        info["commit"] = "unknown"
        info["branch"] = "unknown"
    return info


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
    results = state.get("tool_results", [])
    if not results:
        return None
    parts = []
    for r in results:
        if hasattr(r, "tool_name"):
            parts.append(f"Tool: {r.tool_name}\n{r.data!s}")
        elif isinstance(r, dict):
            parts.append(f"Tool: {r.get('tool_name', 'unknown')}\n{r.get('data', '')!s}")
    return "\n---\n".join(parts) if parts else None


def _format_node_trace(state: Any) -> str | None:
    trace = state.get("node_trace", [])
    if not trace:
        return None
    return json.dumps(trace, indent=2, default=str)


async def run_question(
    question_id: str,
    question_text: str,
    tool_catalog: Any,
) -> RunResult:
    start = time.monotonic()
    try:
        state = await run_agent(
            query=question_text,
            session_id=f"eval_{question_id}",
            question_id=question_id,
            tool_catalog=tool_catalog,
            use_checkpointing=False,
        )
        duration_ms = (time.monotonic() - start) * 1000
        answer = state.get("final_answer", "")
        return RunResult(
            question_id=question_id,
            question_text=question_text,
            answer=answer or "",
            rag_context=_format_rag_matches(state),
            tool_results=_format_tool_results(state),
            node_trace=_format_node_trace(state),
            tools_used=state.get("tools_used", []),
            duration_ms=duration_ms,
            success=bool(answer),
        )
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
