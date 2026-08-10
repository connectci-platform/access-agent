"""Format agent state into judge-prompt sections (shared by single- and multi-turn runners)."""

import json
from typing import Any


def format_rag_matches(state: Any) -> str | None:
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


def format_tool_results(state: Any) -> str | None:
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


def format_node_trace(state: Any) -> str | None:
    trace = state.get("node_trace", [])
    if not trace:
        return None
    return json.dumps(trace, indent=2, default=str)
