"""Per-request side-channel for turn observations absent from final_state.

Carries six kinds of data the agent graph doesn't surface in final_state:
retrieved chunks (search_access_documents flattens them to a string before
returning), whether SummarizationMiddleware fired, per-tool-call durations,
the turn's OTEL trace id, the per-model-call ``<think>`` reasoning the LLM
wrapper strips from responses, and the set of normalized rp_name slugs
search_access_documents has already searched this turn (so a second scoped
call to the same slug can widen instead of repeating a search the model
already saw fail — see record_scoped_search).

We carry them out via a ContextVar holding one mutable dict. The route resets
it per turn; the doc-search tool, the summarization middleware, the MCP tool
wrapper, the root-span trace-id recorder, and the LLM think-stripper mutate it
**in place**; the loop
node reads tool_timings to pair durations onto current-turn ToolResults, and
the route reads the full capture after the stream and hands it to the reporter.
Mutating-in-place (never reassigning the ContextVar inside child tasks) is what
makes writes visible across the asyncio tasks LangGraph spawns — child tasks
inherit the same dict reference. Pure observation: nothing here changes what the
LLM sees or how it answers.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..services.uky_client import UKYChunk

_SNIPPET_CHARS = 280

_turn_capture: ContextVar[dict[str, Any] | None] = ContextVar("turn_capture", default=None)


def _fresh_capture() -> dict[str, Any]:
    return {
        "searched": False,
        "chunks": [],
        "summarized": False,
        "tool_timings": [],
        "trace_id": None,
        "model_reasoning": [],
        "scoped_rp_searched": set(),
    }


def reset_turn_capture() -> None:
    """Start a fresh capture for the current turn (call once per request)."""
    _turn_capture.set(_fresh_capture())


def record_retrieved_chunks(chunks: list[UKYChunk]) -> None:
    """Record a general-corpus retrieval (an empty one still marks a search)."""
    cap = _turn_capture.get()
    if cap is None:
        return
    cap["searched"] = True
    for c in chunks:
        cap["chunks"].append(
            {"rank": c.rank, "url": c.url, "snippet": (c.text or "")[:_SNIPPET_CHARS]}
        )


def mark_summarized() -> None:
    """Record that SummarizationMiddleware compacted history this turn."""
    cap = _turn_capture.get()
    if cap is not None:
        cap["summarized"] = True


def record_tool_timing(tool_name: str, duration_ms: float) -> None:
    """Record one tool invocation's wall-clock duration.

    Appended in completion order; the loop pairs these back to reconstructed
    ToolResults FIFO per tool name (see tool_calling_loop._build_tool_results).
    """
    cap = _turn_capture.get()
    if cap is None:
        return
    cap["tool_timings"].append({"tool_name": tool_name, "duration_ms": int(duration_ms)})


def record_trace_id(trace_id: str) -> None:
    """Record the OTEL trace id (32-hex) of the turn's root span."""
    cap = _turn_capture.get()
    if cap is not None:
        cap["trace_id"] = trace_id


def record_model_reasoning(reasoning: str) -> None:
    """Record one model call's stripped ``<think>`` reasoning (call order)."""
    cap = _turn_capture.get()
    if cap is None or not reasoning:
        return
    cap["model_reasoning"].append(reasoning)


def record_scoped_search(rp_name: str) -> bool:
    """Record a scoped search_access_documents call for ``rp_name`` this turn.

    Returns True iff this exact normalized slug was already searched earlier
    in the same turn — the caller uses that to widen the second call to
    unscoped rather than repeating a scoped search the model already saw
    return unhelpful results for. Outside an active turn (capture is None,
    e.g. a direct call in a test with no reset_turn_capture()), always
    returns False — there's no per-turn state to compare against, so no
    call is ever treated as a repeat.
    """
    cap = _turn_capture.get()
    if cap is None:
        return False
    seen: set[str] = cap["scoped_rp_searched"]
    if rp_name in seen:
        return True
    seen.add(rp_name)
    return False


def get_turn_capture() -> dict[str, Any]:
    """Read the current turn's capture (safe default if never reset)."""
    return _turn_capture.get() or _fresh_capture()
