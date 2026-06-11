"""Per-request side-channel for turn observations absent from final_state.

The reporting layer needs two facts the agent graph doesn't surface in
final_state: the structured retrieval chunks (search_access_documents flattens
them to a string before returning) and whether SummarizationMiddleware fired.
Both happen deep inside the loop.

We carry them out via a ContextVar holding one mutable dict. The route resets
it per turn; the doc-search tool and the summarization middleware mutate it
**in place**; the route reads it after the stream and hands it to the reporter.
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


def reset_turn_capture() -> None:
    """Start a fresh capture for the current turn (call once per request)."""
    _turn_capture.set({"searched": False, "chunks": [], "summarized": False})


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


def get_turn_capture() -> dict[str, Any]:
    """Read the current turn's capture (safe default if never reset)."""
    return _turn_capture.get() or {"searched": False, "chunks": [], "summarized": False}
