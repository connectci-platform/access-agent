"""Token SSE events must never carry the empty-answer sentinel.

qa-bot-core concatenates every `token` event it receives (qa-flow.tsx:
`if (evType === 'token') collectedTokens += (parsed.content || '')`), so
anything streamed lands verbatim in the visible answer. EMPTY_ANSWER_SENTINEL
is wire-protocol only — it keeps the assistant turn non-empty for vLLM, and
tool_calling_loop substitutes real prose into final_answer, which the `done`
event carries. The streaming path never reaches that substitution.

These drive the real _stream_events generator and read the bytes it yields.
The usage logger and turn reporter inside it need Postgres, but both log and
swallow their failures, so the token path runs regardless.
"""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk

from src.api import routes
from src.llm.providers import EMPTY_ANSWER_SENTINEL

LOOP = {"langgraph_node": "tool_calling_loop"}


async def _fake_registry():
    """_stream_events aggregates the live MCP catalog before its stream loop."""
    return SimpleNamespace(catalog={"servers": []})


def _stream(chunks: list[tuple[object, dict]]) -> str:
    """Run the generator over scripted chunks; return what a browser concatenates."""

    async def _fake_stream(**_kwargs):
        for msg, meta in chunks:
            yield "messages", (msg, meta)

    collected: list[str] = []

    async def _consume() -> None:
        gen = routes._stream_events(
            routes.QueryRequest(query="q", session_id="s1"),
            acting_user=None,
            session_id="s1",
            question_id="q1",
            include_trace=False,
        )
        try:
            async for line in gen:
                for part in line.split("\n"):
                    if not part.startswith("data: "):
                        continue
                    try:
                        event = json.loads(part[6:])
                    except json.JSONDecodeError:
                        continue
                    if "content" in event and "session_id" not in event:
                        collected.append(event["content"])
        finally:
            await gen.aclose()

    with (
        patch.object(routes, "stream_agent", _fake_stream),
        patch.object(routes, "get_registry", _fake_registry),
    ):
        asyncio.run(_consume())
    return "".join(collected)


def test_ordinary_tokens_stream():
    # Positive control. Without it every assertion below can pass vacuously
    # against a generator that emits nothing at all.
    assert (
        _stream(
            [
                (AIMessageChunk(content="Anvil "), LOOP),
                (AIMessageChunk(content="supports Python."), LOOP),
            ]
        )
        == "Anvil supports Python."
    )


def test_sentinel_is_not_streamed():
    assert _stream([(AIMessageChunk(content=EMPTY_ANSWER_SENTINEL), LOOP)]) == ""


def test_sentinel_with_whitespace_tail_is_not_streamed():
    # The shape an equality guard misses: the strip layer forwards a
    # post-</think> whitespace chunk verbatim, so the marker arrives prefixed.
    out = _stream([(AIMessageChunk(content=f"\n\n{EMPTY_ANSWER_SENTINEL}"), LOOP)])
    assert EMPTY_ANSWER_SENTINEL not in out


def test_other_nodes_do_not_stream():
    assert _stream([(AIMessageChunk(content="internal"), {"langgraph_node": "other"})]) == ""


def test_complete_messages_do_not_stream():
    # The messages stream carries both AIMessageChunk (tokens) and the complete
    # AIMessage added to state; streaming the latter duplicates the answer.
    assert _stream([(AIMessage(content="whole answer"), LOOP)]) == ""
