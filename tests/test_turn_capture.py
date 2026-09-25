import asyncio
from types import SimpleNamespace

from src.agent.turn_capture import (
    get_turn_capture,
    mark_summarized,
    record_retrieved_chunks,
    record_scoped_search,
    record_tool_timing,
    record_trace_id,
    reset_turn_capture,
)


def _chunk(rank, url, text):
    return SimpleNamespace(rank=rank, url=url, text=text)


def test_records_chunks_lightweight():
    reset_turn_capture()
    record_retrieved_chunks(
        [_chunk(1, "https://a.org", "x" * 500), _chunk(2, "https://b.org", "y")]
    )
    cap = get_turn_capture()
    assert cap["searched"] is True
    assert len(cap["chunks"]) == 2
    assert cap["chunks"][0] == {"rank": 1, "url": "https://a.org", "snippet": "x" * 280}
    assert "text" not in cap["chunks"][0]


def test_empty_search_marks_searched_zero_chunks():
    reset_turn_capture()
    record_retrieved_chunks([])
    cap = get_turn_capture()
    assert cap["searched"] is True
    assert cap["chunks"] == []


def test_record_scoped_search_first_call_returns_false():
    reset_turn_capture()
    assert record_scoped_search("delta") is False


def test_record_scoped_search_repeat_same_slug_returns_true():
    reset_turn_capture()
    record_scoped_search("delta")
    assert record_scoped_search("delta") is True


def test_record_scoped_search_different_slug_returns_false():
    reset_turn_capture()
    record_scoped_search("delta")
    assert record_scoped_search("bridges2") is False


def test_record_scoped_search_forgotten_after_reset():
    reset_turn_capture()
    record_scoped_search("delta")
    reset_turn_capture()
    assert record_scoped_search("delta") is False


def test_record_scoped_search_without_reset_always_false():
    """Outside an active turn (capture is None), there's no per-turn state to
    compare against, so every call is treated as a first call."""
    assert record_scoped_search("delta") is False
    assert record_scoped_search("delta") is False


def test_default_capture_is_safe_without_reset():
    async def _isolated():
        record_retrieved_chunks([_chunk(1, "https://a.org", "x")])
        mark_summarized()
        return get_turn_capture()

    cap = asyncio.run(_isolated())
    assert cap == {
        "searched": False,
        "chunks": [],
        "summarized": False,
        "tool_timings": [],
        "trace_id": None,
        "model_reasoning": [],
        "scoped_rp_searched": set(),
    }


def test_child_task_writes_visible_to_parent():
    async def _main():
        reset_turn_capture()
        await asyncio.create_task(_child())
        return get_turn_capture()

    async def _child():
        record_retrieved_chunks([_chunk(3, "https://c.org", "z")])

    cap = asyncio.run(_main())
    assert len(cap["chunks"]) == 1 and cap["chunks"][0]["rank"] == 3


def test_concurrent_requests_are_isolated():
    # Each request task resets then records its own chunk; neither sees the other's.
    async def _request(rank, url):
        reset_turn_capture()
        # yield control so both tasks interleave between reset and record
        await asyncio.sleep(0)
        record_retrieved_chunks([_chunk(rank, url, "t")])
        await asyncio.sleep(0)
        return get_turn_capture()["chunks"]

    async def _main():
        import asyncio as _a

        return await _a.gather(_request(1, "https://one.org"), _request(2, "https://two.org"))

    results = asyncio.run(_main())
    assert [c[0]["url"] for c in results] == ["https://one.org", "https://two.org"]
    assert all(len(c) == 1 for c in results)  # no cross-task leakage


def test_records_tool_timings_in_order():
    reset_turn_capture()
    record_tool_timing("list_things", 42)
    record_tool_timing("search_access_documents", 7)
    cap = get_turn_capture()
    assert cap["tool_timings"] == [
        {"tool_name": "list_things", "duration_ms": 42},
        {"tool_name": "search_access_documents", "duration_ms": 7},
    ]


def test_records_trace_id():
    reset_turn_capture()
    record_trace_id("ab" * 16)
    assert get_turn_capture()["trace_id"] == "ab" * 16


def test_tool_timing_safe_without_reset():
    async def _isolated():
        record_tool_timing("x", 1)
        record_trace_id("ff" * 16)
        return get_turn_capture()

    cap = asyncio.run(_isolated())
    assert cap["tool_timings"] == [] and cap["trace_id"] is None


def test_records_model_reasoning_in_order():
    from src.agent.turn_capture import record_model_reasoning

    reset_turn_capture()
    record_model_reasoning("first call thoughts")
    record_model_reasoning("")  # empty is skipped
    record_model_reasoning("second call thoughts")
    assert get_turn_capture()["model_reasoning"] == [
        "first call thoughts",
        "second call thoughts",
    ]


def test_model_reasoning_safe_without_reset():
    from src.agent.turn_capture import record_model_reasoning

    async def _isolated():
        record_model_reasoning("x")
        return get_turn_capture()

    cap = asyncio.run(_isolated())
    assert cap["model_reasoning"] == []
