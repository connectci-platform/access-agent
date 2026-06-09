import asyncio
from types import SimpleNamespace

from src.agent.turn_capture import (
    get_turn_capture,
    mark_summarized,
    record_retrieved_chunks,
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


def test_default_capture_is_safe_without_reset():
    async def _isolated():
        record_retrieved_chunks([_chunk(1, "https://a.org", "x")])
        mark_summarized()
        return get_turn_capture()

    cap = asyncio.run(_isolated())
    assert cap == {"searched": False, "chunks": [], "summarized": False}


def test_child_task_writes_visible_to_parent():
    async def _main():
        reset_turn_capture()
        await asyncio.create_task(_child())
        return get_turn_capture()

    async def _child():
        record_retrieved_chunks([_chunk(3, "https://c.org", "z")])

    cap = asyncio.run(_main())
    assert len(cap["chunks"]) == 1 and cap["chunks"][0]["rank"] == 3
