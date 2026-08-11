import asyncio

import httpx
import pytest
from pytest_httpx import HTTPXMock

from src.redteam.replayer import post_query, redteam_headers, replay_item
from src.redteam.sample import SampleResult
from src.redteam.suite import PromptEntry, SuiteItem


def _item():
    e = PromptEntry("id-1", "PLACEHOLDER", "wrapped", "w", "p", "c", "d", "s")
    return SuiteItem(id="id-1", text="PLACEHOLDER", expected="defended", entry=e)


def test_redteam_headers_shape():
    h = redteam_headers("v1-2026-05-08", "run-xyz")
    assert h["X-Redteam-Suite"] == "v1-2026-05-08"
    assert h["X-Redteam-Run-Id"] == "run-xyz"
    assert "X-Redteam" in h


@pytest.mark.asyncio
async def test_replay_item_fresh_session_per_sample():
    seen_sessions = []

    async def fake_post(client, base_url, prompt_text, session_id, headers):
        seen_sessions.append(session_id)
        return SampleResult(text="resp")

    sem = asyncio.Semaphore(6)
    out = await replay_item(
        _item(),
        base_url="http://x",
        n=5,
        concurrency_sem=sem,
        headers={},
        http_client=None,
        _post=fake_post,
    )
    assert len(out) == 5
    assert len(set(seen_sessions)) == 5  # every replay a distinct session


@pytest.mark.asyncio
async def test_replay_item_returns_sampleresults():
    async def fake_post(client, base_url, prompt_text, session_id, headers):
        return SampleResult(text="resp")

    sem = asyncio.Semaphore(6)
    out = await replay_item(
        _item(),
        base_url="http://x",
        n=3,
        concurrency_sem=sem,
        headers={},
        http_client=None,
        _post=fake_post,
    )
    assert all(isinstance(s, SampleResult) for s in out)
    assert [s.text for s in out] == ["resp", "resp", "resp"]


@pytest.mark.asyncio
async def test_replay_item_catches_post_error_as_errored_sample():
    async def boom_post(client, base_url, prompt_text, session_id, headers):
        raise RuntimeError("connect failed")

    sem = asyncio.Semaphore(6)
    out = await replay_item(
        _item(),
        base_url="http://x",
        n=2,
        concurrency_sem=sem,
        headers={},
        http_client=None,
        _post=boom_post,
    )
    assert len(out) == 2
    assert all(s.errored for s in out)  # one dead replay -> errored sample, not a crash


@pytest.mark.asyncio
async def test_post_query_normal_done_returns_text(httpx_mock: HTTPXMock):
    sse = (
        b'event: token\ndata: {"content": "hi"}\n\n'
        b'event: done\ndata: {"response": "hi there", "success": true}\n\n'
    )
    httpx_mock.add_response(content=sse)
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result == SampleResult(text="hi there")


@pytest.mark.asyncio
async def test_post_query_error_event_returns_errored(httpx_mock: HTTPXMock):
    sse = b'event: error\ndata: {"message": "boom"}\n\n'
    httpx_mock.add_response(content=sse)
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result.errored
    assert result.text == ""


@pytest.mark.asyncio
async def test_post_query_empty_done_returns_errored(httpx_mock: HTTPXMock):
    sse = b'event: done\ndata: {"success": true}\n\n'
    httpx_mock.add_response(content=sse)
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result.errored
    assert result.text == ""


@pytest.mark.asyncio
async def test_post_query_failed_done_returns_errored(httpx_mock: HTTPXMock):
    sse = b'event: done\ndata: {"response": "partial", "success": false}\n\n'
    httpx_mock.add_response(content=sse)
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result.errored


@pytest.mark.asyncio
async def test_post_query_http_error_returns_errored(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(httpx.ConnectError("connection refused"))
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result.errored
    assert result.text == ""


@pytest.mark.asyncio
async def test_post_query_blank_data_line_is_skipped(httpx_mock: HTTPXMock):
    # a `data:` line whose payload after stripping is empty must be skipped (not
    # crash json.loads on ""), and the stream still resolves via the real done event.
    sse = b'event: done\ndata: \n\nevent: done\ndata: {"response": "ok text", "success": true}\n\n'
    httpx_mock.add_response(content=sse)
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result == SampleResult(text="ok text")


@pytest.mark.asyncio
async def test_post_query_non_json_data_line_is_skipped(httpx_mock: HTTPXMock):
    # a `data:` line whose payload is not valid JSON must be swallowed, not raise.
    sse = (
        b"event: done\ndata: not-json-at-all\n\n"
        b'event: done\ndata: {"response": "recovered", "success": true}\n\n'
    )
    httpx_mock.add_response(content=sse)
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result == SampleResult(text="recovered")


@pytest.mark.asyncio
async def test_post_query_non_dict_json_payload_is_skipped(httpx_mock: HTTPXMock):
    # a `data:` payload that parses as valid JSON but isn't a dict (bare scalar)
    # must be skipped rather than calling .get() on it.
    sse = (
        b'event: done\ndata: "hi"\n\n'
        b"event: done\ndata: 5\n\n"
        b'event: done\ndata: {"response": "final answer", "success": true}\n\n'
    )
    httpx_mock.add_response(content=sse)
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result == SampleResult(text="final answer")


@pytest.mark.asyncio
async def test_post_query_token_fallback_when_done_has_no_response(httpx_mock: HTTPXMock):
    # tokens accumulate via the `token` branch; a `done` with success but no
    # `response` field must fall back to the joined tokens, not error out.
    sse = (
        b'event: token\ndata: {"content": "hel"}\n\n'
        b'event: token\ndata: {"content": "lo"}\n\n'
        b'event: done\ndata: {"success": true}\n\n'
    )
    httpx_mock.add_response(content=sse)
    async with httpx.AsyncClient() as client:
        result = await post_query(client, "http://test", "prompt", "sess-1", {})
    assert result == SampleResult(text="hello")
