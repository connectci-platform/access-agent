"""Span attributes emitted by UKYClient.retrieve, in particular uky_rag.outcome."""

import asyncio

import httpx
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pytest_httpx import HTTPXMock

from src.services.uky_client import UKYClient


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    # Private provider: the global one may be owned by another library in tests.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "src.services.uky_client.get_tracer", lambda name="test": provider.get_tracer(name)
    )
    return exporter


def _retrieve_attrs(exporter: InMemorySpanExporter) -> dict:
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "uky_rag.retrieve"
    return dict(spans[0].attributes or {})


async def test_retrieve_outcome_ok(httpx_mock: HTTPXMock, exporter: InMemorySpanExporter):
    httpx_mock.add_response(json={"documents": [{"rank": 1, "text": "t", "url": "u"}]})
    await UKYClient().retrieve("q")
    attrs = _retrieve_attrs(exporter)
    assert attrs["uky_rag.outcome"] == "ok"
    assert attrs["uky_rag.chunk_count"] == 1


async def test_retrieve_outcome_empty(httpx_mock: HTTPXMock, exporter: InMemorySpanExporter):
    httpx_mock.add_response(json={"documents": []})
    await UKYClient().retrieve("q")
    attrs = _retrieve_attrs(exporter)
    assert attrs["uky_rag.outcome"] == "empty"
    assert attrs["uky_rag.chunk_count"] == 0
    assert "uky_rag.rp_name" not in attrs


async def test_retrieve_outcome_empty_scoped(httpx_mock: HTTPXMock, exporter: InMemorySpanExporter):
    httpx_mock.add_response(json={"documents": []})
    await UKYClient().retrieve("q", rp_name="delta")
    attrs = _retrieve_attrs(exporter)
    assert attrs["uky_rag.outcome"] == "empty"
    assert attrs["uky_rag.rp_name"] == "delta"


async def test_retrieve_outcome_error_http_status(
    httpx_mock: HTTPXMock, exporter: InMemorySpanExporter
):
    httpx_mock.add_response(status_code=503, text="down")
    with pytest.raises(httpx.HTTPStatusError):
        await UKYClient().retrieve("q")
    attrs = _retrieve_attrs(exporter)
    assert attrs["uky_rag.outcome"] == "error"
    assert "uky_rag.chunk_count" not in attrs


async def test_retrieve_outcome_error_timeout(
    httpx_mock: HTTPXMock, exporter: InMemorySpanExporter
):
    httpx_mock.add_exception(httpx.ReadTimeout("timed out"))
    with pytest.raises(httpx.ReadTimeout):
        await UKYClient().retrieve("q")
    attrs = _retrieve_attrs(exporter)
    assert attrs["uky_rag.outcome"] == "error"
    assert "uky_rag.chunk_count" not in attrs


@pytest.mark.parametrize(
    "body",
    [
        "not json",  # response.json() raises JSONDecodeError
        "[]",  # valid JSON, wrong shape: data.get raises AttributeError
        '{"documents": "x"}',  # string where a list is expected: doc.get raises
    ],
)
async def test_retrieve_outcome_error_bad_body(
    httpx_mock: HTTPXMock, exporter: InMemorySpanExporter, body: str
):
    httpx_mock.add_response(status_code=200, text=body)
    with pytest.raises(Exception):  # noqa: B017 - exception type differs per body
        await UKYClient().retrieve("q")
    attrs = _retrieve_attrs(exporter)
    assert attrs["uky_rag.outcome"] == "error"
    assert "uky_rag.chunk_count" not in attrs


async def test_retrieve_cancelled_leaves_no_outcome(
    httpx_mock: HTTPXMock, exporter: InMemorySpanExporter
):
    def _cancel(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    httpx_mock.add_callback(_cancel)
    with pytest.raises(asyncio.CancelledError):
        await UKYClient().retrieve("q")
    assert "uky_rag.outcome" not in _retrieve_attrs(exporter)


async def test_retrieve_outcome_rejected_scope(
    httpx_mock: HTTPXMock, exporter: InMemorySpanExporter
):
    # The agent retries this unscoped and answers, so it must not read as a failure.
    httpx_mock.add_response(status_code=400, text='{"error": "Invalid rp_name: nope"}')
    with pytest.raises(httpx.HTTPStatusError):
        await UKYClient().retrieve("q", rp_name="nope")
    assert _retrieve_attrs(exporter)["uky_rag.outcome"] == "rejected_scope"


async def test_retrieve_outcome_error_invalid_rp_name_unscoped(
    httpx_mock: HTTPXMock, exporter: InMemorySpanExporter
):
    httpx_mock.add_response(status_code=400, text="invalid rp_name")
    with pytest.raises(httpx.HTTPStatusError):
        await UKYClient().retrieve("q")
    assert _retrieve_attrs(exporter)["uky_rag.outcome"] == "error"


async def test_retrieve_outcome_error_scoped_other_400(
    httpx_mock: HTTPXMock, exporter: InMemorySpanExporter
):
    httpx_mock.add_response(status_code=400, text="query too long")
    with pytest.raises(httpx.HTTPStatusError):
        await UKYClient().retrieve("q", rp_name="delta")
    assert _retrieve_attrs(exporter)["uky_rag.outcome"] == "error"
