"""Tests for src.probe.docs: the docs-tool (UKY chat-mcp) health case, run
alongside the MCP-server probe table. No live calls — a fake UKY client is
injected. A raised backend error is a FAIL; a clean response (even with empty
chunks) is a PASS, matching the MCP probe's empty-result-is-a-pass rule."""

from __future__ import annotations

import httpx

from src.probe.docs import probe_docs_tool
from src.services.uky_client import UKYRetrieval


class FakeUKYClient:
    def __init__(self, retrieval: UKYRetrieval | None = None, exc: Exception | None = None):
        self._retrieval = retrieval
        self._exc = exc

    async def retrieve(self, query: str, rp_name: str | None = None) -> UKYRetrieval:
        if self._exc is not None:
            raise self._exc
        assert self._retrieval is not None
        return self._retrieval


async def test_probe_docs_tool_empty_chunks_is_success() -> None:
    client = FakeUKYClient(retrieval=UKYRetrieval(chunks=[]))

    result = await probe_docs_tool(client=client)

    assert result.tool_name == "search_access_documents"
    assert result.success is True
    assert result.error is None


async def test_probe_docs_tool_http_status_error_is_failure() -> None:
    request = httpx.Request("POST", "https://uky.example/access/chat-mcp/api/retrieve-docs")
    response = httpx.Response(400, request=request)
    client = FakeUKYClient(
        exc=httpx.HTTPStatusError("Bad Request", request=request, response=response)
    )

    result = await probe_docs_tool(client=client)

    assert result.tool_name == "search_access_documents"
    assert result.success is False
    assert result.error is not None
    assert "Bad Request" in result.error or "400" in result.error


async def test_probe_docs_tool_generic_exception_is_isolated_as_failure() -> None:
    client = FakeUKYClient(exc=RuntimeError("otel export failed"))

    result = await probe_docs_tool(client=client)

    assert result.tool_name == "search_access_documents"
    assert result.success is False
    assert result.error is not None
    assert "otel export failed" in result.error


async def test_probe_docs_tool_defaults_to_get_uky_client(monkeypatch) -> None:
    fake_client = FakeUKYClient(retrieval=UKYRetrieval(chunks=[]))
    monkeypatch.setattr(
        "src.services.uky_client.get_uky_client",
        lambda: fake_client,
    )

    result = await probe_docs_tool()

    assert result.tool_name == "search_access_documents"
    assert result.success is True
    assert result.error is None
