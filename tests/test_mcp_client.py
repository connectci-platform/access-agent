"""Tests for MCP client."""

import httpx
import pytest
from pytest_httpx import HTTPXMock

from src.config import settings
from src.tools.mcp_client import (
    MCPClient,
    MCPToolResult,
    get_shared_client,
)


class TestMCPToolResult:
    """Tests for MCPToolResult model."""

    def test_successful_result(self):
        result = MCPToolResult(
            success=True,
            data={"resources": [{"name": "test"}]},
            duration_ms=100,
        )
        assert result.success is True
        assert result.data == {"resources": [{"name": "test"}]}
        assert result.error is None
        assert result.duration_ms == 100

    def test_failed_result(self):
        result = MCPToolResult(
            success=False,
            error="Connection timeout",
            duration_ms=5000,
        )
        assert result.success is False
        assert result.data is None
        assert result.error == "Connection timeout"


class TestMCPClient:
    """Tests for MCPClient."""

    def test_get_server_url_known(self):
        client = MCPClient()
        url = client.get_server_url("compute-resources")
        assert "compute-resources" in url or "3002" in url

    def test_get_server_url_unknown(self):
        client = MCPClient()
        with pytest.raises(ValueError, match="Unknown MCP server"):
            client.get_server_url("nonexistent-server")

    @pytest.mark.asyncio
    async def test_call_tool_success(self, httpx_mock: HTTPXMock):
        """Test successful tool call."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:3002/tools/search_resources",
            json={
                "content": [
                    {
                        "type": "text",
                        "text": '{"resources": [{"name": "ACES"}]}',
                    }
                ]
            },
        )

        client = MCPClient()
        # Override server URL for test
        client._server_urls["compute-resources"] = "http://localhost:3002"

        result = await client.call_tool(
            server="compute-resources",
            tool_name="search_resources",
            arguments={"query": "ACES"},
        )

        assert result.success is True
        assert result.data == {"resources": [{"name": "ACES"}]}
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_call_tool_with_acting_user(self, httpx_mock: HTTPXMock):
        """Test tool call includes X-Acting-User header when provided."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:3002/tools/create_announcement",
            json={
                "content": [
                    {
                        "type": "text",
                        "text": '{"id": "123", "title": "Test"}',
                    }
                ]
            },
        )

        client = MCPClient()
        client._server_urls["announcements"] = "http://localhost:3002"

        result = await client.call_tool(
            server="announcements",
            tool_name="create_announcement",
            arguments={"title": "Test"},
            acting_user="jsmith@access-ci.org",
        )

        assert result.success is True

        # Verify the header was sent
        request = httpx_mock.get_request()
        assert request.headers.get("X-Acting-User") == "jsmith@access-ci.org"

    @pytest.mark.asyncio
    async def test_call_tool_without_acting_user_no_header(self, httpx_mock: HTTPXMock):
        """Test tool call omits X-Acting-User header when not provided."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:3002/tools/search_resources",
            json={
                "content": [
                    {
                        "type": "text",
                        "text": '{"resources": []}',
                    }
                ]
            },
        )

        client = MCPClient()
        client._server_urls["compute-resources"] = "http://localhost:3002"

        result = await client.call_tool(
            server="compute-resources",
            tool_name="search_resources",
            arguments={"query": "test"},
        )

        assert result.success is True

        # Verify the header was NOT sent
        request = httpx_mock.get_request()
        assert "X-Acting-User" not in request.headers

    @pytest.mark.asyncio
    async def test_call_tool_http_error(self, httpx_mock: HTTPXMock):
        """Test tool call with HTTP error."""
        httpx_mock.add_response(
            method="POST",
            url="http://localhost:3002/tools/search_resources",
            status_code=500,
            text="Internal Server Error",
        )

        client = MCPClient()
        client._server_urls["compute-resources"] = "http://localhost:3002"

        result = await client.call_tool(
            server="compute-resources",
            tool_name="search_resources",
            arguments={"query": "test"},
        )

        assert result.success is False
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_call_tool_unknown_server(self):
        """Test tool call with unknown server."""
        client = MCPClient()
        result = await client.call_tool(
            server="unknown-server",
            tool_name="some_tool",
            arguments={},
        )

        assert result.success is False
        assert "Unknown MCP server" in result.error

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("server", "seconds"),
        [("xdmod", 60.0), ("compute-resources", 30.0)],
    )
    async def test_call_tool_timeout_depends_on_server(
        self, httpx_mock: HTTPXMock, server: str, seconds: float
    ):
        """Slow XDMoD queries get longer than other servers, per request."""
        httpx_mock.add_response(method="POST", json={"content": []})
        client = MCPClient()
        client._server_urls[server] = "http://localhost:3002"

        await client.call_tool(server=server, tool_name="any_tool", arguments={})

        timeout = httpx_mock.get_request().extensions["timeout"]
        assert timeout["read"] == seconds

    def test_slowest_server_leaves_time_to_answer(self):
        """A timed-out tool call must not use up the whole turn."""
        assert max(settings.mcp_server_timeouts.values()) <= settings.AGENT_TURN_TIMEOUT_S / 2

    def test_parse_mcp_response_text_content(self):
        """Test parsing MCP text content response."""
        client = MCPClient()
        data = {"content": [{"type": "text", "text": '{"key": "value"}'}]}
        result = client._parse_mcp_response(data)
        assert result == {"key": "value"}

    def test_parse_mcp_response_plain_text(self):
        """Test parsing plain text response."""
        client = MCPClient()
        data = {"content": [{"type": "text", "text": "Plain text response"}]}
        result = client._parse_mcp_response(data)
        assert result == "Plain text response"

    def test_parse_mcp_response_direct_data(self):
        """Test parsing direct data (non-MCP format)."""
        client = MCPClient()
        data = {"direct": "data"}
        result = client._parse_mcp_response(data)
        assert result == {"direct": "data"}


class TestSharedClient:
    """Tests for shared HTTP client."""

    def test_get_shared_client_creates_client(self):
        client = get_shared_client()
        assert client is not None
        assert not client.is_closed

    def test_get_shared_client_returns_same_instance(self):
        client1 = get_shared_client()
        client2 = get_shared_client()
        assert client1 is client2


def test_call_tool_emits_span(monkeypatch):
    import asyncio

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from src.tools.mcp_client import MCPClient

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "src.telemetry.spans.get_tracer",
        lambda name="test": provider.get_tracer(name),
    )

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    class _Client:
        async def post(self, url, json=None, headers=None, timeout=None):
            return _Resp()

    monkeypatch.setattr("src.tools.mcp_client.get_shared_client", lambda timeout: _Client())
    monkeypatch.setattr(MCPClient, "get_server_url", lambda self, s: "http://x")

    result = asyncio.run(
        MCPClient().call_tool(server="srv", tool_name="list_things", arguments={"a": 1})
    )
    assert result.success

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "mcp.call_tool.list_things"
    assert spans[0].attributes["mcp.server"] == "srv"
    assert spans[0].attributes["mcp.success"] is True
    assert spans[0].attributes["mcp.duration_ms"] >= 0


def test_call_tool_failure_span_has_error_status(monkeypatch):
    import asyncio

    import httpx
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import StatusCode

    from src.tools.mcp_client import MCPClient

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "src.telemetry.spans.get_tracer",
        lambda name="test": provider.get_tracer(name),
    )

    class _Client:
        async def post(self, url, json=None, headers=None, timeout=None):
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr("src.tools.mcp_client.get_shared_client", lambda timeout: _Client())
    monkeypatch.setattr(MCPClient, "get_server_url", lambda self, s: "http://x")

    result = asyncio.run(MCPClient().call_tool(server="srv", tool_name="list_things", arguments={}))
    assert not result.success

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].attributes["mcp.success"] is False
    assert "Timeout" in spans[0].attributes["mcp.error"]
    assert spans[0].status.status_code is StatusCode.ERROR


def test_call_tool_generic_exception_returns_error(monkeypatch):
    import asyncio

    from src.tools.mcp_client import MCPClient

    class _Client:
        async def post(self, url, json=None, headers=None, timeout=None):
            raise ValueError("kaboom")

    monkeypatch.setattr("src.tools.mcp_client.get_shared_client", lambda timeout: _Client())
    monkeypatch.setattr(MCPClient, "get_server_url", lambda self, s: "http://x")

    result = asyncio.run(MCPClient().call_tool(server="srv", tool_name="t", arguments={}))
    assert not result.success
    assert result.error.startswith("ValueError: kaboom")


def _mcp_span_for(monkeypatch, post, arguments):
    """Run one call_tool through a stubbed HTTP client; return (result, span)."""
    import asyncio

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from src.tools.mcp_client import MCPClient

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "src.telemetry.spans.get_tracer",
        lambda name="test": provider.get_tracer(name),
    )

    class _Client:
        async def post(self, url, json=None, headers=None, timeout=None):
            return await post()

    monkeypatch.setattr("src.tools.mcp_client.get_shared_client", lambda timeout: _Client())
    monkeypatch.setattr(MCPClient, "get_server_url", lambda self, s: "http://x")
    result = asyncio.run(
        MCPClient().call_tool(server="srv", tool_name="create_ticket", arguments=arguments)
    )
    (span,) = exporter.get_finished_spans()
    return result, span


def _exported(span) -> str:
    values = [str(v) for v in span.attributes.values()]
    values += [str(v) for e in span.events for v in (e.attributes or {}).values()]
    return " ".join([*values, span.status.description or ""])


SECRET = "my allocation on anvil ran out, I am Jane Doe"


def test_mcp_span_records_argument_keys_not_values(monkeypatch):
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    async def post():
        return _Resp()

    _, span = _mcp_span_for(monkeypatch, post, {"title": "help", "description": SECRET})
    assert span.attributes["mcp.argument_keys"] == "description,title"
    assert SECRET not in _exported(span)
    assert "help" not in _exported(span)


def test_mcp_span_error_omits_response_body(monkeypatch):
    import httpx

    request = httpx.Request("POST", "http://x")
    response = httpx.Response(422, text=f"invalid description: {SECRET}", request=request)

    async def post():
        raise httpx.HTTPStatusError("bad", request=request, response=response)

    result, span = _mcp_span_for(monkeypatch, post, {"description": SECRET})
    assert SECRET in result.error  # the agent still gets the detail
    assert span.attributes["mcp.error"] == "HTTP 422"
    assert span.status.description == "HTTP 422"
    assert SECRET not in _exported(span)


def test_mcp_span_error_omits_exception_text(monkeypatch):
    async def post():
        raise ValueError(f"could not parse {SECRET}")

    result, span = _mcp_span_for(monkeypatch, post, {})
    assert SECRET in result.error
    assert span.attributes["mcp.error"] == "ValueError"
    assert SECRET not in _exported(span)


def test_trace_mcp_call_records_only_the_exception_type(monkeypatch):
    import pytest
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.trace import StatusCode

    from src.telemetry.spans import trace_mcp_call

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "src.telemetry.spans.get_tracer",
        lambda name="test": provider.get_tracer(name),
    )

    with pytest.raises(ValueError), trace_mcp_call("srv", "create_ticket", {"d": SECRET}):
        raise ValueError(SECRET)

    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.events[0].attributes["exception.type"] == "ValueError"
    assert SECRET not in _exported(span)


async def _timeout_sent_for(monkeypatch, server: str, client: MCPClient) -> float:
    """The timeout call_tool puts on the request to ``server``."""
    seen: list[float] = []

    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"ok": True}

    class _Client:
        async def post(self, url, json=None, headers=None, timeout=None):
            seen.append(timeout)
            return _Resp()

    monkeypatch.setattr("src.tools.mcp_client.get_shared_client", lambda timeout: _Client())
    monkeypatch.setattr(MCPClient, "get_server_url", lambda self, s: "http://x")
    await client.call_tool(server=server, tool_name="t", arguments={})
    return seen[0]


@pytest.mark.parametrize("server", ["xdmod", "xdmod-data"])
async def test_xdmod_servers_get_the_longer_timeout_on_each_request(monkeypatch, server):
    # On the request, because the client is shared and keeps the first timeout it
    # was built with, which can be health_check's 5 seconds.
    monkeypatch.setattr("src.tools.mcp_client.settings.XDMOD_TOOL_TIMEOUT_S", 75.0)

    assert await _timeout_sent_for(monkeypatch, server, MCPClient()) == 75.0


async def test_other_servers_keep_the_default_timeout(monkeypatch):
    assert await _timeout_sent_for(monkeypatch, "compute-resources", MCPClient()) == 30.0


async def test_a_client_timeout_applies_to_servers_with_no_setting_of_their_own(monkeypatch):
    client = MCPClient(timeout=7.0)

    assert await _timeout_sent_for(monkeypatch, "compute-resources", client) == 7.0


@pytest.mark.asyncio
async def test_a_timeout_is_marked_so_code_need_not_read_the_error_text(httpx_mock: HTTPXMock):
    httpx_mock.add_exception(httpx.ReadTimeout("slow"))

    result = await MCPClient().call_tool("xdmod", "get_chart_data", {})

    assert result.timed_out is True
    assert result.error == "Timeout calling xdmod/get_chart_data"


@pytest.mark.asyncio
async def test_other_failures_are_not_marked_as_timeouts(httpx_mock: HTTPXMock):
    httpx_mock.add_response(status_code=500)

    result = await MCPClient().call_tool("xdmod", "get_chart_data", {})

    assert result.success is False
    assert result.timed_out is False


async def test_a_cancelled_call_marks_its_span_as_an_error(monkeypatch):
    import asyncio
    from unittest.mock import MagicMock

    from opentelemetry.trace import StatusCode

    span = MagicMock()
    trace = MagicMock()
    trace.return_value.__enter__.return_value = span
    monkeypatch.setattr("src.tools.mcp_client.trace_mcp_call", trace)

    class _Client:
        async def post(self, url, json=None, headers=None, timeout=None):
            await asyncio.Event().wait()

    monkeypatch.setattr("src.tools.mcp_client.get_shared_client", lambda timeout: _Client())
    monkeypatch.setattr(MCPClient, "get_server_url", lambda self, s: "http://x")

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(MCPClient().call_tool("xdmod", "get_chart_data", {}), 0.01)

    assert span.set_status.call_args.args[0].status_code == StatusCode.ERROR
    span.set_attribute.assert_any_call("mcp.success", False)
    span.set_attribute.assert_any_call("mcp.error", "cancelled")
