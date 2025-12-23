"""Tests for MCP client."""

import pytest
from pytest_httpx import HTTPXMock

from src.tools.mcp_client import MCPClient, MCPToolResult, get_shared_client


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
