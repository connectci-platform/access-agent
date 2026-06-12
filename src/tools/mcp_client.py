"""HTTP client for MCP (Model Context Protocol) server communication."""

import json
import logging
import time
from typing import Any

import httpx
from pydantic import BaseModel

from ..config import settings
from ..telemetry import create_async_client
from ..telemetry.spans import trace_mcp_call

logger = logging.getLogger(__name__)

# Global shared HTTP client for connection pooling
_shared_client: httpx.AsyncClient | None = None


def get_shared_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """Get or create the shared HTTP client for connection pooling.

    Uses create_async_client() which provides OpenTelemetry instrumentation
    for trace context propagation to MCP servers.

    Args:
        timeout: Request timeout in seconds.

    Returns:
        A shared AsyncClient instance with trace context propagation.
    """
    global _shared_client
    if _shared_client is None or _shared_client.is_closed:
        _shared_client = create_async_client(timeout=timeout)
        logger.debug("Created shared HTTP client with connection pooling")
    return _shared_client


async def close_shared_client() -> None:
    """Close the shared HTTP client. Call on application shutdown."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        await _shared_client.aclose()
        _shared_client = None
        logger.debug("Closed shared HTTP client")


class MCPToolResult(BaseModel):
    """Result from an MCP tool call."""

    success: bool
    data: Any = None
    error: str | None = None
    duration_ms: int = 0


class MCPClient:
    """HTTP client for calling MCP server tools.

    MCP servers expose tools via REST HTTP endpoints:
    - URL: POST {server_url}/tools/{tool_name}
    - Body: {"arguments": {...}}
    - Response: {"content": [{"type": "text", "text": "..."}]}

    Uses a shared connection pool for efficiency.
    """

    def __init__(self, timeout: float = 30.0):
        """Initialize the MCP client.

        Args:
            timeout: Request timeout in seconds.
        """
        self.timeout = timeout
        self._server_urls = settings.mcp_server_urls

    def get_server_url(self, server_name: str) -> str:
        """Get the URL for an MCP server.

        Args:
            server_name: Name of the MCP server (e.g., 'compute-resources').

        Returns:
            The server URL.

        Raises:
            ValueError: If the server name is unknown.
        """
        url = self._server_urls.get(server_name)
        if not url:
            raise ValueError(f"Unknown MCP server: {server_name}")
        return url

    async def call_tool(
        self,
        server: str,
        tool_name: str,
        arguments: dict[str, Any],
        acting_user: str | None = None,
    ) -> MCPToolResult:
        """Execute an MCP tool call.

        Args:
            server: Name of the MCP server.
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.
            acting_user: ACCESS ID of user performing action (e.g., jsmith@access-ci.org).

        Returns:
            MCPToolResult with success status and data or error.
        """
        start_ms = int(time.time() * 1000)

        try:
            server_url = self.get_server_url(server)
        except ValueError as e:
            return MCPToolResult(
                success=False,
                error=str(e),
                duration_ms=int(time.time() * 1000) - start_ms,
            )

        url = f"{server_url}/tools/{tool_name}"
        client = get_shared_client(self.timeout)

        logger.info(f"MCP call: {tool_name} with args: {arguments}")

        # Build headers
        headers = {"Content-Type": "application/json"}
        if acting_user:
            headers["X-Acting-User"] = acting_user
        if server in settings.mcp_servers_requiring_api_key and settings.MCP_API_KEY:
            headers["X-Api-Key"] = settings.MCP_API_KEY

        with trace_mcp_call(server, tool_name, arguments) as span:
            try:
                response = await client.post(
                    url,
                    json={"arguments": arguments},
                    headers=headers,
                )
                response.raise_for_status()

                data = response.json()

                # Parse MCP response format
                # Response: {"content": [{"type": "text", "text": "{...json...}"}]}
                parsed_data = self._parse_mcp_response(data)

                logger.info(
                    f"MCP response for {tool_name}: success, data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else type(parsed_data)}"
                )

                result = MCPToolResult(
                    success=True,
                    data=parsed_data,
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

            except httpx.HTTPStatusError as e:
                result = MCPToolResult(
                    success=False,
                    error=f"HTTP {e.response.status_code}: {e.response.text[:500]}",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
            except httpx.TimeoutException:
                result = MCPToolResult(
                    success=False,
                    error=f"Timeout calling {server}/{tool_name}",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
            except Exception as e:
                result = MCPToolResult(
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

            span.set_attribute("mcp.duration_ms", result.duration_ms)
            span.set_attribute("mcp.success", result.success)
            if result.error:
                span.set_attribute("mcp.error", result.error[:300])
            return result

    def _parse_mcp_response(self, data: Any) -> Any:
        """Parse MCP response format to extract actual data.

        MCP responses come in format:
        {"content": [{"type": "text", "text": "{...json...}"}]}

        We extract and parse the text content.
        """
        # If it's already the direct data format, return as-is
        if not isinstance(data, dict):
            return data

        # Check for MCP content format
        content = data.get("content")
        if not content or not isinstance(content, list):
            return data

        # Find text content
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text", "")
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text

        return data

    async def health_check(self, server: str) -> bool:
        """Check if an MCP server is reachable.

        Args:
            server: Name of the MCP server.

        Returns:
            True if server responds, False otherwise.
        """
        try:
            server_url = self.get_server_url(server)
            client = get_shared_client(5.0)
            response = await client.get(f"{server_url}/health")
            return response.status_code == 200
        except Exception:
            return False
