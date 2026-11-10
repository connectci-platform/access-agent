"""HTTP client for MCP (Model Context Protocol) server communication."""

import asyncio
import json
import logging
import time
from typing import Any

import httpx
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel

from ..config import settings
from ..telemetry import create_async_client
from ..telemetry.spans import trace_mcp_call
from .mcp_token import McpTokenUnavailableError, get_mcp_token_exchanger

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
    # The result's top-level structuredContent, kept apart so the model never sees it.
    structured: dict[str, Any] | None = None
    error: str | None = None
    duration_ms: int = 0
    timed_out: bool = False


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

    def timeout_for(self, server: str, tool_name: str | None = None) -> float:
        """The timeout a call to ``tool_name`` on ``server`` gets, in seconds."""
        tool_timeout = settings.mcp_tool_timeouts.get((server, tool_name or ""))
        if tool_timeout is not None:
            return tool_timeout
        return settings.mcp_server_timeouts.get(server, self.timeout)

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

        # Build headers. The agent keeps structuredContent out of the model's
        # context (it becomes the tool message's artifact), so it asks for it;
        # servers leave it out for callers that don't.
        headers = {
            "Content-Type": "application/json",
            "X-Include-Structured-Content": "1",
        }
        if acting_user:
            headers["X-Acting-User"] = acting_user
        if server in settings.mcp_servers_requiring_api_key and settings.MCP_API_KEY:
            headers["X-Api-Key"] = settings.MCP_API_KEY
        if acting_user and server in settings.mcp_servers_acting_for_users:
            refusal = await self._add_acting_user_token(headers, server, acting_user)
            if refusal is not None:
                return MCPToolResult(
                    success=False,
                    error=refusal,
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

        with trace_mcp_call(server, tool_name, arguments) as span:
            # span_error is what Honeycomb sees: response bodies and exception
            # text can echo tool arguments or record data, so it carries only
            # the status code or exception type. result.error keeps the detail
            # for the agent and local logs.
            span_error: str | None = None
            try:
                response = await self._post(
                    client, url, arguments, headers, server, tool_name, acting_user
                )
                response.raise_for_status()

                data = response.json()
                structured = data.get("structuredContent") if isinstance(data, dict) else None

                # Parse MCP response format
                # Response: {"content": [{"type": "text", "text": "{...json...}"}]}
                parsed_data = self._parse_mcp_response(data)

                logger.info(
                    f"MCP response for {tool_name}: success, data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else type(parsed_data)}"
                )

                result = MCPToolResult(
                    success=True,
                    data=parsed_data,
                    structured=structured if isinstance(structured, dict) else None,
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

            except httpx.HTTPStatusError as e:
                span_error = f"HTTP {e.response.status_code}"
                result = MCPToolResult(
                    success=False,
                    error=_status_error(e.response, token_sent="Authorization" in headers),
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
            except httpx.TimeoutException:
                span_error = f"Timeout calling {server}/{tool_name}"
                result = MCPToolResult(
                    success=False,
                    error=span_error,
                    duration_ms=int(time.time() * 1000) - start_ms,
                    timed_out=True,
                )
            except asyncio.CancelledError:
                # Cut off at the turn's deadline, or the turn itself was stopped.
                span.set_attribute("mcp.success", False)
                span.set_attribute("mcp.error", "cancelled")
                span.set_status(Status(StatusCode.ERROR, "cancelled"))
                raise
            except Exception as e:
                span_error = type(e).__name__
                result = MCPToolResult(
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

            span.set_attribute("mcp.duration_ms", result.duration_ms)
            span.set_attribute("mcp.success", result.success)
            if span_error:
                span.set_attribute("mcp.error", span_error)
            if not result.success:
                # call_tool converts exceptions to error results, so the context
                # manager's own ERROR handling never fires — set status here so
                # span-status error queries (e.g. the Honeycomb board) see failures.
                span.set_status(Status(StatusCode.ERROR, span_error or "tool call failed"))
            return result

    async def _add_acting_user_token(
        self, headers: dict[str, str], server: str, acting_user: str
    ) -> str | None:
        """Adds the acting-user token for a server that acts for users.

        Returns a refusal for the caller, or None to send the call. With
        MCP_ACTING_USER_TOKEN=send a failed exchange leaves X-Acting-User alone
        (rollout step 2); with require the call is not sent at all, since the
        servers refuse the bare header from step 3 on.
        """
        mode = settings.MCP_ACTING_USER_TOKEN
        if mode == "off":
            return None
        # Imported here: src.api imports the agent, which imports this module.
        from ..api.identity import current_agent_token

        try:
            token = await get_mcp_token_exchanger().token_for(acting_user, current_agent_token())
        except McpTokenUnavailableError as e:
            if mode == "send":
                logger.warning(
                    "MCP acting-user token unavailable for %s, sending X-Acting-User only: %s",
                    server,
                    e,
                )
                return None
            logger.error("MCP acting-user token unavailable for %s, call not sent: %s", server, e)
            return (
                "This action wasn't sent: the researcher's sign-in couldn't be confirmed with "
                f"the ACCESS tool server ({e}). Ask them to sign in again and retry."
            )
        headers["Authorization"] = f"Bearer {token}"
        return None

    async def _post(
        self,
        client: httpx.AsyncClient,
        url: str,
        arguments: dict[str, Any],
        headers: dict[str, str],
        server: str,
        tool_name: str,
        acting_user: str | None,
    ) -> httpx.Response:
        response = await client.post(
            url,
            json={"arguments": arguments},
            headers=headers,
            # Per request: the shared client keeps whichever timeout it
            # was first built with, which can be health_check's.
            timeout=self.timeout_for(server, tool_name),
        )
        if "Authorization" in headers and _token_refused(response):
            response = await self._after_token_refused(
                client, response, headers, server, tool_name, arguments, acting_user
            )
        return response

    async def _after_token_refused(
        self,
        client: httpx.AsyncClient,
        response: httpx.Response,
        headers: dict[str, str],
        server: str,
        tool_name: str,
        arguments: dict[str, Any],
        acting_user: str | None,
    ) -> httpx.Response:
        """A server refused the token (401) or couldn't check it (503).

        The servers answer both before running the tool, so nothing was done.
        The cached token is dropped either way. With send (rollout step 2) the
        call is retried once with X-Acting-User alone; with require the
        refusal stands, and the caller reports it as a sign-in problem.
        """
        if acting_user:
            get_mcp_token_exchanger().invalidate(acting_user)
        if settings.MCP_ACTING_USER_TOKEN != "send":
            logger.error("MCP server %s refused the acting-user token", server)
            return response
        logger.warning(
            "MCP server %s refused the acting-user token, retrying with X-Acting-User", server
        )
        del headers["Authorization"]
        return await client.post(
            str(response.request.url),
            json={"arguments": arguments},
            headers=headers,
            timeout=self.timeout_for(server, tool_name),
        )

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


def _token_refused(response: httpx.Response) -> bool:
    """A server's answer to a token it won't accept, or can't check right now."""
    if response.status_code == 401:
        return "invalid_token" in response.headers.get("www-authenticate", "")
    return response.status_code == 503


def _status_error(response: httpx.Response, *, token_sent: bool) -> str:
    error = f"HTTP {response.status_code}: {response.text[:500]}"
    if token_sent and _token_refused(response):
        return (
            "This action wasn't completed: the ACCESS tool server didn't accept the "
            f"researcher's sign-in ({error}). Ask them to sign in again and retry."
        )
    return error
