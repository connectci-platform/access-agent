"""Tool registry for loading and managing MCP tool catalog."""

import asyncio
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from ..config import settings

logger = logging.getLogger(__name__)


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""

    name: str
    type: str = "string"
    required: bool = False
    description: str = ""


class ToolDefinition(BaseModel):
    """Definition of an MCP tool."""

    name: str
    description: str
    server: str
    parameters: list[ToolParameter] = []


class ToolRegistry:
    """Registry for MCP tool catalog.

    Loads tool definitions from a catalog (pre-loaded, file, or URL) and provides
    lookup functionality.
    """

    def __init__(
        self,
        catalog: dict[str, Any] | None = None,
        catalog_path: str | None = None,
        catalog_url: str | None = None,
    ):
        """Initialize the tool registry.

        Args:
            catalog: Pre-loaded catalog dict (from CatalogAggregator).
            catalog_path: Path to local catalog JSON file.
            catalog_url: URL to fetch catalog from.
        """
        self._catalog_path = catalog_path or settings.MCP_CATALOG_PATH
        self._catalog_url = catalog_url or settings.MCP_CATALOG_URL
        self._catalog: dict[str, Any] = {}
        self._tools: dict[str, ToolDefinition] = {}
        self._quick_lookup: dict[str, dict[str, str]] = {}

        # If catalog provided, build registry immediately
        if catalog:
            self._catalog = catalog
            self._build_registry()
            self._apply_capability_filter()

    async def load(self) -> None:
        """Load the tool catalog from file or URL.

        Note: If catalog was provided to __init__, this is a no-op.
        """
        if self._catalog:
            # Already loaded from constructor
            return

        if self._catalog_path:
            self._catalog = self._load_from_file(self._catalog_path)
        elif self._catalog_url:
            self._catalog = await self._load_from_url(self._catalog_url)
        else:
            raise ValueError("Must provide either catalog_path or catalog_url")

        self._build_registry()
        self._apply_capability_filter()

    def _load_from_file(self, path: str) -> dict[str, Any]:
        """Load catalog from a local JSON file."""
        with Path(path).open() as f:
            result: dict[str, Any] = json.load(f)
            return result

    async def _load_from_url(self, url: str) -> dict[str, Any]:
        """Load catalog from a URL."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result

    def _build_registry(self) -> None:
        """Build the tool registry from catalog data."""
        # Extract quick lookup if available
        self._quick_lookup = self._catalog.get("quick_lookup", {})

        # Build tool definitions
        self._tools = {}

        # Handle different catalog formats
        if "servers" in self._catalog:
            # Full catalog format with servers array
            for server_info in self._catalog["servers"]:
                server_name = server_info.get("server", "")
                for tool in server_info.get("tools", []):
                    self._add_tool(tool, server_name)
        elif "tools" in self._catalog:
            # Compact format with tools array
            for tool in self._catalog["tools"]:
                # Look up server from quick_lookup
                server_name = self._quick_lookup.get(tool["name"], {}).get("server", "")
                self._add_tool(tool, server_name)

    def _apply_capability_filter(self) -> None:
        """Drop tools from MCP servers whose owning capability is disabled.

        The capability registry is the single source of truth for what the
        agent can do; this method enforces that at the tool catalog layer
        so the planner never sees tools from disabled capabilities.
        """
        # Local import to avoid a circular dependency at module load.
        from ..agent.domains.capabilities import get_capability_registry

        try:
            allowed = get_capability_registry().enabled_mcp_servers()
        except Exception as exc:
            # Belt-and-suspenders: if the registry fails to build, don't
            # silently drop every tool. Log and skip the filter.
            logger.warning("Capability registry unavailable, skipping tool filter: %s", exc)
            return

        before = len(self._tools)
        filtered = {name: tool for name, tool in self._tools.items() if tool.server in allowed}
        self._tools = filtered
        # Also strip filtered tools from quick_lookup so get_server_for_tool
        # can't resurrect them via the fallback path.
        self._quick_lookup = {
            name: info for name, info in self._quick_lookup.items() if name in filtered
        }
        # Filter the raw catalog dict too — this is what gets passed to the
        # agent graph as tool_catalog state, and the planner reads it
        # directly. Without this, disabled tools leak into the agent.
        if "servers" in self._catalog:
            self._catalog["servers"] = [
                {
                    **s,
                    "tools": [t for t in s.get("tools", []) if t.get("name", "") in filtered],
                }
                for s in self._catalog.get("servers", [])
                if s.get("server", "") in allowed
            ]
        if "tools" in self._catalog:
            self._catalog["tools"] = [
                t for t in self._catalog["tools"] if t.get("name", "") in filtered
            ]
        if "quick_lookup" in self._catalog:
            self._catalog["quick_lookup"] = self._quick_lookup

        after = len(self._tools)
        logger.info(
            "Tool catalog filtered by capabilities: %d → %d tools (%d MCP servers allowed)",
            before,
            after,
            len(allowed),
        )

    def _add_tool(self, tool_data: dict[str, Any], server_name: str) -> None:
        """Add a tool to the registry."""
        parameters = []
        for param in tool_data.get("parameters", []):
            parameters.append(
                ToolParameter(
                    name=param.get("name", ""),
                    type=param.get("type", "string"),
                    required=param.get("required", False),
                    description=param.get("description", ""),
                )
            )

        tool_def = ToolDefinition(
            name=tool_data["name"],
            description=tool_data.get("description", ""),
            server=server_name,
            parameters=parameters,
        )
        self._tools[tool_def.name] = tool_def

    @property
    def catalog(self) -> dict[str, Any]:
        """Get the raw catalog data."""
        return self._catalog

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        """Get all registered tools."""
        return self._tools

    @property
    def tool_count(self) -> int:
        """Get the number of registered tools."""
        return len(self._tools)

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def get_server_for_tool(self, tool_name: str) -> str | None:
        """Get the server name for a tool."""
        tool = self._tools.get(tool_name)
        if tool:
            return tool.server

        # Fallback to quick_lookup
        lookup = self._quick_lookup.get(tool_name, {})
        return lookup.get("server")

    def get_catalog_for_prompt(self) -> str:
        """Get a compact catalog representation for LLM prompts.

        Returns a formatted string with tool names, descriptions, and parameters.
        """
        lines = []
        for tool in self._tools.values():
            params_str = ", ".join(
                f"{p.name}: {p.type}" + ("*" if p.required else "") for p in tool.parameters
            )
            lines.append(f"- {tool.name}({params_str}): {tool.description[:100]}")

        return "\n".join(lines)

    def validate_tool_call(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> tuple[bool, str | None]:
        """Validate a tool call against the catalog.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            Tuple of (is_valid, error_message).
        """
        tool = self._tools.get(tool_name)
        if not tool:
            return False, f"Unknown tool: {tool_name}"

        # Check required parameters
        for param in tool.parameters:
            if param.required and param.name not in arguments:
                return False, f"Missing required parameter: {param.name}"

        return True, None


class CatalogAggregator:
    """Aggregates tool catalogs from multiple MCP servers.

    Fetches the /tools endpoint from each configured MCP server and
    combines them into a unified catalog.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        server_urls: dict[str, str] | None = None,
    ):
        """Initialize the aggregator.

        Args:
            timeout: HTTP timeout for each server request.
            server_urls: Optional dict of server names to URLs. If not provided,
                uses settings.mcp_server_urls.
        """
        self._timeout = timeout
        self._server_urls = server_urls
        self._catalog: dict[str, Any] = {}
        self._last_refresh: datetime | None = None

    async def fetch_catalog(self, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch and aggregate catalogs from all MCP servers.

        Args:
            force_refresh: Force refresh even if catalog is cached.

        Returns:
            Aggregated catalog with all tools.
        """
        if self._catalog and not force_refresh and self._last_refresh:
            # Return cached catalog if available
            return self._catalog

        server_urls = self._server_urls or settings.mcp_server_urls
        logger.info(f"Fetching catalog from {len(server_urls)} MCP servers")

        # Fetch from all servers in parallel
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [
                self._fetch_server_tools(client, server_url) for server_url in server_urls.values()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Build aggregated catalog
        servers = []
        quick_lookup: dict[str, dict[str, str]] = {}
        total_tools = 0

        for (server_name, server_url), result in zip(server_urls.items(), results, strict=True):
            if isinstance(result, BaseException):
                logger.warning(f"Failed to fetch from {server_name}: {result}")
                servers.append(
                    {
                        "server": server_name,
                        "url": server_url,
                        "status": "unavailable",
                        "error": str(result),
                        "tool_count": 0,
                        "tools": [],
                    }
                )
                continue

            tools = result.get("tools", [])
            tool_count = len(tools)
            total_tools += tool_count

            # Process tools and build quick_lookup
            processed_tools = []
            for tool in tools:
                tool_name = tool.get("name", "")
                processed_tool = self._process_tool(tool)
                processed_tools.append(processed_tool)

                # Add to quick_lookup
                quick_lookup[tool_name] = {
                    "server": server_name,
                    "url": f"{server_url}/tools/{tool_name}",
                }

            servers.append(
                {
                    "server": server_name,
                    "url": server_url,
                    "status": "available",
                    "tool_count": tool_count,
                    "tools": processed_tools,
                }
            )
            logger.info(f"Fetched {tool_count} tools from {server_name}")

        self._catalog = {
            "generated_at": datetime.now(UTC).isoformat(),
            "total_servers": len(server_urls),
            "servers_available": sum(1 for s in servers if s["status"] == "available"),
            "total_tools": total_tools,
            "servers": servers,
            "quick_lookup": quick_lookup,
        }
        self._last_refresh = datetime.now(UTC)

        logger.info(
            f"Catalog aggregated: {total_tools} tools from "
            f"{self._catalog['servers_available']}/{len(server_urls)} servers"
        )

        return self._catalog

    async def _fetch_server_tools(
        self,
        client: httpx.AsyncClient,
        server_url: str,
    ) -> dict[str, Any]:
        """Fetch tools from a single MCP server.

        Args:
            client: HTTP client to use.
            server_url: Base URL of the server.

        Returns:
            Dict with tools array.
        """
        url = f"{server_url}/tools"
        response = await client.get(url)
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def _process_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        """Process a tool definition into standard format.

        Args:
            tool: Raw tool definition from MCP server.

        Returns:
            Processed tool with standardized parameters.
        """
        # Extract parameters from inputSchema
        parameters = []
        input_schema = tool.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])

        for param_name, param_def in properties.items():
            parameters.append(
                {
                    "name": param_name,
                    "type": param_def.get("type", "string"),
                    "required": param_name in required,
                    "description": param_def.get("description", ""),
                    "enum": param_def.get("enum"),
                    "default": param_def.get("default"),
                }
            )

        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": parameters,
        }

    @property
    def catalog(self) -> dict[str, Any]:
        """Get the current cached catalog."""
        return self._catalog

    @property
    def last_refresh(self) -> datetime | None:
        """Get the timestamp of the last catalog refresh."""
        return self._last_refresh


# Global aggregator instance
_aggregator: CatalogAggregator | None = None


def get_catalog_aggregator() -> CatalogAggregator:
    """Get the global catalog aggregator instance."""
    global _aggregator
    if _aggregator is None:
        _aggregator = CatalogAggregator()
    return _aggregator
