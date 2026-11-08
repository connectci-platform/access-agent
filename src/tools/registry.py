"""Tool registry for loading and managing MCP tool catalog."""

import asyncio
import contextlib
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
            # A shallow copy: the filter below reassigns top-level keys, and
            # this dict is usually the aggregator's cached catalog, which
            # /health and /catalog read and which must never be mutated.
            self._catalog = dict(catalog)
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


# Background recovery of servers a catalog fetch found unavailable. MCP
# servers restarting alongside the agent come back within seconds to a
# minute, so the first retries are quick; the delay doubles to a one-minute
# cap so a longer outage costs one cheap request a minute, and the loop gives
# up after about a quarter hour (2+4+8+16+32+15*60 s), when the outage is one
# for a human and POST /api/v1/catalog/refresh. A server that fails after it
# was fetched is handled per call, so there is no periodic full refresh.
RECOVERY_INITIAL_DELAY = 2.0
RECOVERY_MAX_DELAY = 60.0
RECOVERY_MAX_ATTEMPTS = 20


class CatalogAggregator:
    """Aggregates tool catalogs from multiple MCP servers.

    Fetches the /tools endpoint from each configured MCP server and
    combines them into a unified catalog.

    The cached catalog is replaced, never mutated: every build assembles and
    capability-filters a new dict, then swaps it in with one assignment, so a
    reader holding ``catalog`` always has a complete, filtered snapshot.
    """

    def __init__(
        self,
        timeout: float = 10.0,
        server_urls: dict[str, str] | None = None,
        recovery_initial_delay: float = RECOVERY_INITIAL_DELAY,
        recovery_max_delay: float = RECOVERY_MAX_DELAY,
        recovery_max_attempts: int = RECOVERY_MAX_ATTEMPTS,
    ):
        """Initialize the aggregator.

        Args:
            timeout: HTTP timeout for each server request.
            server_urls: Optional dict of server names to URLs. If not provided,
                uses settings.mcp_server_urls.
            recovery_initial_delay: Seconds before the first retry of an
                unavailable server; doubles each attempt.
            recovery_max_delay: Cap on the retry delay, in seconds.
            recovery_max_attempts: Retries before the recovery loop gives up.
        """
        self._timeout = timeout
        self._server_urls = server_urls
        self._catalog: dict[str, Any] = {}
        self._last_refresh: datetime | None = None
        self._recovery_initial_delay = recovery_initial_delay
        self._recovery_max_delay = recovery_max_delay
        self._recovery_max_attempts = recovery_max_attempts
        self._recovery_task: asyncio.Task[None] | None = None

    async def fetch_catalog(self, force_refresh: bool = False) -> dict[str, Any]:
        """Fetch and aggregate catalogs from all MCP servers.

        A build that leaves servers unavailable starts the background
        recovery loop (unless one is already running).

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

        results = await self._fetch_servers(server_urls)
        servers = [
            self._server_entry(name, url, result)
            for (name, url), result in zip(server_urls.items(), results, strict=True)
        ]
        self._swap(servers)

        logger.info(
            f"Catalog aggregated: {self._catalog['total_tools']} tools from "
            f"{self._catalog['servers_available']}/{len(server_urls)} servers"
        )
        self._ensure_recovery()
        return self._catalog

    async def _fetch_servers(self, server_urls: dict[str, str]) -> list[Any]:
        """Fetch /tools from each server in parallel; failures come back as exceptions."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            tasks = [self._fetch_server_tools(client, url) for url in server_urls.values()]
            results: list[Any] = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def _server_entry(self, server_name: str, server_url: str, result: Any) -> dict[str, Any]:
        """One server's catalog entry from its /tools result (or the error)."""
        if isinstance(result, BaseException):
            logger.warning(f"Failed to fetch from {server_name}: {result}")
            return {
                "server": server_name,
                "url": server_url,
                "status": "unavailable",
                "error": str(result),
                "tool_count": 0,
                "tools": [],
            }
        tools = [self._process_tool(tool) for tool in result.get("tools", [])]
        logger.info(f"Fetched {len(tools)} tools from {server_name}")
        return {
            "server": server_name,
            "url": server_url,
            "status": "available",
            "tool_count": len(tools),
            "tools": tools,
        }

    def _swap(self, servers: list[dict[str, Any]]) -> None:
        """Build the full, filtered catalog from server entries, then swap it in.

        The new dict is complete before the single assignment, so no request
        sees a partial or unfiltered catalog.
        """
        quick_lookup: dict[str, dict[str, str]] = {}
        for server in servers:
            for tool in server["tools"]:
                quick_lookup[tool["name"]] = {
                    "server": server["server"],
                    "url": f"{server['url']}/tools/{tool['name']}",
                }
        catalog = {
            "generated_at": datetime.now(UTC).isoformat(),
            "total_servers": len(servers),
            "servers_available": sum(1 for s in servers if s["status"] == "available"),
            "total_tools": sum(s["tool_count"] for s in servers),
            "servers": servers,
            "quick_lookup": quick_lookup,
        }
        # Filter on every build, not just the first: the catalog used to be
        # capability-filtered only via ToolRegistry mutating this dict by
        # reference at startup, so a force_refresh rebuilt it UNFILTERED and
        # the public /catalog endpoint advertised disabled servers' tools
        # until restart (#240). Filtering before the swap makes every
        # (re)build, including a recovery merge, truthful on its own.
        self._catalog = self._apply_capability_filter(catalog)
        self._last_refresh = datetime.now(UTC)

    def _apply_capability_filter(self, catalog: dict[str, Any]) -> dict[str, Any]:
        """Return ``catalog`` without servers whose owning capability is disabled.

        Sibling of ``ToolRegistry._apply_capability_filter`` (which filters
        the registry's own tool/lookup state); this one keeps the
        aggregator's cached dict — the payload behind ``/api/v1/catalog`` —
        truthful across refreshes. Counts are recomputed so they reflect the
        filtered catalog.
        """
        try:
            # Local import (inside the try so an import failure degrades to
            # skip-filter rather than 500ing /catalog) to avoid a circular
            # dependency at module load.
            from ..agent.domains.capabilities import get_capability_registry

            allowed = get_capability_registry().enabled_mcp_servers()
        except Exception as exc:
            # Belt-and-suspenders: if the registry fails to build, don't
            # silently drop every server. Log and skip the filter.
            logger.warning("Capability registry unavailable, skipping catalog filter: %s", exc)
            return catalog

        servers = [s for s in catalog["servers"] if s.get("server", "") in allowed]
        # ALL counts must describe the filtered universe. /health reports
        # "degraded" (and the production deploy gate hard-fails) whenever
        # servers_available < total_servers, so leaving total_servers at the
        # raw configured count would mark every deployment with a disabled
        # capability degraded — the default prod config disables one.
        return {
            **catalog,
            "servers": servers,
            "quick_lookup": {
                name: info
                for name, info in catalog["quick_lookup"].items()
                if info.get("server", "") in allowed
            },
            "total_servers": len(servers),
            "servers_available": sum(1 for s in servers if s["status"] == "available"),
            "total_tools": sum(s.get("tool_count", 0) for s in servers),
        }

    # ── Background recovery ──────────────────────────────────────────

    def _unavailable_servers(self) -> dict[str, str]:
        """Name → URL of servers the current (filtered) catalog lacks.

        Read from the filtered catalog, so a disabled server is never retried.
        """
        return {
            s["server"]: s["url"]
            for s in self._catalog.get("servers", [])
            if s.get("status") != "available"
        }

    def _recovery_delay(self, attempt: int) -> float:
        """Seconds to wait before retry ``attempt`` (0-based)."""
        delay: float = min(self._recovery_initial_delay * 2**attempt, self._recovery_max_delay)
        return delay

    def _ensure_recovery(self) -> None:
        """Start the recovery loop if servers are missing and none is running."""
        if not self._unavailable_servers():
            return
        if self._recovery_task is not None and not self._recovery_task.done():
            return
        self._recovery_task = asyncio.create_task(self._recover(), name="catalog-recovery")

    async def _recover(self) -> None:
        """Retry only the unavailable servers with backoff until all are back."""
        for attempt in range(self._recovery_max_attempts):
            await asyncio.sleep(self._recovery_delay(attempt))
            # Re-read each attempt: a refresh may have changed what's missing.
            pending = self._unavailable_servers()
            if not pending:
                return
            try:
                results = await self._fetch_servers(pending)
            except Exception:
                logger.exception("Catalog recovery attempt %d failed", attempt + 1)
                continue
            recovered = {
                name: self._server_entry(name, url, result)
                for (name, url), result in zip(pending.items(), results, strict=True)
                if not isinstance(result, BaseException)
            }
            if recovered:
                # Merge into the catalog as it is now (a refresh may have
                # swapped it during the fetch), replacing only entries that
                # are still unavailable.
                self._swap(
                    [
                        recovered[s["server"]]
                        if s["server"] in recovered and s.get("status") != "available"
                        else s
                        for s in self._catalog["servers"]
                    ]
                )
                logger.info(
                    "Catalog recovery: %s back; %d/%d servers available",
                    ", ".join(sorted(recovered)),
                    self._catalog["servers_available"],
                    self._catalog["total_servers"],
                )
            if not self._unavailable_servers():
                return
        logger.warning(
            "Catalog recovery giving up after %d attempts; still unavailable: %s. "
            "POST /api/v1/catalog/refresh once they are back.",
            self._recovery_max_attempts,
            ", ".join(sorted(self._unavailable_servers())),
        )

    async def stop_recovery(self) -> None:
        """Cancel the recovery loop, if any, and wait for it to finish."""
        task, self._recovery_task = self._recovery_task, None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @property
    def recovery_task(self) -> "asyncio.Task[None] | None":
        """The background recovery task, if one was started."""
        return self._recovery_task

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

        # The server's own readOnlyHint, when it publishes one. None for a
        # server that hasn't, or a tool sent with "annotations": null.
        annotations = tool.get("annotations")
        hint = annotations.get("readOnlyHint") if isinstance(annotations, dict) else None
        return {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": parameters,
            "read_only": hint if isinstance(hint, bool) else None,
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
