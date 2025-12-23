"""MCP tool integration for ACCESS Documentation Agent."""

from .mcp_client import MCPClient, MCPToolResult
from .registry import CatalogAggregator, ToolRegistry, get_catalog_aggregator

__all__ = [
    "CatalogAggregator",
    "MCPClient",
    "MCPToolResult",
    "ToolRegistry",
    "get_catalog_aggregator",
]
