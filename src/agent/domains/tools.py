"""MCP tool wrappers for domain agents.

Wraps MCP tools as LangChain BaseTool instances so they can be used
with create_react_agent's function-calling loop.
"""

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from ...tools.mcp_client import MCPClient
from ..turn_capture import record_tool_timing
from .config import DomainAgentConfig

logger = logging.getLogger(__name__)

# Map MCP/JSON-Schema types to Python types for pydantic model generation
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "number": float,
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _build_args_schema(
    tool_name: str,
    parameters: list[dict[str, Any]],
) -> type[BaseModel]:
    """Build a pydantic model from MCP tool parameters.

    Args:
        tool_name: Name of the tool (used for model naming).
        parameters: Parameter definitions from the catalog.

    Returns:
        A dynamically created pydantic BaseModel class.
    """
    fields: dict[str, Any] = {}

    for param in parameters:
        name = param["name"]
        py_type = _TYPE_MAP.get(param.get("type", "string"), str)
        required = param.get("required", False)
        description = param.get("description", "")

        if required:
            fields[name] = (py_type, Field(description=description))
        else:
            default = param.get("default")
            fields[name] = (py_type | None, Field(default=default, description=description))

    # If no parameters, return an empty model
    if not fields:
        return create_model(f"{tool_name}_Args")

    model_name = f"{tool_name}_Args"
    return create_model(model_name, **fields)


class MCPToolWrapper(BaseTool):
    """Wraps an MCP tool as a LangChain BaseTool for use with react agents."""

    tool_server: str
    """MCP server name hosting this tool."""

    mcp_client: MCPClient
    """Client for calling the MCP server."""

    acting_user: str | None = None
    """ACCESS ID of the acting user."""

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async _arun")

    async def _arun(self, **kwargs: Any) -> str:
        """Call the MCP tool and return the result as a JSON string."""
        # Filter out None values — MCP servers don't expect them
        arguments = {k: v for k, v in kwargs.items() if v is not None}

        logger.info(f"Domain agent calling {self.tool_server}/{self.name} with {arguments}")

        result = await self.mcp_client.call_tool(
            server=self.tool_server,
            tool_name=self.name,
            arguments=arguments,
            acting_user=self.acting_user,
        )

        record_tool_timing(self.name, result.duration_ms)

        if result.success:
            return json.dumps(result.data, default=str)
        return json.dumps({"error": result.error})


def create_domain_tools(
    config: DomainAgentConfig,
    tool_catalog: dict[str, Any],
    acting_user: str | None = None,
) -> list[BaseTool]:
    """Create LangChain tool wrappers for a domain's MCP tools.

    Filters the full tool catalog to only include tools from the
    domain's configured MCP servers.

    Args:
        config: Domain agent config specifying which servers to include.
        tool_catalog: The full MCP tool catalog.
        acting_user: ACCESS ID for auth headers.

    Returns:
        List of BaseTool instances ready for create_react_agent.
    """
    client = MCPClient()
    tools: list[BaseTool] = []

    servers = tool_catalog.get("servers", [])
    for server_info in servers:
        server_name = server_info.get("server", "")
        if server_name not in config.mcp_servers:
            continue

        if server_info.get("status") != "available":
            logger.warning(f"Domain {config.name}: server {server_name} unavailable, skipping")
            continue

        for tool_def in server_info.get("tools", []):
            tool_name = tool_def.get("name", "")
            description = tool_def.get("description", "")
            parameters = tool_def.get("parameters", [])

            args_schema = _build_args_schema(tool_name, parameters)

            wrapper = MCPToolWrapper(
                name=tool_name,
                description=description,
                args_schema=args_schema,
                tool_server=server_name,
                mcp_client=client,
                acting_user=acting_user,
            )
            tools.append(wrapper)

    logger.info(
        f"Created {len(tools)} tools for domain '{config.name}' from servers {config.mcp_servers}"
    )
    return tools


def create_mcp_tools_from_catalog(
    tool_catalog: dict[str, Any],
    acting_user: str | None = None,
) -> list[BaseTool]:
    """Create LangChain tool wrappers for every available tool in the catalog.

    Unlike create_domain_tools, this function is not scoped to a single
    domain — it materializes every enabled MCP tool so the tool_calling_loop
    can pick from the full set. Respects the capability registry via the
    tool_catalog input (caller populates the catalog from enabled
    capabilities only).

    Args:
        tool_catalog: The MCP tool catalog dict with "servers" key.
        acting_user: ACCESS ID for auth headers, or None for anonymous.

    Returns:
        List of MCPToolWrapper instances, one per available tool in the catalog.
    """
    client = MCPClient()
    tools: list[BaseTool] = []

    servers = tool_catalog.get("servers", [])
    for server_info in servers:
        server_name = server_info.get("server", "")
        if not server_name:
            continue
        # Treat missing status as "available" — older catalogs may omit it,
        # and we only want to skip when status is explicitly something else.
        status = server_info.get("status", "available")
        if status != "available":
            logger.warning(
                f"Server {server_name} unavailable (status={status}), "
                f"skipping for tool_calling_loop"
            )
            continue

        for tool_def in server_info.get("tools", []):
            tool_name = tool_def.get("name", "")
            description = tool_def.get("description", "")
            parameters = tool_def.get("parameters", [])

            args_schema = _build_args_schema(tool_name, parameters)

            wrapper = MCPToolWrapper(
                name=tool_name,
                description=description,
                args_schema=args_schema,
                tool_server=server_name,
                mcp_client=client,
                acting_user=acting_user,
            )
            tools.append(wrapper)

    logger.info(
        f"Created {len(tools)} tools for tool_calling_loop from catalog "
        f"({len(servers)} servers in catalog)"
    )
    return tools
