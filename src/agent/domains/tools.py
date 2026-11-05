"""MCP tool wrappers for domain agents.

Wraps MCP tools as LangChain BaseTool instances so they can be used
with create_react_agent's function-calling loop.
"""

import json
import logging
from typing import Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from ...tools.mcp_client import MCPClient
from ..auth_tools import ToolIdentityClass, tool_identity_class
from ..middleware.requires_auth import CoarseReason, refusal_artifact, refusal_text
from ..turn_capture import record_tool_timing

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

    can_write: bool = False
    """Whether this run may make changes. Off unless the caller's context says so."""

    read_only: bool | None = None
    """The server's readOnlyHint for this tool, or None when it published none."""

    # The artifact carries a refusal's reason, as the identity gate's does.
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> tuple[str, Any]:
        raise NotImplementedError("Use async _arun")

    async def _arun(self, **kwargs: Any) -> tuple[str, Any]:
        """Call the MCP tool; return its result as a JSON string, with no artifact.

        The MCP servers trust ``X-Acting-User``, so the agent is the only place
        a write is stopped. The identity gate refuses first; this checks again
        at the one path to ``call_tool``, against the same rule, so a context
        plumbing mistake can't let a write through. It returns a refusal rather
        than raising, because a tool that raises aborts the whole turn.
        """
        is_write = tool_identity_class(self.name, self.read_only) is ToolIdentityClass.WRITE
        if is_write and not self.can_write:
            logger.error(
                "write tool %s reached the MCP wrapper without write permission; refused",
                self.name,
            )
            reason: CoarseReason = "write_scope" if self.acting_user else "write_action"
            return refusal_text(reason), refusal_artifact(reason)

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
            return json.dumps(result.data, default=str), None
        return json.dumps({"error": result.error}), None


def create_mcp_tools_from_catalog(
    tool_catalog: dict[str, Any],
    acting_user: str | None = None,
    *,
    can_write: bool = False,
) -> list[BaseTool]:
    """Create LangChain tool wrappers for every available tool in the catalog.

    It materializes every enabled MCP tool so the tool_calling_loop can pick
    from the full set. Respects the capability registry via the
    tool_catalog input (caller populates the catalog from enabled
    capabilities only).

    Args:
        tool_catalog: The MCP tool catalog dict with "servers" key.
        acting_user: ACCESS ID for auth headers, or None for anonymous.
        can_write: Whether write tools may reach the MCP server this run.

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
                can_write=can_write,
                read_only=tool_def.get("read_only"),
            )
            tools.append(wrapper)

    logger.info(
        f"Created {len(tools)} tools for tool_calling_loop from catalog "
        f"({len(servers)} servers in catalog)"
    )
    return tools
