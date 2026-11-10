"""MCP tool wrappers for domain agents.

Wraps MCP tools as LangChain BaseTool instances so they can be used
with create_react_agent's function-calling loop.
"""

import asyncio
import json
import logging
import time
from typing import Annotated, Any, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, BeforeValidator, Field, WithJsonSchema, create_model

from ...tool_attachments import NOT_ATTACHED_NOTE, externalize
from ...tools.mcp_client import MCPClient
from ..auth_tools import ToolIdentityClass, tool_identity_class
from ..middleware.requires_auth import CoarseReason, refusal_artifact, refusal_text
from ..middleware.turn_budget import TIMED_OUT_ARTIFACT
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


def _parse_json_text(expected: type) -> BeforeValidator:
    """Accept an object or array the model sent as JSON text.

    Some models send ``'{"resource": "ACES"}'`` instead of the object. Rejecting
    that led the model to retry without the filter and answer for the wrong
    scope. Text that isn't the expected shape is left for validation to reject.
    """

    def parse(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return parsed if isinstance(parsed, expected) else value

    return BeforeValidator(parse)


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
        py_type: Any = _TYPE_MAP.get(param.get("type", "string"), str)
        if py_type in (dict, list):
            py_type = Annotated[py_type, _parse_json_text(py_type)]
        declared = param.get("json_schema")
        if declared:
            # Only the schema shown to the model changes. Validation stays on
            # the plain Python type, so a value the server accepts but doesn't
            # list (an enum alias) isn't rejected here.
            py_type = Annotated[py_type, WithJsonSchema(declared)]
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

    attachment_thread_id: str | None = None
    """The full-screen thread a large attachment is stored under, so it isn't
    checkpointed with every message (src/tool_attachments.py). None keeps it
    on the message."""

    # The artifact carries a refusal's reason, as the identity gate's does, or
    # marks a timeout.
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"

    class Config:
        arbitrary_types_allowed = True

    @property
    def call_timeout(self) -> float:
        """The timeout a call to this tool gets, in seconds."""
        return self.mcp_client.timeout_for(self.tool_server, self.name)

    def _run(self, **kwargs: Any) -> tuple[str, Any]:
        raise NotImplementedError("Use async _arun")

    async def _arun(self, **kwargs: Any) -> tuple[str, Any]:
        """Call the MCP tool; return its result as a JSON string, plus any structured content as the artifact.

        Until the servers require the acting-user token (D8-2946, rollout step
        3), they trust ``X-Acting-User``, so the agent is the only place a write
        is stopped. The identity gate refuses first; this checks again
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

        started = time.monotonic()
        try:
            result = await self.mcp_client.call_tool(
                server=self.tool_server,
                tool_name=self.name,
                arguments=arguments,
                acting_user=self.acting_user,
                # Only a full-screen thread's client reads structured content,
                # and only there can a large attachment be stored.
                include_structured=self.attachment_thread_id is not None,
            )
        except asyncio.CancelledError:
            # Cut off at the turn's deadline. It still ran, so it still counts.
            record_tool_timing(self.name, (time.monotonic() - started) * 1000)
            raise

        record_tool_timing(self.name, result.duration_ms)

        if result.success:
            artifact = None
            data = result.data
            if result.structured:
                structured, dropped = await asyncio.to_thread(
                    externalize, result.structured, self.attachment_thread_id
                )
                if dropped:
                    data = (
                        f"{data}\n\n{NOT_ATTACHED_NOTE}"
                        if isinstance(data, str)
                        else {"result": data, "note": NOT_ATTACHED_NOTE}
                    )
                artifact = {"structured_content": structured} if structured else None
            return json.dumps(data, default=str), artifact
        return json.dumps({"error": result.error}), TIMED_OUT_ARTIFACT if result.timed_out else None


def create_mcp_tools_from_catalog(
    tool_catalog: dict[str, Any],
    acting_user: str | None = None,
    *,
    can_write: bool = False,
    attachment_thread_id: str | None = None,
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
        attachment_thread_id: The full-screen thread to store large
            attachments under, or None to keep them on the message.

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
                attachment_thread_id=attachment_thread_id,
            )
            tools.append(wrapper)

    logger.info(
        f"Created {len(tools)} tools for tool_calling_loop from catalog "
        f"({len(servers)} servers in catalog)"
    )
    return tools
