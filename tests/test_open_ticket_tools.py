"""Anyone may file a support ticket, signed in or not (D8-3069).

Someone who can't sign in is who most needs a login ticket. The ticket server
files a ticket without a verified sign-in as unverified, the same as the
public ticket form, so the identity gate lets the three ticket tools through
while still refusing every other write.
"""

import pytest

from src.agent.auth_tools import OPEN_TICKET_TOOLS
from src.agent.middleware.requires_auth import refusal_reason
from tests.agent_runtime import context_for

ANON = context_for(None)
READ_ONLY_TOKEN = context_for("user123", can_write=False)
TICKET_TOOLS = ["create_support_ticket", "create_login_ticket", "report_security_incident"]


def test_the_open_tools_are_exactly_the_ticket_tools():
    assert sorted(OPEN_TICKET_TOOLS) == sorted(TICKET_TOOLS)


@pytest.mark.parametrize("tool", TICKET_TOOLS)
@pytest.mark.parametrize("context", [ANON, READ_ONLY_TOKEN], ids=["anonymous", "read-only"])
def test_anyone_may_file_a_ticket(context, tool):
    assert refusal_reason(context, tool) is None


def test_other_writes_are_still_refused():
    assert refusal_reason(ANON, "register_for_event") == "write_action"
    assert refusal_reason(ANON, "create_event") == "write_action"
    assert refusal_reason(READ_ONLY_TOKEN, "create_announcement") == "write_scope"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", TICKET_TOOLS)
async def test_the_wrapper_lets_an_anonymous_ticket_through(tool):
    from src.agent.domains.tools import MCPToolWrapper
    from src.tools.mcp_client import MCPClient, MCPToolResult

    class FakeClient(MCPClient):
        def __init__(self) -> None:
            super().__init__()
            self.sent: list[str] = []

        async def call_tool(self, server, tool_name, arguments, **kwargs):
            self.sent.append(tool_name)
            return MCPToolResult(success=True, data={"success": True})

    client = FakeClient()
    wrapper = MCPToolWrapper(
        name=tool,
        description="file a ticket",
        tool_server="jsm",
        mcp_client=client,
        acting_user=None,
        can_write=False,
    )

    await wrapper._arun(summary="Locked out", description="Can't sign in")

    assert client.sent == [tool]


@pytest.mark.asyncio
async def test_the_wrapper_still_refuses_other_anonymous_writes():
    from src.agent.domains.tools import MCPToolWrapper
    from src.tools.mcp_client import MCPClient

    wrapper = MCPToolWrapper(
        name="register_for_event",
        description="register",
        tool_server="events",
        mcp_client=MCPClient(),
        acting_user=None,
        can_write=False,
    )

    _, artifact = await wrapper._arun(eventinstance_id="1")

    assert artifact == {"refusal": {"reason": "write_action"}}


@pytest.mark.asyncio
async def test_the_wrapper_opens_ticket_tools_only_on_the_ticket_server():
    from src.agent.domains.tools import MCPToolWrapper
    from src.tools.mcp_client import MCPClient

    wrapper = MCPToolWrapper(
        name="create_support_ticket",
        description="a tool on another server that shares the name",
        tool_server="events",
        mcp_client=MCPClient(),
        acting_user=None,
        can_write=False,
    )

    _, artifact = await wrapper._arun(summary="x", description="y")

    assert artifact == {"refusal": {"reason": "write_action"}}
