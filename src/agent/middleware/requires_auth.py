"""Middleware: refuse write/private-read tool calls for an anonymous caller.

When an ANONYMOUS caller's turn tries to invoke a write or private-read tool,
letting the call reach the MCP layer produces a raw error the model has to
narrate badly. Instead this middleware refuses the call with a ``ToolMessage``
whose prose tells the model what happened and what to tell the user, and
whose ``artifact`` carries the coarse reason for code. The artifact is never
sent to the model.

The gate also keeps the turn's own record of refusals, which the loop node
returns as ``tool_refusals`` so the routes can prompt the caller to sign in.

Name-collision note: ``CapabilityConfig.requires_auth: bool`` (domains/config.py)
is a dormant, unrelated dataclass field — different namespace, do not conflate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from ...config import settings
from ..auth_tools import ToolIdentityClass, tool_identity_class

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
    from langgraph.types import Command

    from ..state import ToolRefusal

# Coarse reasons only — never resource- or request-specific. A write attempt
# and a private-data read are different asks ("sign in to do that" vs "sign
# in to see that"), so the two strings are kept distinct for client copy.
CoarseReason = Literal["write_action", "private_data"]


def _coarse_reason(identity_class: ToolIdentityClass) -> CoarseReason:
    return "write_action" if identity_class is ToolIdentityClass.WRITE else "private_data"


def refusal_text(reason: CoarseReason) -> str:
    """The refused call's result, written for the model.

    It stays in the thread after the user signs in, so it holds only until
    then rather than for a turn whose edges the model can't see, and it never
    names the user or the request.
    """
    if reason == "write_action":
        need = "Making this change needs"
        avoid = (
            "Until they sign in, don't call tools that make changes, "
            "and don't say the change happened."
        )
    else:
        need = "Seeing the user's own information needs"
        avoid = (
            "Until they sign in, don't call tools that read the user's own information, "
            "and don't describe information you didn't get."
        )
    return (
        f"Not done. {need} the user to sign in with their ACCESS account at "
        f"{settings.LOGIN_URL}. Tell them so. {avoid} Help with the rest of the request."
    )


def _tool_name(tool: Any) -> str | None:
    return tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)


class RequiresAuthMiddleware(AgentMiddleware):
    """Refuse write/private-read tool calls for an anonymous caller.

    ``acting_user`` is captured by closure at construction time (it lives on
    the OUTER graph's state, not the inner ``create_agent`` state the react
    loop sees). The loop node builds one instance per turn, so ``refusals``
    is this turn's record.

    Once a call has been refused, every tool this caller would be refused is
    left out of the rest of the turn's model calls. The model can't retry,
    and it can still use public tools to answer the rest of the request. A
    call that names a hidden tool anyway is still refused here.
    """

    def __init__(self, acting_user: str | None) -> None:
        super().__init__()
        self._acting_user = acting_user
        self.refusals: list[ToolRefusal] = []

    def _refusal_reason(self, name: str | None) -> CoarseReason | None:
        if self._acting_user is not None or not name:
            return None
        identity_class = tool_identity_class(name)
        if identity_class is ToolIdentityClass.NONE:
            return None
        return _coarse_reason(identity_class)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call["name"]
        reason = self._refusal_reason(name)
        if reason is None:
            return await handler(request)
        self.refusals.append({"tool_name": name, "reason": reason})
        return ToolMessage(
            content=refusal_text(reason),
            name=name,
            tool_call_id=request.tool_call["id"],
            status="error",
            artifact={"refusal": {"reason": reason}},
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if self.refusals:
            allowed = [t for t in request.tools if self._refusal_reason(_tool_name(t)) is None]
            request = request.override(tools=allowed)
        return await handler(request)
