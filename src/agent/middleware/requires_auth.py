"""Middleware: refuse tool calls the caller may not make.

An anonymous caller may not call write or private-read tools, and a signed-in
caller whose token lacks ``agent:write`` may not call write tools. Letting such
a call reach the MCP layer produces a raw error the model has to narrate
badly. Instead this middleware refuses the call with a ``ToolMessage``
whose prose tells the model what happened and what to tell the user, and
whose ``artifact`` carries the coarse reason for code. The artifact is never
sent to the model.

The gate also keeps the turn's own record of refusals, which the loop node
returns as ``tool_refusals`` so the routes can annotate the turn: a sign-in
prompt for an anonymous caller, and no prompt for ``write_scope``.
The caller comes from the run's ``AgentContext``, which the inner
``create_agent`` inherits as ``request.runtime.context``.

Name-collision note: ``CapabilityConfig.requires_auth: bool`` (domains/config.py)
is a dormant, unrelated dataclass field — different namespace, do not conflate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from ...config import settings
from ..auth_tools import ToolIdentityClass, tool_identity_class
from ..context import AgentContext, require_context

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
    from langgraph.types import Command

    from ..state import ToolRefusal

# Coarse reasons only — never resource- or request-specific. A write attempt
# and a private-data read are different asks ("sign in to do that" vs "sign
# in to see that"), so the strings are kept distinct for client copy.
# write_scope is a signed-in caller whose token can't make changes; signing in
# again doesn't help them, so clients offer no sign-in for it.
CoarseReason = Literal["write_action", "private_data", "write_scope"]


def refusal_reason(context: AgentContext, name: str | None) -> CoarseReason | None:
    """Why this caller may not call tool ``name``, or None when they may.

    The one rule behind both refusing a call and hiding tools after a refusal.
    """
    if not name:
        return None
    identity_class = tool_identity_class(name)
    if identity_class is ToolIdentityClass.NONE:
        return None
    if context.acting_user is None:
        return "write_action" if identity_class is ToolIdentityClass.WRITE else "private_data"
    if identity_class is ToolIdentityClass.WRITE and not context.can_write:
        return "write_scope"
    return None


def refusal_artifact(reason: CoarseReason) -> dict[str, Any]:
    """The refused call's artifact: read by code, never sent to the model."""
    return {"refusal": {"reason": reason}}


def refusal_text(reason: CoarseReason) -> str:
    """The refused call's result, written for the model.

    It stays in the thread after the user signs in, so it holds only until
    then rather than for a turn whose edges the model can't see, and it never
    names the user or the request. ``write_scope`` can't say "until they sign
    in", because the user already has; it is about this chat instead.
    """
    if reason == "write_scope":
        return (
            "Not done. This chat can read information for the user but can't make "
            "changes on their behalf. Don't call tools that make changes in this "
            "chat, and don't say the change happened. Tell the user the change "
            "can't be made from here, without suggesting they sign in. For a "
            "support request, point them to "
            "https://support.access-ci.org/open-a-ticket. Help with the rest of "
            "the request."
        )
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
    """Refuse tool calls the caller may not make (see ``refusal_reason``).

    The caller is read from the run's context at call time, never captured
    when the gate is built. The loop node builds one instance per turn, so
    ``refusals`` is this turn's record.

    Once a call has been refused, every tool this caller would be refused is
    left out of the rest of the turn's model calls. The model can't retry,
    and it can still use public tools to answer the rest of the request. A
    call that names a hidden tool anyway is still refused here.
    """

    def __init__(self) -> None:
        super().__init__()
        self.refusals: list[ToolRefusal] = []

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        name = request.tool_call["name"]
        reason = refusal_reason(require_context(request.runtime.context), name)
        if reason is None:
            return await handler(request)
        self.refusals.append({"tool_name": name, "reason": reason})
        return ToolMessage(
            content=refusal_text(reason),
            name=name,
            tool_call_id=request.tool_call["id"],
            status="error",
            artifact=refusal_artifact(reason),
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if self.refusals:
            context = require_context(request.runtime.context)
            allowed = [t for t in request.tools if refusal_reason(context, _tool_name(t)) is None]
            request = request.override(tools=allowed)
        return await handler(request)
