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
from langchain_core.messages import HumanMessage, ToolMessage

from ...config import settings
from ..auth_tools import OPEN_TICKET_TOOLS, ToolIdentityClass, tool_identity_class
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


def refusal_reason(
    context: AgentContext, name: str | None, read_only: bool | None = None
) -> CoarseReason | None:
    """Why this caller may not call tool ``name``, or None when they may.

    The one rule behind both refusing a call and hiding tools after a refusal.
    ``read_only`` is the tool server's readOnlyHint, when there is one; see
    ``tool_identity_class`` for how it is used.
    """
    if not name or name in OPEN_TICKET_TOOLS:
        return None
    identity_class = tool_identity_class(name, read_only)
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


SIGNED_IN_SINCE_TEXT = (
    "This call was refused because the user wasn't signed in. The user has signed "
    "in since, so if they ask again, call the tool again."
)
SIGNED_IN_NOTE = "(The user is signed in now. Earlier sign-in requests no longer apply.)"


def _cleared_by_sign_in(message: Any, context: AgentContext) -> bool:
    """Whether this refusal no longer applies to this signed-in caller."""
    if not isinstance(message, ToolMessage) or not isinstance(message.artifact, dict):
        return False
    refusal = message.artifact.get("refusal")
    reason = refusal.get("reason") if isinstance(refusal, dict) else None
    return reason == "private_data" or (reason == "write_action" and context.can_write)


def signed_in_since(messages: list[Any], context: AgentContext) -> list[Any]:
    """The model's view of a thread whose user signed in after a refusal.

    A refusal stays in the thread, and the model can repeat it instead of
    calling the tool again. For a signed-in caller the model sees it rewritten,
    with a note on the latest request. The stored thread doesn't change.
    """
    if context.acting_user is None or not any(_cleared_by_sign_in(m, context) for m in messages):
        return messages
    view = [
        m.model_copy(update={"content": SIGNED_IN_SINCE_TEXT})
        if _cleared_by_sign_in(m, context)
        else m
        for m in messages
    ]
    last = len(view) - 1
    while last >= 0 and not isinstance(view[last], HumanMessage):
        last -= 1
    refused_before = any(_cleared_by_sign_in(m, context) for m in messages[: max(last, 0)])
    if refused_before and isinstance(view[last].content, str):
        note = f"{view[last].content}\n\n{SIGNED_IN_NOTE}"
        view[last] = view[last].model_copy(update={"content": note})
    return view


def _tool_name(tool: Any) -> str | None:
    return tool.get("name") if isinstance(tool, dict) else getattr(tool, "name", None)


def _read_only(tool: Any) -> bool | None:
    """The tool's readOnlyHint as the catalog carried it; None when unknown."""
    hint = tool.get("read_only") if isinstance(tool, dict) else getattr(tool, "read_only", None)
    return hint if isinstance(hint, bool) else None


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
        # request.tool is the registered tool, or None for a name the agent
        # doesn't have; then only the name rules apply.
        reason = refusal_reason(
            require_context(request.runtime.context), name, _read_only(request.tool)
        )
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
        context = require_context(request.runtime.context)
        request = request.override(messages=signed_in_since(request.messages, context))
        if self.refusals:
            allowed = [
                t
                for t in request.tools
                if refusal_reason(context, _tool_name(t), _read_only(t)) is None
            ]
            request = request.override(tools=allowed)
        return await handler(request)
