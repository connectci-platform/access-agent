"""Middleware: short-circuit anon write/private-read tool calls into a signal.

When an ANONYMOUS caller's turn tries to invoke a write or private-read tool,
letting the call reach the MCP layer produces a raw error the model has to
narrate badly. Instead this middleware suppresses the call and injects a
sentinel ``ToolMessage`` the loop node scans for after ``agent.ainvoke``
returns (see ``tool_calling_loop.py``'s post-invoke scan), so the caller can
be prompted to sign in via a structured ``requires_auth`` SSE event.

Name-collision note: ``CapabilityConfig.requires_auth: bool`` (domains/config.py)
is a dormant, unrelated dataclass field — different namespace, do not conflate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import ToolMessage

from ..auth_tools import ToolIdentityClass, tool_identity_class

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ToolCallRequest
    from langchain_core.messages import BaseMessage
    from langgraph.types import Command

# Content marker for the suppressed-call ToolMessage. The loop node scans
# result_messages for this exact string (see tool_calling_loop.py); it must
# never be shown to the user, so it is overwritten/dropped before any
# user-facing rendering, same discipline as the empty-answer sentinel.
REQUIRES_AUTH_SENTINEL = "__requires_auth__"

# Coarse reasons only — never resource- or request-specific. A write attempt
# and a private-data read are different asks ("sign in to do that" vs "sign
# in to see that"), so the two strings are kept distinct for client copy.
CoarseReason = Literal["write_action", "private_data"]


def _coarse_reason(identity_class: ToolIdentityClass) -> CoarseReason:
    return "write_action" if identity_class is ToolIdentityClass.WRITE else "private_data"


class RequiresAuthMiddleware(AgentMiddleware):
    """Suppress write/private-read tool calls for an anonymous caller.

    ``acting_user`` is captured by closure at construction time (it lives on
    the OUTER graph's state, not the inner ``create_agent`` state the react
    loop sees — see ``tool_calling_loop_node``, which builds this middleware
    with ``acting_user`` in scope).

    Suppressing a call here does NOT halt the react loop — it only replaces
    that one tool's result with the sentinel. The model may call another
    tool or answer directly afterward; the post-invoke scan in the loop node
    catches the sentinel regardless of what happens next.
    """

    def __init__(self, acting_user: str | None) -> None:
        super().__init__()
        self._acting_user = acting_user

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        if self._acting_user is None:
            name = request.tool_call["name"]
            identity_class = tool_identity_class(name)
            if identity_class is not ToolIdentityClass.NONE:
                reason = _coarse_reason(identity_class)
                return ToolMessage(
                    content=f"{REQUIRES_AUTH_SENTINEL}:{reason}",
                    name=name,
                    tool_call_id=request.tool_call["id"],
                    status="error",
                )
        return await handler(request)


def find_requires_auth_reason(messages: list[BaseMessage]) -> CoarseReason | None:
    """Scan a result-message list for the sentinel; return its coarse reason.

    Mirrors the loop node's other post-invoke message scans (e.g. the
    final-answer scan in ``tool_calling_loop_node``). Returns the FIRST
    sentinel found — one anon-gated call is enough to require sign-in for
    the whole turn, and a single login prompt is all the client can show.
    """
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else None
        if content and content.startswith(REQUIRES_AUTH_SENTINEL):
            _, _, reason = content.partition(":")
            if reason in ("write_action", "private_data"):
                return reason  # type: ignore[return-value]
    return None
