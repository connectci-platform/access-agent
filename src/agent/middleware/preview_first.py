"""Middleware: a registration goes through only after the user answers its preview.

``register_for_event`` previews without ``confirmed`` and registers with it, and
the model chooses which. Left to the model, it can register someone who never
saw the preview, so a confirmed call is sent as a preview unless this event has
already been previewed and the user has replied since.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import TypeAdapter, ValidationError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langgraph.prebuilt.tool_node import ToolCallRequest
    from langgraph.types import Command

TOOL_NAME = "register_for_event"
EVENT_ARG = "eventinstance_id"


_BOOL = TypeAdapter(bool)


def _is_confirmed(value: Any) -> bool:
    """Whether the tool's validation would read this as true.

    Models can send "true", which validation only turns into True after this
    middleware has run, so ``is True`` would let it through.
    """
    try:
        return _BOOL.validate_python(value)
    except ValidationError:
        return False


def _is_preview(message: ToolMessage) -> bool:
    try:
        result = json.loads(str(message.content))
    except json.JSONDecodeError:
        return False
    return isinstance(result, dict) and result.get("status") == "preview"


class PreviewFirstMiddleware(AgentMiddleware):
    """Send a confirmed registration as a preview until the user has answered one."""

    @staticmethod
    def _messages(request: Any) -> list[Any]:
        return list(request.state.get("messages", []))

    @staticmethod
    def _may_confirm(messages: list[Any], event_id: Any) -> bool:
        """Whether the user has replied since this event's latest preview.

        Reads the thread in order, because call ids repeat across a thread: a
        result belongs to the nearest earlier call with its id. A preview is
        recognized by its result, not its arguments, since a downgraded call
        keeps ``confirmed: true`` in the stored message. Any other result for
        this event uses the preview up, so each write needs a fresh one.
        """
        calls: dict[str | None, Any] = {}
        previewed = False
        answered = False
        for message in messages:
            if isinstance(message, AIMessage):
                calls.update({call["id"]: call for call in message.tool_calls})
            elif isinstance(message, HumanMessage):
                answered = previewed
            elif isinstance(message, ToolMessage):
                call = calls.get(message.tool_call_id)
                if (
                    call
                    and call["name"] == TOOL_NAME
                    and str(call["args"].get(EVENT_ARG)) == str(event_id)
                ):
                    previewed = _is_preview(message)
                    answered = False
        return previewed and answered

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        args = call["args"]
        if (
            call["name"] == TOOL_NAME
            and _is_confirmed(args.get("confirmed", False))
            and not self._may_confirm(self._messages(request), args.get(EVENT_ARG))
        ):
            request = request.override(tool_call={**call, "args": {**args, "confirmed": False}})
        return await handler(request)
