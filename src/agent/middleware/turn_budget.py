"""Middleware: answer from what the turn has before its time limit.

The routes stop a turn at ``AGENT_TURN_TIMEOUT_S`` and throw it away, good
results included. So tools stop ``AGENT_ANSWER_RESERVE_S`` before that: a call
still running is cut off unless it makes a change, a later call is not run, and
the model gets no tools and has to answer. The model isn't asked to watch the time. It can't see it, and
for this model a tool result outweighs prompt text.

A call that timed out isn't run again with the same arguments this turn, since
a retry costs a full timeout and fails the same way. That record lives here
rather than in the messages, which compaction can rewrite. One instance per turn.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from ..auth_tools import ToolIdentityClass, tool_identity_class
from ..turn_capture import record_tool_timing

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse, ToolCallRequest
    from langgraph.types import Command

# On a tool result whose call timed out, so code needn't read the error text.
TIMED_OUT_ARTIFACT = {"timed_out": True}

# These stay in the thread, so they speak only of this request. The next one can
# try the same call.
PAST_DEADLINE_TEXT = (
    "Not run. This request is out of time for tool calls. Answer now from the tool "
    "results you already have, and tell the user what you couldn't get."
)
CUT_OFF_TEXT = (
    "Stopped. This call was still running when this request ran out of time, so "
    "there is no result. Answer now from the tool results you already have, and tell "
    "the user what you couldn't get. Don't draw a conclusion about anything you "
    "couldn't check."
)
REPEAT_TEXT = (
    "Not run. This same call already timed out during this request, and repeating "
    "it won't help. Don't call it again with these arguments. Use the other results "
    "you have, try a narrower query, or tell the user this part couldn't be retrieved."
)
ANSWER_NOW_NOTE = (
    "(This request is out of time, so no more tools are available. Answer now from "
    "the tool results you already have, and tell the user plainly what you couldn't "
    "get. Your searches may be incomplete, so never say something doesn't exist or "
    "that there are none. Say what you found and that there may be more.)"
)


def _not_run(call: Any, text: str, reason: str) -> ToolMessage:
    """A call that never ran, recorded like a refusal so it isn't counted as a tool run."""
    return ToolMessage(
        content=text,
        name=call["name"],
        tool_call_id=call["id"],
        status="error",
        artifact={"refusal": {"reason": reason}},
    )


def _timed_out(result: Any) -> bool:
    return isinstance(result, ToolMessage) and result.artifact == TIMED_OUT_ARTIFACT


def _with_answer_note(messages: list[Any]) -> list[Any]:
    """The model's view of the thread once time is up. The stored thread doesn't change."""
    last = messages[-1]
    if isinstance(last, ToolMessage) and isinstance(last.content, str):
        noted = last.model_copy(update={"content": f"{last.content}\n\n{ANSWER_NOW_NOTE}"})
        return [*messages[:-1], noted]
    return [*messages, HumanMessage(content=ANSWER_NOW_NOTE)]


class TurnBudgetMiddleware(AgentMiddleware):
    """Stop tools at ``tool_deadline`` (``time.monotonic()``) and block timed-out repeats."""

    def __init__(self, tool_deadline: float) -> None:
        super().__init__()
        self.tool_deadline = tool_deadline
        self.deadline_hit = False
        self._timed_out_calls: set[str] = set()

    @staticmethod
    def _call_key(call: Any, tool: Any) -> str:
        """What the tool would receive, not what the model wrote.

        Models send one value several ways ("true", JSON text for an object, a
        null for an unset field), and call ids repeat. The tool's schema reduces
        those forms to one. Arguments it would reject are compared as sent.
        """
        args = call["args"]
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            with suppress(ValidationError):
                args = schema.model_validate(args).model_dump(mode="json", exclude_none=True)
        return json.dumps([call["name"], args], sort_keys=True, default=str)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        started = time.monotonic()
        remaining = self.tool_deadline - started
        if remaining <= 0:
            return _not_run(call, PAST_DEADLINE_TEXT, "turn_deadline")
        key = self._call_key(call, request.tool)
        if key in self._timed_out_calls:
            return _not_run(call, REPEAT_TEXT, "repeat_timeout")
        # A change is left to finish: cutting it off would leave it unknown whether
        # it happened. Its own tool timeout still bounds it.
        read_only = getattr(request.tool, "read_only", None)
        if tool_identity_class(call["name"], read_only) is ToolIdentityClass.WRITE:
            result = await handler(request)
            if _timed_out(result):
                self._timed_out_calls.add(key)
            return result
        try:
            result = await asyncio.wait_for(handler(request), remaining)
        except TimeoutError:
            self._timed_out_calls.add(key)
            # The tool records its own timing when it finishes, and this one didn't.
            record_tool_timing(call["name"], (time.monotonic() - started) * 1000)
            return ToolMessage(
                content=CUT_OFF_TEXT,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
                artifact=TIMED_OUT_ARTIFACT,
            )
        if _timed_out(result):
            self._timed_out_calls.add(key)
        return result

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        if time.monotonic() >= self.tool_deadline:
            self.deadline_hit = True
            request = request.override(tools=[], messages=_with_answer_note(request.messages))
        return await handler(request)
