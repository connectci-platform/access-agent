"""Middleware: answer from what the turn has before its time limit.

The routes stop a turn at ``AGENT_TURN_TIMEOUT_S`` and throw it away, good
results included. So tools stop ``AGENT_ANSWER_RESERVE_S`` before that: a call
still running is cut off unless it makes a change, a later call is not run, and
the model gets no tools and has to answer. A model call that wouldn't end before
that point loses its tools too, since a call it asked for would only be refused
and cost a second model call. The model isn't asked to watch the time. It can't
see it, and for this model a tool result outweighs prompt text.

A call that timed out isn't run again with the same arguments this turn, since
a retry costs a full timeout and fails the same way. That record lives here
rather than in the messages, which compaction can rewrite. One instance per turn.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, ValidationError

from ..auth_tools import ToolIdentityClass, tool_identity_class

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ToolCallRequest
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
WRITE_TIMED_OUT_TEXT = (
    "Outcome unknown. This change timed out before the server replied, so it may or "
    "may not have gone through. Don't try it again during this request. Tell the user "
    "you can't confirm it either way, and to check before trying again."
)
WRITE_REPEAT_TEXT = (
    "Not run again. This same change timed out earlier in this request, so it may or "
    "may not have gone through. Don't try it again during this request. Tell the user "
    "you can't confirm it either way, and to check before trying again."
)
ANSWER_NOW_NOTE = (
    "(This request is out of time, so no more tools are available. Answer now from "
    "the tool results you already have, and tell the user plainly what you couldn't "
    "get. Your searches may be incomplete, so never say something doesn't exist or "
    "that there are none. Say what you found and that there may be more. This is your "
    "final message, so don't say you'll search or do anything next.)"
)
# The turn's answer when the model's last response held no text for the user.
OUT_OF_TIME_ANSWER = (
    "I ran out of time before I could finish answering. Please try again, or ask about "
    "a smaller part of your question."
)

# With tools=[] vLLM has nothing to parse a call against, so it can arrive as text.
_RAW_TOOL_CALL_RE = re.compile(r"<tool_call>.*?(?:</tool_call>|\Z)", re.DOTALL)


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
        # Set when the model left no text and the turn's answer is OUT_OF_TIME_ANSWER.
        self.gave_up = False
        self._longest_model_call_s = 0.0
        self._timed_out_calls: set[str] = set()

    def out_of_time(self) -> bool:
        """Whether a model call starting now is expected to end past the deadline."""
        return time.monotonic() + self._longest_model_call_s >= self.tool_deadline

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
                args = schema.model_validate(args).model_dump(
                    mode="json", exclude_none=True, exclude_defaults=True
                )
        return json.dumps([call["name"], args], sort_keys=True, default=str)

    async def _run_read(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
        remaining: float,
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        try:
            return await asyncio.wait_for(handler(request), remaining)
        except TimeoutError:
            return ToolMessage(
                content=CUT_OFF_TEXT,
                name=call["name"],
                tool_call_id=call["id"],
                status="error",
                artifact=TIMED_OUT_ARTIFACT,
            )

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        call = request.tool_call
        read_only = getattr(request.tool, "read_only", None)
        is_write = tool_identity_class(call["name"], read_only) is ToolIdentityClass.WRITE
        key = self._call_key(call, request.tool)
        # Before the deadline check, so a change of unknown outcome is never
        # described as one that just wasn't run.
        if key in self._timed_out_calls:
            text = WRITE_REPEAT_TEXT if is_write else REPEAT_TEXT
            return _not_run(call, text, "repeat_timeout")
        # A change is left to finish: cutting it off would leave it unknown whether
        # it happened. So one starts only if its own timeout ends by the deadline.
        # Every write is an MCP tool, which is what has a call_timeout.
        lead = getattr(request.tool, "call_timeout", 0.0) if is_write else 0.0
        remaining = self.tool_deadline - time.monotonic()
        if remaining <= lead:
            return _not_run(call, PAST_DEADLINE_TEXT, "turn_deadline")
        if is_write:
            result = await handler(request)
        else:
            result = await self._run_read(request, handler, remaining)
        if not _timed_out(result):
            return result
        self._timed_out_calls.add(key)
        if is_write and isinstance(result, ToolMessage):
            return result.model_copy(update={"content": WRITE_TIMED_OUT_TEXT, "status": "error"})
        return result

    def _answer_only(self, message: AIMessage) -> AIMessage:
        """The last response as an answer, so the turn ends here.

        A tool call left on it would loop into the recursion limit, whose apology
        replaces everything. The copy in additional_kwargs goes too, because the
        next request would replay it as a call with no result.
        """
        text = _RAW_TOOL_CALL_RE.sub("", message.text).strip()
        if not text:
            self.gave_up = True
        return message.model_copy(
            update={
                "content": text or OUT_OF_TIME_ANSWER,
                "tool_calls": [],
                "invalid_tool_calls": [],
                "additional_kwargs": {
                    k: v for k, v in message.additional_kwargs.items() if k != "tool_calls"
                },
            }
        )

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        last_call = self.out_of_time()
        if last_call:
            self.deadline_hit = True
            request = request.override(tools=[], messages=_with_answer_note(request.messages))
        started = time.monotonic()
        response = await handler(request)
        self._longest_model_call_s = max(self._longest_model_call_s, time.monotonic() - started)
        if not last_call:
            return response
        return ModelResponse(
            result=[
                self._answer_only(m) if isinstance(m, AIMessage) else m for m in response.result
            ],
            structured_response=response.structured_response,
        )
