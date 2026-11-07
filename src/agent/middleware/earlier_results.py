"""Middleware: the model sees results from earlier turns as earlier.

Live data can change between turns and a failure can pass, but a thread keeps
both word for word. Shown them as they are, the model answers a repeated
question from the thread, with no tool call, and calls the old result current.
Prompt rules don't hold for Qwen, which trusts message content over them, so the
model's view of the thread is rewritten instead. The stored thread doesn't
change, so ``request.state`` keeps the original messages for code that reads
them, such as ``PreviewFirstMiddleware``.

An earlier turn is anything before the latest human message. That holds after
compaction, which moves questions forward but keeps the current question ahead
of this turn's results. Tools are told apart by the result's ``name``, not its
call id, because vLLM reuses call ids across a thread.

A live result keeps its body, so "summarize that" can still be answered without
a tool call. The note on the latest message leaves that choice to the model.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..auth_tools import ToolIdentityClass, tool_identity_class
from ..domains.capabilities import live_data_tool_names

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse

EARLIER_RESULT_PREFIX = (
    "[Result from an earlier turn, so it may be out of date. If the user asks for "
    "this data again, call the tool again.]"
)
EARLIER_FAILURE_TEXT = (
    "This call failed in an earlier turn. That failure is over and says nothing "
    "about whether the tool works now. If the user asks for this data again, call "
    "the tool again."
)
EARLIER_NOTES_PREFIX = (
    "[These notes are from earlier turns. Any live data or failures in them may be out of date.]"
)
EARLIER_TURN_NOTE = (
    "(Results from before this message are out of date, and so are earlier answers "
    "based on them. If this message asks for live data, call the tool again even if "
    "it was asked before, and don't call an earlier result current. Results after "
    "this message are current. If this message only discusses an earlier answer, "
    "no tool call is needed.)"
)


def _is_failure(message: ToolMessage) -> bool:
    """Whether this result is a failure, as the loop's turn accounting reads one."""
    if message.status == "error":
        return True
    try:
        result = json.loads(str(message.content))
    except json.JSONDecodeError:
        return False
    return isinstance(result, dict) and "error" in result


def _is_cleared(message: ToolMessage) -> bool:
    """Whether context editing replaced this result's body, and its artifact."""
    return bool(message.response_metadata.get("context_editing", {}).get("cleared"))


def _is_refusal(message: ToolMessage) -> bool:
    return isinstance(message.artifact, dict) and "refusal" in message.artifact


def _earlier_tool_text(message: ToolMessage, live_tools: frozenset[str]) -> str | None:
    """What the model should see for this earlier result, or None to leave it.

    A refusal may still apply, and ``signed_in_since`` handles the ones that
    don't. A failed write keeps its reason and gets no invitation to retry. A
    cleared result has no body to judge, and may be a refusal without its
    artifact, so only its tool decides.
    """
    name = message.name
    if not name or _is_refusal(message):
        return None
    if not _is_cleared(message) and _is_failure(message):
        is_write = tool_identity_class(name) is ToolIdentityClass.WRITE
        return None if is_write else EARLIER_FAILURE_TEXT
    return f"{EARLIER_RESULT_PREFIX}\n\n{message.text}" if name in live_tools else None


def _earlier_text(message: Any, live_tools: frozenset[str]) -> str | None:
    if isinstance(message, ToolMessage):
        return _earlier_tool_text(message, live_tools)
    if (
        isinstance(message, AIMessage)
        and message.additional_kwargs.get("lc_source") == "summarization"
    ):
        return f"{EARLIER_NOTES_PREFIX}\n\n{message.text}"
    return None


def earlier_results_view(messages: list[Any]) -> list[Any]:
    """The model's view of a thread, with earlier turns' results marked as earlier.

    Returns ``messages`` itself when nothing needs marking. Also used by the
    summarization token counter, on suffixes of the thread that may hold no
    human message.
    """
    latest = len(messages) - 1
    while latest >= 0 and not isinstance(messages[latest], HumanMessage):
        latest -= 1

    live_tools = live_data_tool_names()
    view = list(messages)
    marked = False
    for index, message in enumerate(messages[: max(latest, 0)]):
        text = _earlier_text(message, live_tools)
        if text is not None:
            view[index] = message.model_copy(update={"content": text})
            marked = True
    if not marked:
        return messages
    if isinstance(view[latest].content, str):
        note = f"{view[latest].content}\n\n{EARLIER_TURN_NOTE}"
        view[latest] = view[latest].model_copy(update={"content": note})
    return view


class EarlierResultsMiddleware(AgentMiddleware):
    """Send the model ``earlier_results_view`` of the thread."""

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(request.override(messages=earlier_results_view(request.messages)))
