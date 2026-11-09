"""Suggest follow-up questions after a full-screen answer.

Runs after tool_calling_loop on every turn, but does work only on a
full-screen thread run (``AgentContext.thread_run``); the widget's /query
path and the eval get an empty update. One short model call turns the
person's question, the answer, and the tools this run could call into one to
three questions the person might ask next. They are attached to the answer as
``additional_kwargs["suggested_questions"]`` by returning the answer again
under its own id, which ``add_messages`` replaces in place, so the answer's
text and its streamed tokens are untouched.

The answer always stands. A timeout, a model error, unusable output, or an
answer that was an apology or a sign-in refusal all mean no suggestions and
no exception. The call is tagged ``nostream`` so its tokens never reach a
client as answer text, and it is bounded by what is left of the turn so it
can't push a finished answer past the route's timeout.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.constants import TAG_NOSTREAM
from langgraph.runtime import Runtime  # noqa: TC002 - LangGraph reads node annotations at runtime

from ...llm import get_llm, is_empty_answer
from ...telemetry import get_tracer
from ..auth_tools import ToolIdentityClass, tool_identity_class
from ..context import AgentContext, require_context
from ..tools import search_access_documents
from ..turn_capture import get_turn_capture

logger = logging.getLogger(__name__)

SUGGESTED_QUESTIONS_KEY = "suggested_questions"
MAX_SUGGESTIONS = 3
MAX_QUESTION_CHARS = 120
SUGGESTION_TIMEOUT_S = 8.0
# Left between the end of this call and the route's turn deadline, so the
# values event still goes out before the route gives up on the turn.
_DEADLINE_MARGIN_S = 2.0
# Below this there is no point starting a call.
_MIN_CALL_S = 1.0
_MAX_TOKENS = 200
_TEMPERATURE = 0.2
_ANSWER_CHARS = 1500
_QUESTION_CHARS = 500
_TOOL_DESCRIPTION_CHARS = 120

_SYSTEM_PROMPT = """\
You suggest follow-up questions for the ACCESS Assistant, which helps researchers \
use ACCESS cyberinfrastructure.

Given a person's question, the assistant's answer, and the tools the assistant can \
use, suggest 1 to 3 short questions the person is likely to ask next.

Rules:
- Write each one as the person would ask it, in the first person, under 100 characters.
- Make them specific to this conversation, naming the resource, software, or topic.
- Do not repeat the question or ask for something the answer already covers.
- Only suggest questions the listed tools or ACCESS documentation can answer.
- Only ask for information. Never ask the assistant to do or change anything, \
such as registering, cancelling, creating, updating, submitting, or opening a ticket.

Reply with only a JSON array of strings, like ["First question?", "Second question?"]. \
Reply with [] if there is no good follow-up."""

# A suggestion is sent as the person's next message, so one that asks the
# assistant to act would be a request they never made. The prompt says so,
# and this enforces it.
_ACTION_VERBS = (
    r"(?:register|unregister|sign|enroll|cancel|delete|remove|submit|create|update|edit"
    r"|change|modify|post|publish|open|file|book|reserve|rsvp|send|add|approve|request"
    r"|schedule|restore|renew|extend|make|set|join|leave|withdraw)"
)
_REQUEST_LEAD = (
    r"(?:(?:can|could|would|will) you(?: please)?|please|help me|go ahead and"
    r"|i(?:'d| would) like (?:you )?to|i want (?:you )?to|i need (?:you )?to|let'?s)"
)
_ACTION_PATTERNS = (
    # An imperative: "Register me for...", "Cancel my registration".
    re.compile(rf"^\s*(?:{_REQUEST_LEAD}\s+)?{_ACTION_VERBS}\b", re.IGNORECASE),
    # A request anywhere: "Can you sign me up...", "...so please create one".
    re.compile(rf"\b{_REQUEST_LEAD}\s+{_ACTION_VERBS}\b", re.IGNORECASE),
    re.compile(
        r"\b(?:sign|register|enroll|add|book|rsvp)\s+(?:me|us)\b"
        r"|\b(?:open|file|submit|create|start|raise)\s+(?:a|an|the|my|new)?\s*"
        r"(?:support\s+)?(?:ticket|request)\b",
        re.IGNORECASE,
    ),
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^\w\s]")


def parse_suggestions(text: str | None) -> list[str]:
    """The strings in the model's JSON array, or [] when there is none.

    Tolerates a fenced block and prose around the array.
    """
    if not text or is_empty_answer(text):
        return []
    fenced = _FENCE.search(text)
    body = fenced.group(1) if fenced else text
    start, end = body.find("["), body.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        # Bracketed at both ends, so it is a list if it parses at all.
        parsed = json.loads(body[start : end + 1])
    except ValueError:
        return []
    return [item for item in parsed if isinstance(item, str)]


def _normalize(text: str) -> str:
    return _WHITESPACE.sub(" ", _NON_WORD.sub("", text.lower())).strip()


def asks_to_act(question: str) -> bool:
    """Whether a suggestion asks the assistant to do or change something."""
    return any(p.search(question) for p in _ACTION_PATTERNS)


def filter_suggestions(candidates: list[str], original_question: str) -> list[str]:
    """Keep at most three clean, distinct, information-only questions."""
    seen = {_normalize(original_question)}
    kept: list[str] = []
    for candidate in candidates:
        question = _WHITESPACE.sub(" ", candidate).strip().strip("\"'").strip()
        key = _normalize(question)
        if not key or key in seen or len(question) > MAX_QUESTION_CHARS:
            continue
        if asks_to_act(question):
            continue
        seen.add(key)
        kept.append(question)
        if len(kept) == MAX_SUGGESTIONS:
            break
    return kept


def _one_line(description: Any) -> str:
    first = str(description or "").strip().splitlines()
    line = first[0].strip() if first else ""
    return line[:_TOOL_DESCRIPTION_CHARS]


def tool_summaries(tool_catalog: dict[str, Any], *, signed_in: bool) -> list[str]:
    """``name: description`` for each tool this caller could use to answer.

    Write tools are left out, since a suggestion only asks for information,
    and so are sign-in tools for an anonymous caller, whose question would
    only be refused.
    """
    lines = [f"{search_access_documents.name}: {_one_line(search_access_documents.description)}"]
    for server in tool_catalog.get("servers", []) or []:
        if server.get("status", "available") != "available":
            continue
        for tool in server.get("tools", []) or []:
            name = tool.get("name")
            if not name:
                continue
            identity = tool_identity_class(name, tool.get("read_only"))
            if identity is ToolIdentityClass.WRITE:
                continue
            if identity is ToolIdentityClass.AUTH_READ and not signed_in:
                continue
            lines.append(f"{name}: {_one_line(tool.get('description'))}")
    return lines


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        )
    return ""


def _answer_to_follow(state: dict[str, Any]) -> AIMessage | None:
    """The turn's answer when it is one worth following up, else None.

    Not an apology for a missing answer, not a turn the identity gate refused
    (its answer is a sign-in prompt), and not a message still asking for tools.
    """
    if state.get("answer_unavailable") or state.get("tool_refusals"):
        return None
    messages = state.get("messages") or []
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, AIMessage) or last.tool_calls or not last.id:
        return None
    text = _text(last.content).strip()
    if not text or is_empty_answer(text):
        return None
    return last


def _time_left() -> float:
    """Seconds this call may take: the cap, or less near the turn's deadline."""
    deadline: float | None = get_turn_capture().get("turn_deadline")
    if deadline is None:
        return SUGGESTION_TIMEOUT_S
    return min(SUGGESTION_TIMEOUT_S, deadline - time.monotonic() - _DEADLINE_MARGIN_S)


def _prompt(question: str, answer: str, tools: list[str]) -> list[Any]:
    if len(answer) > _ANSWER_CHARS:
        answer = answer[:_ANSWER_CHARS].rstrip() + " ..."
    tool_list = "\n".join(f"- {line}" for line in tools)
    return [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Question:\n{question[:_QUESTION_CHARS]}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Tools:\n{tool_list}"
            )
        ),
    ]


async def _suggest(question: str, answer: str, tools: list[str], timeout: float) -> list[str]:
    llm = get_llm(temperature=_TEMPERATURE, max_tokens=_MAX_TOKENS, enable_thinking=False)
    response = await asyncio.wait_for(
        llm.ainvoke(_prompt(question, answer, tools), config={"tags": [TAG_NOSTREAM]}),
        timeout=timeout,
    )
    return filter_suggestions(parse_suggestions(_text(response.content)), question)


async def suggest_follow_ups_node(
    state: dict[str, Any], runtime: Runtime[AgentContext]
) -> dict[str, Any]:
    """Attach follow-up questions to a full-screen answer, or change nothing.

    Consumes state.query, state.messages (the answer is the last one),
    state.tool_catalog, state.answer_unavailable and state.tool_refusals.
    Produces, only when there are suggestions, the answer message again with
    ``additional_kwargs["suggested_questions"]`` set. No node_trace entry: the
    routes keep each node's raw update, so one here would replace the loop's
    in the turn report. The span and log line carry the count and duration.
    """
    context = require_context(runtime.context)
    if not context.thread_run:
        return {}
    answer = _answer_to_follow(state)
    question = (state.get("query") or "").strip()
    if answer is None or not question:
        return {}
    timeout = _time_left()
    if timeout < _MIN_CALL_S:
        logger.info("suggest_follow_ups skipped: %.1fs left in the turn", timeout)
        return {}

    tracer = get_tracer("access-agent.nodes")
    with tracer.start_as_current_span(
        "agent.suggest_follow_ups",
        attributes={"agent.node": "suggest_follow_ups"},
    ) as span:
        tools = tool_summaries(
            state.get("tool_catalog") or {}, signed_in=context.acting_user is not None
        )
        started = time.monotonic()
        outcome = "ok"
        try:
            suggestions = await _suggest(question, _text(answer.content), tools, timeout)
        except TimeoutError:
            outcome, suggestions = "timeout", []
        except Exception:
            logger.exception("suggest_follow_ups failed; the answer stands without suggestions")
            outcome, suggestions = "error", []
        duration_ms = int((time.monotonic() - started) * 1000)

        span.set_attribute("agent.follow_up_count", len(suggestions))
        span.set_attribute("agent.follow_up_duration_ms", duration_ms)
        span.set_attribute("agent.follow_up_outcome", outcome)
        logger.info(
            "suggest_follow_ups complete: %d suggestions in %dms (%s)",
            len(suggestions),
            duration_ms,
            outcome,
        )

    if not suggestions:
        return {}
    updated = answer.model_copy(
        update={
            "additional_kwargs": {**answer.additional_kwargs, SUGGESTED_QUESTIONS_KEY: suggestions}
        }
    )
    return {"messages": [updated]}
