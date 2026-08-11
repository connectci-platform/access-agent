"""Tool-calling loop node — the only node in the agent graph.

LangChain's ``create_agent`` drives a turn-by-turn loop where the LLM
selects tools, sees results as ToolMessages, and continues until it emits a
non-tool-call response. The system prompt instructs the LLM to call
``search_access_documents`` for documentation-style questions; live data
flows from the MCP catalog tools.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.errors import GraphRecursionError

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.tools import BaseTool

from ...config import settings
from ...llm import get_llm
from ...telemetry import get_tracer
from ..domains.capabilities import WRITE_MCP_TOOL_NAMES
from ..domains.tools import create_mcp_tools_from_catalog
from ..prompts.system_prompt import build_system_prompt
from ..state import ToolResult
from ..tools import search_access_documents
from ..turn_capture import get_turn_capture, mark_summarized

logger = logging.getLogger(__name__)


def _build_prompt_and_tools(
    *,
    tool_catalog: dict[str, Any],
    acting_user: str | None,
    resource_context: str | None,
) -> tuple[str, list[BaseTool]]:
    """Assemble the loop's system prompt and tool list.

    Tool list = MCP catalog (read-only-filtered) + ``search_access_documents``.
    Prompt is the loop's system prompt (docs-as-tool framing, announcements +
    JSM choreography appended).
    """
    mcp_tools = _apply_read_only_filter(create_mcp_tools_from_catalog(tool_catalog, acting_user))
    prompt = build_system_prompt(
        acting_user=acting_user,
        resource_context=resource_context,
    )
    return prompt, [*mcp_tools, search_access_documents]


def _apply_read_only_filter(tools: list[BaseTool]) -> list[BaseTool]:
    """Strip write-capable MCP tools when ``settings.READ_ONLY`` is True.

    See `docs/security/write-capability-audit.md`.
    """
    if not settings.READ_ONLY:
        return tools
    before = len(tools)
    filtered = [t for t in tools if t.name not in WRITE_MCP_TOOL_NAMES]
    removed = before - len(filtered)
    if removed:
        logger.info(
            "READ_ONLY=true active in tool_calling_loop — removed %d "
            "write-capable tool(s) from registry",
            removed,
        )
    return filtered


StatusWriter = Callable[[dict[str, Any]], None]


def _resolve_status_writer() -> StatusWriter | None:
    """Return the LangGraph stream writer, or ``None`` outside a runnable.

    ``get_stream_writer()`` raises ``RuntimeError`` when called outside a
    LangGraph runnable (eg. direct invocation in tests). The ``None`` form
    lets call sites stay compact and lets tests exercise the node directly.
    """
    try:
        return get_stream_writer()
    except RuntimeError:
        return None


def _emit_status(message: str) -> None:
    """Emit a status event to the SSE stream, no-op outside runnable context."""
    writer = _resolve_status_writer()
    if writer is not None:
        writer({"type": "status", "message": message})


# Status-bubble wording. Raw tool names (`search_software`, `get_compute_resource`)
# would otherwise surface verbatim in the chat UI. The first token of a tool name
# is its verb; the rest is the subject — a verb→phrase table plus a few overrides
# turns any MCP tool name into a human phrase without an exhaustive list.
_STATUS_VERB_PHRASES = {
    "get": "Looking up",
    "list": "Looking up",
    "search": "Searching",
    "check": "Checking",
    "describe": "Looking up",
    "execute": "Running",
    "analyze": "Analyzing",
    "compare": "Comparing",
    "recommend": "Finding",
    "create": "Preparing",
    "report": "Reporting",
    "delete": "Removing",
}

# Overrides where the generic verb+subject form reads poorly.
_STATUS_TOOL_OVERRIDES = {
    "search_access_documents": "Searching ACCESS documentation...",
    "search_nsf_awards": "Searching NSF awards...",
}


def _friendly_tool_status(name: str) -> str:
    """Turn a raw tool name into a human-readable status-bubble message."""
    override = _STATUS_TOOL_OVERRIDES.get(name)
    if override:
        return override
    verb, _, rest = name.partition("_")
    phrase = _STATUS_VERB_PHRASES.get(verb)
    if phrase and rest:
        return f"{phrase} {rest.replace('_', ' ')}..."
    return f"Working on {name.replace('_', ' ')}..."


def _coerce_content_to_text(content: Any) -> str:
    """Render AIMessage.content as a single string.

    Providers that emit content blocks (Anthropic-style) return a list of
    dicts like ``[{"type": "text", "text": "..."}]``; ``str()`` on that list
    produces Python repr, not the user-facing answer.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content)


class _ToolStatusEmitter(AsyncCallbackHandler):
    """Callback handler that emits a status event each time a tool starts.

    Registered via ``config["callbacks"]`` on the agent invocation, so the
    react loop fires ``on_tool_start`` for every tool the LLM invokes —
    including parallel calls within a single turn. The writer is captured
    at node entry rather than re-resolved on each event, because callback
    handlers may run on a task that doesn't share the node's contextvar
    snapshot (no langgraph runnable context = ``get_stream_writer`` would
    raise).
    """

    def __init__(self, writer: StatusWriter | None) -> None:
        super().__init__()
        self._writer = writer

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Unused today, kept in the signature to document LangChain's
        # on_tool_start contract — `run_id` is the correlation key for any
        # future on_tool_end pairing or LangSmith tracing.
        del input_str, run_id, parent_run_id, tags, metadata, inputs, kwargs
        if self._writer is None:
            return
        name = serialized.get("name") if serialized else None
        if not name:
            return
        self._writer({"type": "status", "message": _friendly_tool_status(name)})


class _TokenUsageAccumulator(AsyncCallbackHandler):
    """Sums token usage and times each LLM call across this turn.

    create_agent calls the model once per tool-calling step; on_llm_end fires
    per call. We accumulate usage_metadata.total_tokens for the turn total
    (summing over final_state messages would double-count, since checkpointing
    prepends prior-turn history) and record one {index, duration_ms,
    total_tokens} entry per call. Start times are keyed by run_id so parallel
    calls pair correctly; an end with no matching start records 0ms.
    """

    def __init__(self) -> None:
        self.total_tokens = 0
        self.model_calls: list[dict[str, Any]] = []
        self._starts: dict[Any, float] = {}

    async def on_llm_start(
        self, serialized: Any, prompts: Any, *, run_id: Any = None, **kwargs: Any
    ) -> None:
        del serialized, prompts, kwargs
        self._starts[run_id] = time.monotonic()

    async def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: Any = None, **kwargs: Any
    ) -> None:
        del serialized, messages, kwargs
        self._starts[run_id] = time.monotonic()

    async def on_llm_end(self, response: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        del kwargs
        call_tokens = 0
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg is not None else None
                if usage:
                    call_tokens += int(usage.get("total_tokens") or 0)
        self.total_tokens += call_tokens
        started = self._starts.pop(run_id, None)
        duration_ms = int((time.monotonic() - started) * 1000) if started is not None else 0
        self.model_calls.append(
            {
                "index": len(self.model_calls),
                "duration_ms": duration_ms,
                "total_tokens": call_tokens or None,
            }
        )


class _FlaggingSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware that records when it actually summarizes.

    The built-in compacts history transparently — nothing in final_state says
    it fired. before_model/abefore_model return non-None only when a summary is
    produced, so we set a per-turn flag (via turn_capture) on that signal.
    Deterministic; no message-count guessing.

    Relies on SummarizationMiddleware.before_model / abefore_model returning None when no compaction occurred — re-verify this invariant on LangChain upgrades.
    """

    def before_model(self, state: Any, runtime: Any) -> Any:
        result = super().before_model(state, runtime)
        if result is not None:
            mark_summarized()
        return result

    async def abefore_model(self, state: Any, runtime: Any) -> Any:
        result = await super().abefore_model(state, runtime)
        if result is not None:
            mark_summarized()
        return result


async def tool_calling_loop_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the tool-calling loop.

    Consumes:
        state.messages: conversation history (HumanMessage / AIMessage / ToolMessage)
        state.query: current user query (already present in messages[-1])
        state.tool_catalog: dict describing MCP tools available to this request
        state.acting_user: optional ACCESS ID for personalized tool calls
        state.resource_context: optional RP slug for resource-scoped queries

    Produces:
        final_answer: LLM's final text response
        messages: full loop message history (caller messages + tool calls + result messages + final)
        tools_used: list of tool-name strings the loop actually invoked
        node_trace: telemetry entry for this node
    """
    tracer = get_tracer("access-agent.nodes")

    status_writer = _resolve_status_writer()
    if status_writer is not None:
        status_writer({"type": "status", "message": "Processing query..."})

    with tracer.start_as_current_span(
        "agent.tool_calling_loop",
        attributes={"agent.node": "tool_calling_loop"},
    ) as span:
        acting_user = state.get("acting_user")

        system_prompt, tools = _build_prompt_and_tools(
            tool_catalog=state.get("tool_catalog") or {},
            acting_user=acting_user,
            resource_context=state.get("resource_context"),
        )

        span.set_attribute("agent.tool_count", len(tools))
        span.set_attribute("agent.authenticated", bool(acting_user))

        llm = get_llm(max_tokens=settings.MAX_TOKENS_LOOP)
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
            middleware=[
                _FlaggingSummarizationMiddleware(
                    model=llm,
                    trigger=("tokens", settings.SUMMARIZATION_TRIGGER_TOKENS),
                    keep=("tokens", settings.SUMMARIZATION_KEEP_TOKENS),
                ),
            ],
        )

        messages = list(state.get("messages", []))
        logger.info(
            "Running tool_calling_loop: %d tools, %d prior messages",
            len(tools),
            len(messages),
        )

        final_answer: str | None = None
        recursion_limit_hit = False

        # recursion_limit = 2 * max_tool_turns + 1; gives the LLM room for
        # roughly 10 tool turns before LangGraph hard-stops.
        recursion_limit = 25
        tool_status_emitter = _ToolStatusEmitter(status_writer)
        token_accumulator = _TokenUsageAccumulator()
        try:
            result = await agent.ainvoke(
                {"messages": messages},
                {
                    "recursion_limit": recursion_limit,
                    "callbacks": [tool_status_emitter, token_accumulator],
                },
            )
            result_messages = result.get("messages", [])
        except GraphRecursionError:
            # create_agent exhausted its recursion budget (e.g., the LLM
            # kept requesting tool calls and never emitted a final answer).
            # Return a user-facing apology rather than a 500 so the chatbot
            # can render something useful.
            recursion_limit_hit = True
            logger.warning(
                "tool_calling_loop hit GraphRecursionError: "
                "tool_count=%d, recursion_limit=%d, messages_so_far=%d",
                len(tools),
                recursion_limit,
                len(messages),
            )
            span.set_attribute("agent.recursion_limit_hit", True)
            result_messages = list(messages)
            final_answer = (
                "I wasn't able to complete an answer for this query within "
                "the allotted tool-turn budget. You can try rephrasing, or "
                "open a support ticket at "
                "https://support.access-ci.org/open-a-ticket."
            )

        if not recursion_limit_hit:
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    final_answer = _coerce_content_to_text(msg.content)
                    break

        tools_used: list[str] = []
        for msg in result_messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name and name not in tools_used:
                        tools_used.append(name)

        tool_result_count = sum(1 for m in result_messages if isinstance(m, ToolMessage))
        tool_results, orphan_count = _build_tool_results(
            result_messages, tools, get_turn_capture().get("tool_timings", [])
        )

        answer_length = len(final_answer) if final_answer else 0
        span.set_attribute("agent.answer_length", answer_length)
        span.set_attribute("agent.tool_calls_made", len(tools_used))
        span.set_attribute("agent.tool_results_received", tool_result_count)
        if orphan_count > 0:
            span.set_attribute("agent.tool_results_orphaned", orphan_count)

        logger.info(
            "tool_calling_loop complete: %d messages, %d tool calls, "
            "%d tool results, answer_len=%d",
            len(result_messages),
            len(tools_used),
            tool_result_count,
            answer_length,
        )

        return {
            "final_answer": final_answer,
            "messages": result_messages,
            "tools_used": tools_used,
            "total_tokens": token_accumulator.total_tokens,
            "model_calls": token_accumulator.model_calls,
            "tool_results": tool_results,
            "node_trace": [
                {
                    "node": "tool_calling_loop",
                    "tool_count": len(tools),
                    "tool_calls_made": len(tools_used),
                    "tools_called": list(tools_used),
                    "tool_results": tool_result_count,
                    "answer_length": answer_length,
                }
            ],
        }


def _build_call_lookup(
    result_messages: list[Any],
) -> dict[str, tuple[str, dict[str, Any]]]:
    """Map tool_call_id → (tool_name, args) from all AIMessage tool_calls in the thread."""
    call_lookup: dict[str, tuple[str, dict[str, Any]]] = {}
    for msg in result_messages:
        if not (isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None)):
            continue
        for tc in msg.tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            tc_args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {})
            if tc_id and tc_name:
                call_lookup[tc_id] = (tc_name, tc_args or {})
    return call_lookup


def _build_tool_results(
    result_messages: list[Any],
    tools: list[Any],
    tool_timings: list[dict[str, Any]],
) -> tuple[list[ToolResult], int]:
    """Back-fill state.tool_results from a react-loop message thread.

    Downstream consumers (eval scorer, observability) read state.tool_results
    in a structured form. The react loop only emits ToolMessages, so we
    reconstruct ToolResult objects by pairing each ToolMessage with its
    originating AIMessage tool_call (by id) and parsing the JSON content
    MCPToolWrapper produced.

    Orphan ToolMessages (no matching tool_call id) are dropped — we never
    fabricate a ToolResult we can't anchor to an AIMessage call. The caller
    still receives the orphan count so telemetry can surface these rare
    cases (e.g., future LLM quirks where a ToolMessage appears without a
    matching call).

    duration_ms is paired from tool_timings (recorded at the call sites) FIFO
    per tool name, but ONLY onto ToolMessages that appear after the last
    HumanMessage in result_messages — those are the current turn's calls.
    Prior-turn ToolMessages (at or before that boundary) keep duration_ms=0;
    0 is the honest value because their wall-clock time was not recorded for
    this turn. Limitation: two parallel calls to the SAME tool in one turn may
    swap durations between them (records append in completion order).

    Every entry carries ``message_index`` — its source ToolMessage's position in
    ``result_messages``. Multi-turn consumers (src/eval/multiturn.py) slice this
    cumulative list to one turn by comparing that index against the same
    last-HumanMessage boundary, which is stateless and therefore immune to
    repeated provider tool_call_ids and to compaction shrinking the rebuild.

    Returns:
        (tool_results, orphan_count) — the reconstructed ToolResult list and
        the number of ToolMessages skipped because their tool_call_id had no
        matching AIMessage tool_call.
    """
    tool_server_lookup: dict[str, str] = {
        getattr(t, "name", ""): getattr(t, "tool_server", "") for t in tools
    }

    call_lookup = _build_call_lookup(result_messages)

    # Find the boundary: ToolMessages at or before the last HumanMessage index
    # belong to prior turns. Only messages after that index get timing data.
    human_boundary = -1
    for i, msg in enumerate(result_messages):
        if isinstance(msg, HumanMessage):
            human_boundary = i

    # Pair capture-recorded durations back to current-turn results, FIFO per tool name.
    timing_queues: dict[str, deque[int]] = {}
    for rec in tool_timings:
        timing_queues.setdefault(rec["tool_name"], deque()).append(rec["duration_ms"])

    results: list[ToolResult] = []
    orphan_count = 0
    for i, msg in enumerate(result_messages):
        if not isinstance(msg, ToolMessage):
            continue
        tc_id = msg.tool_call_id
        if not tc_id or tc_id not in call_lookup:
            orphan_count += 1
            continue
        tool_name, tool_args = call_lookup[tc_id]
        server = tool_server_lookup.get(tool_name, "")
        # Only pop a timing for current-turn messages (after the last HumanMessage).
        duration_ms = 0
        if i > human_boundary:
            queue = timing_queues.get(tool_name)
            if queue:
                duration_ms = queue.popleft()
        result = _parse_tool_message(
            msg, tc_id, tool_name, server, tool_args, duration_ms=duration_ms, message_index=i
        )
        results.append(result)

    leftover = sum(len(q) for q in timing_queues.values())
    if leftover:
        logger.debug(
            "tool timing pairing: %d recorded timing(s) had no matching current-turn ToolMessage",
            leftover,
        )

    return results, orphan_count


def _parse_tool_message(
    msg: ToolMessage,
    tc_id: str,
    tool_name: str,
    server: str,
    tool_args: dict[str, Any],
    duration_ms: int = 0,
    message_index: int = -1,
) -> ToolResult:
    """Build a ToolResult from a single ToolMessage + its originating call metadata.

    ``message_index`` is the ToolMessage's position in the thread it was rebuilt
    from; multi-turn consumers use it to slice this cumulative list down to one
    turn (entries past the last HumanMessage).
    """
    raw_content = msg.content if isinstance(msg.content, str) else str(msg.content)
    try:
        parsed: Any = json.loads(raw_content)
    except (json.JSONDecodeError, TypeError):
        parsed = raw_content

    if isinstance(parsed, dict) and "error" in parsed:
        return ToolResult(
            step_id=tc_id,
            tool_name=tool_name,
            server=server,
            success=False,
            error=str(parsed["error"]),
            arguments=tool_args,
            duration_ms=duration_ms,
            message_index=message_index,
        )

    return ToolResult(
        step_id=tc_id,
        tool_name=tool_name,
        server=server,
        success=True,
        data=parsed,
        arguments=tool_args,
        duration_ms=duration_ms,
        message_index=message_index,
    )
