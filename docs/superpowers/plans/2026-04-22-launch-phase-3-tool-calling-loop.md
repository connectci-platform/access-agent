# Launch Phase 3 — Tool-calling Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the `plan → execute → evaluate → recover → synthesize` node sequence with a single `tool_calling_loop` node that uses LangGraph's `create_react_agent` + LangChain's `.bind_tools()` to let the LLM drive tool selection, execution, and recovery natively. Gate behind `USE_TOOL_CALLING_LOOP=true` so the old path remains functional.

**Architecture:** A single node that:
1. Assembles a system prompt from agent identity, capability description, optional RAG context, and classifier domain hint.
2. Creates LangChain tool wrappers for every enabled MCP tool (reuses existing `MCPToolWrapper`).
3. Runs `create_react_agent(llm, tools, prompt=...)` — an LLM-driven loop that emits tool calls, executes them, feeds results back, and iterates until the LLM produces a final answer.
4. Returns the final text plus message history.

The loop subsumes planning (the LLM picks tools turn-by-turn), execution (the `ToolNode` inside the react agent runs them), evaluation (the LLM decides when its answer is complete), and recovery (the LLM sees tool failures and either retries or pivots). The hand-rolled `$step_N` resolver in `execute.py` retires because tool outputs chain naturally through the message history.

The `rag_answer` and `domain_agent` nodes stay unchanged — the flag only affects queries that currently traverse `plan → execute → evaluate → recover → synthesize`.

**Tech Stack:** Python 3.11+, `langgraph>=1.1.0` (Phase 0 bumped), `langchain-core>=1.3.0`, `langchain-openai>=1.1.0`, existing `MCPToolWrapper` from `src/agent/domains/tools.py`, existing `get_llm()` from `src/llm/providers.py`.

**Spec:** `docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md` §Phase 3.

**Umbrella plan:** `docs/superpowers/plans/2026-04-21-production-launch-umbrella.md`.

**Prerequisites:**
- Phase 0 (deps upgrade to langgraph 1.x) — done, tagged `launch/phase-0-deps-upgrade`.
- Phase 1 (READ_ONLY guard) — done, tagged `launch/phase-1-safety-audit`. The loop inherits the guard because it reads from the capability registry.

**Unblocks:** Phase 5 (UKY vLLM swap via env change), Phase 7 (side-by-side evidence package needs both paths working so we can compare).

---

## Architectural notes — things the plan assumes

**Why `create_react_agent` and not hand-rolled loop:**
- `domain_agent` already uses it successfully — proven pattern in this codebase.
- LangGraph 1.1.9 re-exports it from `langgraph.prebuilt`; verified in Phase 0.
- It handles recursion limits, tool schema binding, message threading, and interruption semantics out of the box.
- Saves ~200 lines of loop code we'd otherwise maintain.

**Where the old nodes' logic goes:**
- `plan.py`'s tool-catalog text-compaction and LLM-driven tool selection → replaced by LangChain's `.bind_tools()` + the LLM's turn-by-turn reasoning.
- `execute.py`'s parallel/sequential/mixed strategies → the react loop does one tool call per turn. Parallelism across tools is available via `parallel_tool_calls=True` on OpenAI's API (on by default in LangChain).
- `execute.py`'s `$step_N` resolver → obsolete. Tool results flow via `ToolMessage` objects; the LLM reads the actual values on the next turn.
- `evaluate.py` → the LLM's decision to stop emitting tool calls IS the evaluation. When the LLM produces a non-tool-call response, the loop terminates.
- `recover.py`'s error classification and retry strategy → the LLM sees failed tool results as `ToolMessage` content and decides on the next turn whether to retry, try an alternative, or explain the failure.

**What's intentionally NOT in this plan (post-launch work):**
- Fine-grained retry budgets (the old `RetryContext` with per-tool and total caps). The new loop uses LangGraph's `recursion_limit` only. If we need budget accounting post-launch, it's a cleanup PR.
- Replacing `synthesize.py` for the non-tool-calling paths. The loop's final answer IS synthesized by the LLM inline; `synthesize.py` only runs when `USE_TOOL_CALLING_LOOP=false`.
- Deleting the old nodes. Spec says "Old path remains functional." We mark them deprecated in comments and leave the code for rollback safety. Deletion is Phase 8 cleanup.

**Behavioral compatibility target:**
- Eval battery pass rate on the new path should match or exceed the old path within the same tolerance the launch spec's Phase 7 comparison allows.
- User-facing latency on the new path is expected to be *lower* on simple queries (one LLM call + tools + one more LLM call, vs the old path's plan → evaluate → synthesize chain of 3 sequential LLM calls). Complex multi-tool queries may be slightly slower due to chattier turn structure. Phase 7 quantifies this.

---

## File Structure

**Create:**
- `src/agent/nodes/tool_calling_loop.py` — the new node.
- `src/agent/prompts/tool_calling_loop.py` — system prompt assembly for the loop.
- `tests/test_tool_calling_loop.py` — unit + integration tests for the new node.

**Modify:**
- `src/config.py` — add `USE_TOOL_CALLING_LOOP: bool = False`.
- `.env.example` — document the new flag.
- `src/agent/graph.py` — add the new node; conditional routing on the flag; keep old path as default.
- `src/agent/domains/tools.py` — add `create_mcp_tools_from_catalog` as a sibling of `create_domain_tools` (scoped vs unscoped). `MCPToolWrapper` itself is unchanged.
- Docstrings at the top of `src/agent/nodes/plan.py`, `execute.py`, `evaluate.py`, `recover.py`, `synthesize.py` — add a one-liner deprecation note referencing the new node.

**Not touched:**
- `src/agent/nodes/rag_answer.py` — unchanged; produces RAG matches for both paths.
- `src/agent/nodes/domain_agent.py` — unchanged; already a tool-calling loop.
- `src/agent/nodes/classify.py` — unchanged; classifier routing runs ahead of both paths.
- `src/agent/nodes/rag_and_plan.py` — unchanged when flag is false; bypassed when flag is true (new routing doesn't enter it).
- `src/llm/providers.py` — `get_llm()` already returns a LangChain `BaseChatModel` that supports `.bind_tools()`; no changes needed.
- `src/agent/state.py` — the existing `AgentState` fields are a superset of what the new node needs.

---

## Task 1: Add the `USE_TOOL_CALLING_LOOP` feature flag

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Add the setting to `Settings`**

Open `src/config.py`. Find the existing block where `READ_ONLY` was added (just after `DISABLED_CAPABILITIES`). Add the new flag immediately after `READ_ONLY`:

```python
    # READ_ONLY: when True, force-disables all write-capable capabilities
    # (manage_announcements, open_ticket, report_login_problem, report_security)
    # by adding them to the disabled set at registry build time. Intended for
    # staging environments, the smoke test window, and any deploy where
    # inadvertent writes would be unacceptable. Overrides nothing else.
    READ_ONLY: bool = False
    # USE_TOOL_CALLING_LOOP: when True, routes tool-using queries through the
    # single-node `tool_calling_loop` (LLM-driven react-style loop) instead of
    # the legacy plan → execute → evaluate → recover → synthesize chain. The
    # old path remains functional when False so we can A/B them in Phase 7
    # evidence collection. Default False during development; flipped True
    # in staging first, then production at cutover.
    USE_TOOL_CALLING_LOOP: bool = False
```

- [ ] **Step 2: Document in `.env.example`**

Find the capability-registry block added during Phase 1 (search for `READ_ONLY=false`). Add immediately after:

```bash
# USE_TOOL_CALLING_LOOP=true routes tool-using queries through the new
# single-node tool_calling_loop (LLM-driven react loop) instead of the
# legacy plan+execute+evaluate+recover+synthesize chain. Default false;
# flip true in staging before cutover. See launch spec Phase 3.
# USE_TOOL_CALLING_LOOP=false
```

- [ ] **Step 3: Commit**

```bash
git add src/config.py .env.example
git commit -m "feat(config): add USE_TOOL_CALLING_LOOP feature flag (Phase 3)"
```

Pre-commit hooks should pass — simple additions, no behavior change.

---

## Task 2: Create the prompts module for the loop

**Files:**
- Create: `src/agent/prompts/tool_calling_loop.py`
- Create: `src/agent/prompts/__init__.py` (if doesn't exist)

The loop's system prompt is the place where the agent's identity, constraints, and optional RAG context merge. Isolating it in a dedicated module makes it reviewable, testable, and easy to iterate on without touching the node logic.

- [ ] **Step 1: Check if `src/agent/prompts/` already exists**

Run:
```bash
ls src/agent/prompts/ 2>&1 || echo "does not exist"
```

If it doesn't exist, create the directory and an empty `__init__.py`:
```bash
mkdir -p src/agent/prompts
touch src/agent/prompts/__init__.py
```

If `src/agent/prompts/` already exists, check whether it has existing content and skip creating `__init__.py`.

- [ ] **Step 2: Create `src/agent/prompts/tool_calling_loop.py`**

Write this file with the following content:

```python
"""System prompt assembly for the tool_calling_loop node (launch Phase 3).

The loop's system prompt weaves together:
  - the agent's general identity and behavioral instructions
  - optional RAG context (when rag_answer produced matches)
  - optional domain hint (when classifier identified a specific domain)
  - optional acting-user identity (when the request is authenticated)

Kept separate from the node logic so prompt iteration is a targeted PR that
doesn't invalidate review of the orchestration code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import RAGMatch


SYSTEM_IDENTITY = """You are the ACCESS-CI assistant. You help US researchers \
understand and use ACCESS-CI — a federally funded program that allocates \
computing resources (supercomputers, cloud, storage) to researchers.

Your job in this conversation: answer the user's question using the tools \
provided, along with any reference context below. Call tools when you need \
live data. Do not invent data you can't verify. If a tool fails or returns \
unexpected output, try a different approach or tell the user what you tried \
and what failed — do not silently drop the failure.

When you produce your final answer:
- Cite specific resources or facts you retrieved. Link to official ACCESS-CI \
pages where relevant.
- Be concise. Researchers want the answer, not ceremony.
- If the answer depends on the user's specific situation (allocations, \
account state), say so clearly and explain how they can check.
- If you genuinely cannot answer, say that and point the user to the support \
ticket path at https://support.access-ci.org/open-a-ticket."""


def build_system_prompt(
    rag_context: str | None = None,
    domain_hint: str | None = None,
    acting_user: str | None = None,
) -> str:
    """Assemble the loop's system prompt from optional context fragments.

    Args:
        rag_context: Formatted text block of RAG matches from rag_answer_node.
            If present, inserted as a "Reference context" section.
        domain_hint: Classifier's domain guess (e.g. "jsm", "announcements").
            If present, inserted as a "Likely domain" hint the LLM can override.
        acting_user: ACCESS ID of the authenticated requester. If present,
            inserted so the LLM can personalize allocation/usage queries.

    Returns:
        Complete system prompt string. Always begins with SYSTEM_IDENTITY.
    """
    sections: list[str] = [SYSTEM_IDENTITY]

    if rag_context:
        sections.append(
            "## Reference context (from retrieval)\n\n"
            "The following is background material retrieved for this query. "
            "Use it as reference; you still need to call tools for live data.\n\n"
            f"{rag_context}"
        )

    if domain_hint:
        sections.append(
            f"## Classifier hint\n\n"
            f"The classifier identified this query's domain as `{domain_hint}`. "
            f"You may override if the user's intent suggests otherwise."
        )

    if acting_user:
        sections.append(
            f"## Acting user\n\nCurrent user: `{acting_user}`. "
            f"Use this when calling tools that accept an acting-user identity "
            f"(allocations, ticket-creation, personal usage queries)."
        )
    else:
        sections.append(
            "## Acting user\n\n"
            "The user is anonymous (not logged in). Tools that require identity "
            "will return auth errors — handle those gracefully and suggest login "
            "at https://access-ci.org/sign-in."
        )

    return "\n\n".join(sections)


def format_rag_matches(matches: list[RAGMatch]) -> str:
    """Render a list of RAGMatch objects as a prompt-ready text block.

    Args:
        matches: List of RAGMatch instances from src.agent.state.

    Returns:
        Markdown-formatted text block, ready to pass to build_system_prompt's
        rag_context argument. Empty string when matches is empty.
    """
    if not matches:
        return ""

    rendered: list[str] = []
    for i, match in enumerate(matches, start=1):
        block = f"### Match {i} (score: {match.similarity_score:.2f})"
        block += f"\n\n**Q:** {match.question}\n\n**A:** {match.answer}"
        block += f"\n\n*Source:* {match.domain}/{match.entity_id}"
        rendered.append(block)

    return "\n\n".join(rendered)
```

**Note on field names:** `RAGMatch` (in `src/agent/state.py`) has the fields `id`, `question`, `answer`, `domain`, `entity_id`, `similarity_score`, and `metadata` — there is no `source` or `score` field. The snippet above uses the real field names. `RAGMatch` is imported under `TYPE_CHECKING` because the annotation is the only reference; a runtime import would create an import cycle with `state.py`.

- [ ] **Step 3: Commit**

```bash
git add src/agent/prompts/
git commit -m "feat(prompts): add system-prompt assembly for tool_calling_loop (Phase 3)"
```

---

## Task 3: Write the failing test suite for `tool_calling_loop_node`

**Files:**
- Create: `tests/test_tool_calling_loop.py`

We start with failing tests so the implementation is driven by concrete requirements. These tests mock the LLM and tool execution layer so they run in CI (no API keys needed), matching the style of `test_auth_e2e.py` and `test_capabilities_read_only.py`.

- [ ] **Step 1: Write the test file**

Create `tests/test_tool_calling_loop.py` with:

```python
"""Tests for tool_calling_loop_node (launch Phase 3).

Strategy: mock the LLM + tools at the create_react_agent boundary so the
node's orchestration logic is tested without live API calls.

The suite grew beyond the 7 behaviors originally scoped as corner cases
surfaced during implementation (tool_results back-fill, None-vs-empty-string
final_answer distinction, GraphRecursionError handling, orphan ToolMessages,
flag-driven routing). Final shape — 18 tests — broken out below:

Core orchestration (8):
1. Node exists and is importable.
2. Basic flow: query in → LLM picks no tools → direct answer out.
3. Tool-calling flow: LLM emits one tool call → tool runs → LLM produces answer.
4. Tool failure: LLM sees a failed tool result and recovers gracefully.
5. No tools available: node handles empty tool_catalog without crashing.
6. Message threading: caller's messages are preserved in output state.
7. System prompt: includes acting_user when authenticated.
8. System prompt: includes rag_context when rag_matches is non-empty.

tool_results back-fill (1):
9. state.tool_results is populated from ToolMessage content (eval-scorer parity).

Hardening / edge cases (3):
10. final_answer is None (not "") when loop emits no AIMessage content.
11. GraphRecursionError caught → user-facing apology, input messages preserved.
12. Orphan ToolMessages are counted (but dropped from tool_results).

Routing / graph structure (6):
13. tool_calling_loop node is registered regardless of flag state.
14. route_after_rag returns "tool_calling_loop" when flag is on.
15. route_after_rag returns "plan" when flag is off (legacy preserved).
16. route_after_rag still ends on confident-static answers when flag is on.
17. route_by_classification forces rag_answer for combined/dynamic when flag is on.
18. route_by_classification preserves rag_and_plan for combined/dynamic when flag is off.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@pytest.fixture
def base_state():
    """Minimal AgentState fields the node consumes."""
    return {
        "messages": [HumanMessage(content="what resources have GPUs?")],
        "query": "what resources have GPUs?",
        "tool_catalog": {
            "servers": [
                {
                    "server": "compute-resources",
                    "tools": [
                        {
                            "name": "search_resources",
                            "description": "Search for compute resources",
                            "inputSchema": {
                                "properties": {"has_gpu": {"type": "boolean"}},
                                "required": [],
                            },
                        }
                    ],
                }
            ]
        },
        "acting_user": None,
        "rag_matches": [],
        "query_classification": None,
    }


@pytest.mark.asyncio
async def test_node_exists_and_is_importable():
    """Sanity: the node function exists at the expected path."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    assert callable(tool_calling_loop_node)


@pytest.mark.asyncio
async def test_direct_answer_no_tools_called(base_state):
    """LLM that returns a final answer immediately produces final_answer in state."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    final_message = AIMessage(content="ACCESS has several GPU resources: Delta, FASTER, ...")

    # Mock create_react_agent to return a compiled-graph-like object whose
    # ainvoke returns the expected messages structure.
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [
            *base_state["messages"],
            final_message,
        ]
    }

    with patch(
        "src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph
    ):
        result = await tool_calling_loop_node(base_state)

    assert result["final_answer"] == final_message.content
    assert any(isinstance(m, AIMessage) for m in result["messages"])
    assert result.get("tools_used") == []


@pytest.mark.asyncio
async def test_single_tool_call_then_answer(base_state):
    """LLM calls one tool, sees result, produces answer — tools_used tracks the call."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[
            {"id": "call_1", "name": "search_resources", "args": {"has_gpu": True}}
        ],
    )
    tool_result_msg = ToolMessage(
        content='{"resources": [{"name": "Delta"}, {"name": "FASTER"}]}',
        tool_call_id="call_1",
    )
    final_msg = AIMessage(content="ACCESS has GPU resources including Delta and FASTER.")

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [*base_state["messages"], tool_call_msg, tool_result_msg, final_msg]
    }

    with patch(
        "src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph
    ):
        result = await tool_calling_loop_node(base_state)

    assert result["final_answer"] == final_msg.content
    assert "search_resources" in result["tools_used"]


@pytest.mark.asyncio
async def test_tool_failure_is_recovered_or_reported(base_state):
    """A failed tool result should not crash the node; LLM's response is returned."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    tool_call = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "search_resources", "args": {}}],
    )
    failure = ToolMessage(
        content='{"error": "timeout contacting compute-resources MCP server"}',
        tool_call_id="c1",
    )
    recovery_answer = AIMessage(
        content="I wasn't able to query the live resource list. You can see the "
        "current list at https://access-ci.org/resources."
    )

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [*base_state["messages"], tool_call, failure, recovery_answer]
    }

    with patch(
        "src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph
    ):
        result = await tool_calling_loop_node(base_state)

    assert result["final_answer"] == recovery_answer.content
    # tools_used still lists the attempted call so traces are honest
    assert "search_resources" in result["tools_used"]


@pytest.mark.asyncio
async def test_empty_tool_catalog_still_produces_answer(base_state):
    """Node must not crash when no tools are available."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    state = {**base_state, "tool_catalog": {"servers": []}}
    answer = AIMessage(content="I don't have live tools available right now, but ...")

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [*state["messages"], answer]}

    with patch(
        "src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph
    ):
        result = await tool_calling_loop_node(state)

    assert result["final_answer"] == answer.content
    assert result.get("tools_used") == []


@pytest.mark.asyncio
async def test_messages_are_accumulated_not_replaced(base_state):
    """The node must return the full message history (caller-supplied + loop-generated)."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    final = AIMessage(content="ok")
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [*base_state["messages"], final]}

    with patch(
        "src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph
    ):
        result = await tool_calling_loop_node(base_state)

    # The original HumanMessage must still be present
    assert any(
        isinstance(m, HumanMessage) and m.content == "what resources have GPUs?"
        for m in result["messages"]
    )


@pytest.mark.asyncio
async def test_system_prompt_includes_acting_user_when_authenticated(base_state):
    """When acting_user is set, the system prompt should mention it."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    state = {**base_state, "acting_user": "jsmith@access-ci.org"}
    answer = AIMessage(content="ok")
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [*state["messages"], answer]}

    captured_prompt = {}

    def capture_prompt(**kwargs):
        captured_prompt["prompt"] = kwargs.get("prompt")
        return mock_graph

    with patch(
        "src.agent.nodes.tool_calling_loop.create_react_agent", side_effect=capture_prompt
    ):
        await tool_calling_loop_node(state)

    assert "jsmith@access-ci.org" in captured_prompt["prompt"]


@pytest.mark.asyncio
async def test_system_prompt_includes_rag_context_when_present(base_state):
    """When rag_matches is non-empty, the system prompt should embed them."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node
    from src.agent.state import RAGMatch

    state = {
        **base_state,
        "rag_matches": [
            RAGMatch(
                id="rag_001",
                question="What GPUs exist?",
                answer="Delta has NVIDIA A100s.",
                domain="compute-resources",
                entity_id="delta",
                similarity_score=0.92,
            )
        ],
    }
    answer = AIMessage(content="ok")
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [*state["messages"], answer]}

    captured_prompt = {}

    def capture_prompt(**kwargs):
        captured_prompt["prompt"] = kwargs.get("prompt")
        return mock_graph

    with patch(
        "src.agent.nodes.tool_calling_loop.create_react_agent", side_effect=capture_prompt
    ):
        await tool_calling_loop_node(state)

    assert "Delta has NVIDIA A100s" in captured_prompt["prompt"]
```

- [ ] **Step 2: Run to confirm failures**

```bash
uv run pytest tests/test_tool_calling_loop.py -v
```

Expected: `ImportError: cannot import name 'tool_calling_loop_node' from 'src.agent.nodes.tool_calling_loop'` on the first test, and cascading failures on the rest. This is the expected TDD baseline. (At TDD time only the 8 core tests exist; the back-fill, hardening, and routing tests are added in later tasks.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_tool_calling_loop.py
git commit -m "test(tool_calling_loop): failing tests for Phase 3 node"
```

---

## Task 4: Implement `tool_calling_loop_node`

**Files:**
- Create: `src/agent/nodes/tool_calling_loop.py`

- [ ] **Step 1: Write the node**

Create `src/agent/nodes/tool_calling_loop.py`:

```python
"""Tool-calling loop node — single-node replacement for plan+execute+evaluate+recover.

Launched behind the USE_TOOL_CALLING_LOOP feature flag (launch Phase 3). When
the flag is True, tool-using queries route here instead of the legacy chain.

Approach: LangGraph's `create_react_agent` drives a turn-by-turn loop where the
LLM selects tools, sees results as ToolMessages, and continues until it emits
a non-tool-call response. Planning, execution, evaluation, and recovery all
happen inside that loop — no separate nodes needed.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from ...llm import get_llm
from ...telemetry import get_tracer
from ..domains.tools import create_mcp_tools_from_catalog
from ..prompts.tool_calling_loop import build_system_prompt, format_rag_matches
from ..state import ToolResult

logger = logging.getLogger(__name__)


async def tool_calling_loop_node(state: dict[str, Any]) -> dict[str, Any]:
    """Run the single-node tool-calling loop.

    Consumes:
        state.messages: conversation history (HumanMessage / AIMessage / ToolMessage)
        state.query: current user query (already present in messages[-1])
        state.tool_catalog: dict describing MCP tools available to this request
        state.acting_user: optional ACCESS ID for personalized tool calls
        state.rag_matches: optional list of RAGMatch objects from rag_answer
        state.query_classification: optional classifier output with .domain

    Produces:
        final_answer: LLM's final text response (None when the loop produced no AI text)
        messages: full loop message history (caller messages + tool calls + result messages + final)
        tools_used: list of tool-name strings the loop actually invoked
        tool_results: reconstructed list[ToolResult] for eval-scorer / observability parity
        node_trace: telemetry entry for this node
    """
    tracer = get_tracer("access-agent.nodes")

    with tracer.start_as_current_span(
        "agent.tool_calling_loop",
        attributes={"agent.node": "tool_calling_loop"},
    ) as span:
        acting_user = state.get("acting_user")
        rag_matches = state.get("rag_matches") or []
        classification = state.get("query_classification")
        domain_hint = classification.domain if classification else None
        tool_catalog = state.get("tool_catalog") or {}

        # Build the system prompt
        rag_context = format_rag_matches(rag_matches) if rag_matches else None
        system_prompt = build_system_prompt(
            rag_context=rag_context,
            domain_hint=domain_hint,
            acting_user=acting_user,
        )

        # Materialize MCP tools as LangChain BaseTool instances
        tools = create_mcp_tools_from_catalog(tool_catalog, acting_user)

        span.set_attribute("agent.tool_count", len(tools))
        span.set_attribute("agent.has_rag_context", bool(rag_context))
        span.set_attribute("agent.authenticated", bool(acting_user))

        # Build the react agent
        llm = get_llm()
        agent = create_react_agent(
            model=llm,
            tools=tools,
            prompt=system_prompt,
        )

        # Run the loop
        messages = list(state.get("messages", []))
        logger.info(
            "Running tool_calling_loop: %d tools, %d prior messages, "
            "rag_context=%s, domain_hint=%s",
            len(tools),
            len(messages),
            bool(rag_context),
            domain_hint,
        )

        # `None` (not "") is the signal that the loop produced no AI text
        # response — this matters to downstream consumers that need to
        # distinguish "no output" from "empty-string output".
        final_answer: str | None = None
        recursion_limit_hit = False

        # recursion_limit = 2 * max_tool_turns + 1; gives the LLM room for
        # roughly 10 tool turns before LangGraph hard-stops. Chosen to match
        # the legacy max_attempts * max_retries envelope loosely.
        recursion_limit = 25
        try:
            result = await agent.ainvoke(
                {"messages": messages},
                {"recursion_limit": recursion_limit},
            )
            result_messages = result.get("messages", [])
        except GraphRecursionError:
            # See Step 2.6: graceful-degradation wrapping. The react loop
            # exhausted its budget; return a user-facing apology rather than
            # bubble a 500 to the chatbot.
            recursion_limit_hit = True
            span.set_attribute("agent.recursion_limit_hit", True)
            result_messages = list(messages)
            final_answer = (
                "I wasn't able to complete an answer for this query within "
                "the allotted tool-turn budget. You can try rephrasing, or "
                "open a support ticket at "
                "https://support.access-ci.org/open-a-ticket."
            )

        # Extract the final text answer only when the loop ran successfully.
        # AIMessage.content is typed `str | list[str | dict]` (multimodal);
        # the loop only emits string content for final answers but we narrow
        # defensively.
        if not recursion_limit_hit:
            for msg in reversed(result_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    content = msg.content
                    final_answer = content if isinstance(content, str) else str(content)
                    break

        # Extract the set of tools the loop actually called
        tools_used: list[str] = []
        for msg in result_messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for tc in msg.tool_calls:
                    name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                    if name and name not in tools_used:
                        tools_used.append(name)

        # Count how many tool results came back (used/successful)
        tool_result_count = sum(1 for m in result_messages if isinstance(m, ToolMessage))

        # Step 2.5: back-fill tool_results for eval-scorer parity.
        tool_results, orphan_count = _build_tool_results(result_messages, tools)

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
            "tool_results": tool_results,
            "node_trace": [
                {
                    "node": "tool_calling_loop",
                    "tool_count": len(tools),
                    "tool_calls_made": len(tools_used),
                    "tool_results": tool_result_count,
                    "answer_length": answer_length,
                }
            ],
        }
```

- [ ] **Step 2: Add the catalog-to-tools helper**

The existing `create_domain_tools(config, tool_catalog, acting_user)` in `src/agent/domains/tools.py` takes a domain config and filters to that domain's servers. For the tool_calling_loop we want ALL enabled tools — not scoped to a single domain. Add a sibling function that mirrors `create_domain_tools`'s real signatures (`MCPToolWrapper` uses `mcp_client=` / `tool_server=`; `_build_args_schema(tool_name, parameters)` takes two positional args):

Open `src/agent/domains/tools.py`. Find `create_domain_tools`. After it, add:

```python
def create_mcp_tools_from_catalog(
    tool_catalog: dict[str, Any],
    acting_user: str | None = None,
) -> list[BaseTool]:
    """Create LangChain tool wrappers for every available tool in the catalog.

    Unlike create_domain_tools, this function is not scoped to a single
    domain — it materializes every enabled MCP tool so the tool_calling_loop
    can pick from the full set. Respects the capability registry via the
    tool_catalog input (caller populates the catalog from enabled
    capabilities only).

    Args:
        tool_catalog: The MCP tool catalog dict with "servers" key.
        acting_user: ACCESS ID for auth headers, or None for anonymous.

    Returns:
        List of MCPToolWrapper instances, one per available tool in the catalog.
    """
    client = MCPClient()
    tools: list[BaseTool] = []

    servers = tool_catalog.get("servers", [])
    for server_info in servers:
        server_name = server_info.get("server", "")
        if not server_name:
            continue
        # Mirror create_domain_tools: older catalogs may omit status, so we
        # treat missing status as "available" and only skip when it's
        # explicitly something else.
        status = server_info.get("status", "available")
        if status != "available":
            logger.warning(
                f"Server {server_name} unavailable (status={status}), "
                f"skipping for tool_calling_loop"
            )
            continue

        for tool_def in server_info.get("tools", []):
            tool_name = tool_def.get("name", "")
            description = tool_def.get("description", "")
            parameters = tool_def.get("parameters", [])

            args_schema = _build_args_schema(tool_name, parameters)

            wrapper = MCPToolWrapper(
                name=tool_name,
                description=description,
                args_schema=args_schema,
                tool_server=server_name,
                mcp_client=client,
                acting_user=acting_user,
            )
            tools.append(wrapper)

    return tools
```

Reuses `MCPClient`, `MCPToolWrapper`, and `_build_args_schema` that already live in that file. Note these are the shapes that actually shipped (not what the earlier draft of this plan used):
- `MCPToolWrapper` takes `mcp_client=` (not `client=`) and `tool_server=` (not `server=`).
- `_build_args_schema(tool_name, parameters)` has a two-argument signature, not a one-arg one.
- The `status != "available"` skip mirrors the existing `create_domain_tools` behavior so both helpers treat server availability consistently.

- [ ] **Step 2.5: Add `_build_tool_results` helper and populate `tool_results` return key**

Downstream consumers — in particular the eval scorer at `src/eval/scorer.py` — read `state["tool_results"]` in the same `list[ToolResult]` shape the legacy `plan+execute` path produced. The react loop only emits `AIMessage` tool_calls and `ToolMessage` results; without a back-fill, the Phase 7 old-vs-new comparison would see "no tools used" for every loop response.

Add a `_build_tool_results(result_messages, tools)` module-level helper that:

1. Builds a `tool_name → tool_server` lookup from the materialized `tools` list (so each `ToolResult` records which MCP server served the call).
2. Walks `result_messages` once to build a `tool_call_id → (tool_name, args)` lookup from every `AIMessage.tool_calls` entry.
3. Walks `result_messages` a second time, and for each `ToolMessage`:
   - Looks up the originating call by `msg.tool_call_id`.
   - Parses the content as JSON; if the parsed value is a dict with a top-level `"error"` key, emits a failed `ToolResult` (`success=False`, `error=<that value>`); otherwise emits a successful `ToolResult` (`success=True`, `data=<parsed>`).
   - Orphan `ToolMessages` (no matching `AIMessage` call) are silently dropped but counted; the helper returns `(tool_results, orphan_count)`.
4. Sets `step_id = tool_call_id` on each `ToolResult` so traces can cross-reference the call that produced it.
5. Leaves `duration_ms` at the `ToolResult` default (0). The react loop doesn't track per-call timing; the eval scorer doesn't read `duration_ms`, so this is acceptable.

Caller-side wiring:
- Call the helper immediately before building the telemetry attributes.
- When `orphan_count > 0`, set `span.set_attribute("agent.tool_results_orphaned", orphan_count)`. (Don't set it when it's 0 — keeps traces uncluttered for the healthy case.)
- Include `"tool_results": tool_results` in the return dict (already shown in the Step 1 snippet).

Import `ToolResult` at the top of the file: `from ..state import ToolResult`. Import `json` for the `json.loads` parse step.

- [ ] **Step 2.6: Wrap `agent.ainvoke` for graceful degradation**

Two failure modes need handling so the chatbot gets a renderable response instead of an HTTP 500:

1. **`langgraph.errors.GraphRecursionError`** — thrown when the react loop exhausts `recursion_limit` without the LLM producing a final answer (e.g., a pathological tool-call loop). Catch it, set `span.set_attribute("agent.recursion_limit_hit", True)`, preserve the caller's input messages as `result_messages`, and set `final_answer` to a short user-facing apology that points at `https://support.access-ci.org/open-a-ticket`.
2. **Empty `AIMessage.content` throughout the loop** — the loop terminated without the LLM emitting any textual response. Leave `final_answer` as `None` (not `""`) so downstream callers can distinguish "loop produced nothing" from "LLM emitted an empty string". `node_trace[0]["answer_length"]` uses `len(final_answer) if final_answer else 0` to avoid a `TypeError` on the `None` path.

Both behaviors are already woven into the Step 1 snippet above — call this out as a discrete sub-step so future readers understand why the `try/except` and the `Optional[str]` typing exist.

- [ ] **Step 3: Run the tests**

```bash
uv run pytest tests/test_tool_calling_loop.py -v
```

Expected: all 8 core tests PASS (the back-fill, hardening, and routing tests are added in later tasks).

If any fail, inspect the failure, fix minimally, re-run. Common likely issues:
- Mock patch path is wrong (e.g., `src.agent.nodes.tool_calling_loop.create_react_agent` vs where it's imported from). Adjust the patch target or the import in the node.
- `create_mcp_tools_from_catalog` signature mismatch with the helper you wrote. Align.
- Tests reference `RAGMatch` dataclass; verify it's importable from `src.agent.state`.

- [ ] **Step 4: Run the full non-e2e suite to confirm no regressions**

```bash
uv run pytest tests/ -q -m "not e2e"
```

Expected: previous pass count plus the 8 new core tests. Nothing else should move. (Routing and hardening tests land in Task 5; `tool_results` back-fill test is added alongside the Step 2.5 implementation.)

- [ ] **Step 5: Commit**

```bash
git add src/agent/nodes/tool_calling_loop.py src/agent/domains/tools.py
git commit -m "$(cat <<'EOF'
feat(agent): implement tool_calling_loop_node (Phase 3)

Single-node replacement for plan+execute+evaluate+recover. Uses LangGraph's
create_react_agent to run an LLM-driven tool-calling loop; planning,
execution, evaluation, and recovery happen inside the loop via the LLM's
own reasoning.

Not yet wired into the graph — gated behind USE_TOOL_CALLING_LOOP flag;
graph routing lands in the next task.

Adds create_mcp_tools_from_catalog() helper in src/agent/domains/tools.py
as a sibling of create_domain_tools() (scoped vs unscoped).

Spec: docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md §Phase 3
EOF
)"
```

---

## Task 5: Wire the feature flag into graph routing

**Files:**
- Modify: `src/agent/graph.py`

This is where the old path meets the new path. When `USE_TOOL_CALLING_LOOP=True`, tool-using queries route to `tool_calling_loop_node` and bypass `plan → execute → evaluate → recover → synthesize`. When False, the existing graph is unchanged.

- [ ] **Step 1: Register the new node in the graph**

Open `src/agent/graph.py`. Find where existing nodes are added (likely a `.add_node("plan", plan_node)` sequence). Add the new node. Import it at the top:

```python
from .nodes.tool_calling_loop import tool_calling_loop_node
```

In the graph builder (search for `StateGraph(AgentState)` or similar):

```python
workflow.add_node("tool_calling_loop", tool_calling_loop_node)
```

Do this unconditionally — the node is registered, but whether the graph routes to it is controlled by the flag below.

- [ ] **Step 2: Add the flag-aware router**

Find `route_after_rag` (from the architecture map, around graph.py:146). Its current logic has **four distinct `return "plan"` sites**:

1. Disabled-domain fallback — `if not get_capability_registry().is_domain_enabled(domain)` after a domain was detected.
2. Combined/dynamic with matches — `query_type in ("combined", "dynamic")` and `rag_matches` is non-empty.
3. Combined/dynamic without matches — same `query_type` check, empty `rag_matches`.
4. Static deflection fallback — `_rag_answer_is_deflection(final_answer)` is true.
5. Static no-match fallback — no `final_answer` returned by RAG at all.

(The Literal return annotation plus four/five logical branches is the reality. Earlier drafts of this plan showed a single `if settings.USE_TOOL_CALLING_LOOP: return "tool_calling_loop"` appended at the end — that doesn't reach all the sites.)

Pick a `tool_path` local once at the top, and substitute it at every `return "plan"` site. This keeps the function readable and guarantees the flag flips all legacy-path returns together:

```python
def route_after_rag(
    state: AgentState,
) -> Literal["end", "plan", "tool_calling_loop", "domain_agent"]:
    """Decide next node after rag_answer produces matches.

    When USE_TOOL_CALLING_LOOP=true, every legacy "plan" branch routes to
    "tool_calling_loop" instead. The "end" and "domain_agent" paths are
    unaffected.
    """
    # Single point-of-substitution for the four-five legacy "plan" returns.
    tool_path: Literal["plan", "tool_calling_loop"] = (
        "tool_calling_loop" if settings.USE_TOOL_CALLING_LOOP else "plan"
    )

    classification = state.get("query_classification")
    query_type = classification.query_type if classification else "static"
    domain = classification.domain if classification else None

    # Domain branch (unchanged): domain enabled → domain_agent; disabled → tool_path
    if domain:
        from .domains.capabilities import get_capability_registry
        if not get_capability_registry().is_domain_enabled(domain):
            return tool_path
        return "domain_agent"

    # Combined/dynamic: tools either way — uses tool_path
    if query_type in ("combined", "dynamic"):
        return tool_path

    # Static: end if confident, tool_path if deflection or no match
    final_answer = state.get("final_answer")
    if final_answer:
        if _rag_answer_is_deflection(final_answer):
            return tool_path
        return "end"
    return tool_path
```

Two related changes land with this refactor:

1. The `Literal` return-type annotation widens from `Literal["end", "plan", "domain_agent"]` to `Literal["end", "plan", "tool_calling_loop", "domain_agent"]`.
2. The `_build_graph_structure` conditional-edge mapping for `rag_answer` must include the new target:

```python
builder.add_conditional_edges(
    "rag_answer",
    route_after_rag,
    {
        "end": END,
        "plan": "plan",
        "tool_calling_loop": "tool_calling_loop",
        "domain_agent": "domain_agent",
    },
)
```

Add import of settings at the top of `graph.py` if it isn't already imported:

```python
from ..config import settings
```

Preserve all existing behavior for the false path — the `tool_path` local evaluates to `"plan"` when the flag is off, so every legacy branch returns the same value it did before.

- [ ] **Step 3: Also update the combined/dynamic path router**

Find where `classify` routes to `rag_and_plan` (the combined/dynamic path). Currently `classify → rag_and_plan → execute → ...`. When the flag is true, the combined path should route `classify → rag_answer → tool_calling_loop` (the loop handles tool selection inline after seeing RAG context). Since `rag_answer` already runs for both paths when the flag is true, the `rag_and_plan` node is bypassed.

Find `route_by_classification`:

```python
def route_by_classification(state: AgentState) -> str:
    """Decide initial path based on classifier output."""
    classification = state.get("query_classification")

    if settings.USE_TOOL_CALLING_LOOP:
        # New path: always go through rag_answer first; tool_calling_loop
        # handles combined/dynamic cases using RAG context + tools.
        return "rag_answer"

    # Legacy behavior preserved below
    if classification and classification.query_type in ("combined", "dynamic"):
        return "rag_and_plan"
    return "rag_answer"
```

- [ ] **Step 4: Add the edge `tool_calling_loop → END`**

Right after the `workflow.add_node("tool_calling_loop", ...)` line:

```python
workflow.add_edge("tool_calling_loop", END)
```

The loop terminates with a final answer; no synthesize step needed on this path.

- [ ] **Step 5: Test the flag toggles between paths**

Add tests at the bottom of `tests/test_tool_calling_loop.py`. The factory function exported by `src/agent/graph.py` is `create_agent_graph` (earlier plan drafts called it `build_graph` — that name doesn't exist). The compiled-graph introspection API on LangGraph 1.x is `graph.get_graph().nodes`, not `graph.nodes`.

Rather than walk the graph structure, the shipped tests exercise the router functions directly (simpler and more targeted), plus one sanity-check that the node is registered:

```python
@pytest.mark.asyncio
async def test_graph_registers_tool_calling_loop_node():
    """The tool_calling_loop node is registered regardless of flag state."""
    from src.agent.graph import create_agent_graph

    graph = create_agent_graph()
    graph_obj = graph.get_graph()
    assert "tool_calling_loop" in graph_obj.nodes


@pytest.mark.asyncio
async def test_route_after_rag_uses_tool_calling_loop_when_flag_on(monkeypatch):
    """With flag on, route_after_rag returns tool_calling_loop where it would have returned plan."""
    from src.agent.graph import route_after_rag

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", True)

    # No classification, no final_answer: hits the "no UKY match, fall back" branch
    state = {"query_classification": None, "final_answer": None}
    assert route_after_rag(state) == "tool_calling_loop"


@pytest.mark.asyncio
async def test_route_after_rag_preserves_plan_when_flag_off(monkeypatch):
    """With flag off (default), route_after_rag returns plan as before."""
    from src.agent.graph import route_after_rag

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", False)

    state = {"query_classification": None, "final_answer": None}
    assert route_after_rag(state) == "plan"


@pytest.mark.asyncio
async def test_route_after_rag_still_ends_on_confident_static_when_flag_on(monkeypatch):
    """Flag doesn't change the 'confident RAG → END' decision."""
    from src.agent.graph import route_after_rag

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", True)

    state = {
        "query_classification": None,
        "final_answer": (
            "ACCESS has multiple GPU resources. "
            "See https://access-ci.org/resources for the full list."
        ),
    }
    assert route_after_rag(state) == "end"


@pytest.mark.asyncio
async def test_route_by_classification_forces_rag_answer_when_flag_on(monkeypatch):
    """With flag on, combined/dynamic queries route to rag_answer so the loop can consume RAG context."""
    from src.agent.graph import route_by_classification
    from src.agent.state import QueryClassification

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", True)

    state = {"query_classification": QueryClassification(query_type="combined")}
    assert route_by_classification(state) == "rag_answer"


@pytest.mark.asyncio
async def test_route_by_classification_preserves_rag_and_plan_when_flag_off(monkeypatch):
    """Default path unchanged: combined/dynamic → rag_and_plan."""
    from src.agent.graph import route_by_classification
    from src.agent.state import QueryClassification

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", False)

    state = {"query_classification": QueryClassification(query_type="combined")}
    assert route_by_classification(state) == "rag_and_plan"
```

- [ ] **Step 6: Run the tests**

```bash
uv run pytest tests/test_tool_calling_loop.py -v
```

Expected: all tests PASS. If the node-registration test fails because of an unexpected introspection API, inspect a compiled graph interactively:

```bash
uv run python -c "from src.agent.graph import create_agent_graph; g = create_agent_graph(); print(g.get_graph().nodes)"
```

On LangGraph 1.x the compiled graph exposes `.get_graph().nodes` (not `.nodes` directly). Adjust the assertion if the API has shifted.

- [ ] **Step 7: Run the full non-e2e suite**

```bash
uv run pytest tests/ -q -m "not e2e"
```

Expected: no regressions.

- [ ] **Step 8: Commit**

```bash
git add src/agent/graph.py tests/test_tool_calling_loop.py
git commit -m "$(cat <<'EOF'
feat(graph): route to tool_calling_loop when USE_TOOL_CALLING_LOOP=true (Phase 3)

Adds tool_calling_loop as a graph node and branches on the feature flag:
- flag=true: classify → rag_answer → tool_calling_loop → END (new path)
- flag=false: unchanged legacy path (plan → execute → evaluate → recover → synthesize)

Domain-agent branch (rag_answer → domain_agent → END) is unaffected by
the flag — it was already a tool-calling loop for its scoped use case.

Spec: docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md §Phase 3
EOF
)"
```

---

## Task 6: Mark the legacy nodes deprecated

**Files:**
- Modify: `src/agent/nodes/plan.py`
- Modify: `src/agent/nodes/execute.py`
- Modify: `src/agent/nodes/evaluate.py`
- Modify: `src/agent/nodes/recover.py`
- Modify: `src/agent/nodes/synthesize.py`

Deprecation comments keep the code running (spec says "old path remains functional") but make the replacement story obvious to any future reader who lands in these files.

- [ ] **Step 1: Add a deprecation note to each file's module docstring**

For each of the five files, prepend the existing module docstring with a deprecation block. Pattern (for `plan.py`):

```python
"""Plan node — DEPRECATED: superseded by tool_calling_loop_node (Phase 3).

This node runs only when USE_TOOL_CALLING_LOOP=false. The new path in
src/agent/nodes/tool_calling_loop.py folds planning, execution, evaluation,
and recovery into a single LLM-driven loop. See
docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md §Phase 3.

Retained for rollback safety until the feature-flag cutover is complete.

--- Original docstring below ---

<existing content>
"""
```

Apply the same shape to `execute.py`, `evaluate.py`, `recover.py`, `synthesize.py`. Keep each file's original docstring body intact after the deprecation block.

- [ ] **Step 2: Commit**

```bash
git add src/agent/nodes/plan.py src/agent/nodes/execute.py src/agent/nodes/evaluate.py src/agent/nodes/recover.py src/agent/nodes/synthesize.py
git commit -m "docs(agent): mark plan/execute/evaluate/recover/synthesize deprecated (Phase 3)

Old nodes remain functional when USE_TOOL_CALLING_LOOP=false. Slated for
deletion after the feature-flag cutover lands in production (launch Phase 8)."
```

---

## Task 7: Eval-battery parity check

**Files:** none (smoke + evidence task).

This task runs the existing eval battery on both paths and confirms the new path is at least as good as the old one. It's the single biggest risk in Phase 3 — a successful code path that returns worse answers is a launch-blocker.

- [ ] **Step 1: Confirm the local eval runner works**

Run a small battery first to verify the runner itself:

```bash
cd /Users/josephbacal/Projects/sweet-and-fizzy/access-ci/access-agent
source a3_results/.venv/bin/activate 2>/dev/null || true
# exact command depends on the eval harness; check existing a3 runs
ls a3_results/ 2>&1 | head
```

Use whatever command Joe used for the most recent a3 run as the template. If unclear, run `uv run python -m src.eval --help` and pick the subcommand that executes a battery against the local agent.

- [ ] **Step 2: Run baseline (flag off)**

```bash
USE_TOOL_CALLING_LOOP=false uv run python -m src.eval run \
  --battery combined \
  --session phase3-baseline-off
```

Wait for completion. Record pass rate, mean latency, any obvious failures. Save outputs under `a3_results/phase3-baseline-off/`.

- [ ] **Step 3: Run new path (flag on)**

```bash
USE_TOOL_CALLING_LOOP=true uv run python -m src.eval run \
  --battery combined \
  --session phase3-loop-on
```

Record same metrics. Save outputs under `a3_results/phase3-loop-on/`.

- [ ] **Step 4: Produce the side-by-side comparison**

Use the existing comparison-report infrastructure (CLAUDE.md documents this — search for "A.3 Bake-Off HTML Comparison Report"). Adapt:

```bash
cd access-agent
docker compose exec -T postgres psql -U postgres -d langgraph -c \
  "COPY (SELECT row_to_json(r) FROM rag_comparison_logs r WHERE session_id IN ('phase3-baseline-off', 'phase3-loop-on') ORDER BY id) TO STDOUT" \
  2>/dev/null > ../phase3_comparison.json
```

Then adapt the `visual-explainer` skill or `a3-run4-comparison.html` format into `phase3-comparison.html`.

- [ ] **Step 5: Triage any regressions**

For each question where the new path scores lower than the old path, open the row in the HTML report and inspect:

- Is the LLM failing to call a tool it should? → tune system prompt.
- Is the LLM calling a tool correctly but interpreting the result wrong? → probably no fix here; may be LLM-model-specific and resolve when Phase 5 swaps in UKY's model.
- Is a tool failing differently in the loop vs the old path? → investigate tool wrapper (`create_mcp_tools_from_catalog`).
- Is the answer materially the same but scored lower by the judge? → judge-calibration issue, not a launch blocker.

Fix what's fixable in-place; document what's not and surface to Andrew / Drew for the Phase 7 evidence review.

- [ ] **Step 6: Commit the comparison report + notes**

```bash
git add a3_results/phase3-*/  # whatever the eval runner produced
git commit -m "eval(phase3): baseline-vs-loop comparison results"
```

If the comparison is decisive (loop clearly equal-or-better), note that in the commit message. If there's a notable regression pattern, link to where it's documented.

---

## Task 8: Deprecation cleanup for the `$step_N` resolver and retry bookkeeping

**Files:**
- Modify: `src/agent/nodes/execute.py` (add deprecation note to `_resolve_reference`, `_resolve_parameters`)

The `$step_N` resolver is the most hand-rolled bit of the old path. It retires with the old path but needs a deprecation note so future readers understand its scope.

- [ ] **Step 1: Mark the resolver functions**

In `src/agent/nodes/execute.py`, find `_resolve_parameters` and `_resolve_reference`. Add a short comment above each:

```python
# DEPRECATED (Phase 3): used only by the legacy plan→execute path when
# USE_TOOL_CALLING_LOOP=false. The new tool_calling_loop does not need
# $step_N substitution — tool outputs flow via ToolMessage content and
# the LLM reads actual values on subsequent turns. Delete alongside
# the legacy path during the post-launch cleanup.
def _resolve_parameters(...):
    ...
```

Same for `_resolve_reference`.

- [ ] **Step 2: Commit**

```bash
git add src/agent/nodes/execute.py
git commit -m "docs(agent): mark \$step_N resolver deprecated (Phase 3)"
```

---

## Execution notes

**Expected total time:** 4-8 hours of focused work for a subagent-driven implementer. Eval battery runs (Task 7) are the longest single step — plan an hour for the battery + analysis.

**Rollback strategy:** every task is a separate commit; if a task's changes break the build, revert just that commit. The feature flag ensures the old path keeps working even if the new node has bugs — an accidental production rollout with a broken loop just means users silently keep hitting the legacy path.

**Known unknowns (raised at plan time; resolve during execution):**
1. ~~LangGraph 1.x compiled-graph introspection API for Task 5 Step 5 — adjust tests against actual API.~~ **Resolved** (commit `b8b9153`): the API is `graph.get_graph().nodes`. Tests use that form.
2. Exact eval-runner subcommand for Task 7 Step 2 — pattern-match on Joe's most recent a3 run.
3. Whether the loop's recursion_limit of 25 is too low for some multi-tool queries — adjust upward based on eval results.

**Out of scope (post-launch):**
- Deleting the legacy nodes (Phase 8 cleanup).
- Switching to `langchain.agents.create_agent` (the LangChain 1.x umbrella's newer entry point) — requires installing the `langchain` umbrella package, which is not currently a dep.
- Explicit retry/budget accounting in the loop (currently relies on `recursion_limit` only).
- Tool-result summarization/compression for long-context-window management (the old path had `SYNTHESIS_TOKEN_BUDGET` logic; the new path trusts the LLM's context window).

---

## Implementation notes — deviations from plan

This plan was authored before the implementation began; several snippets referenced APIs and field names that differed from reality. Snippets in the task sections above have been corrected in-place. The following substantive deviations are worth calling out:

**Additions beyond the original plan:**
- **Task 4 back-fill of `state["tool_results"]`** (commit `251def1`) — the eval scorer at `src/eval/scorer.py` reads `state["tool_results"]`; without back-fill, Phase 7's old-vs-new comparison would have seen "no tools used" for every loop response. Added as Step 2.5 in this plan post-hoc.
- **Task 4 graceful-degradation wrapping** (commit `92ae941`) — `GraphRecursionError` and empty-AI-message-content cases now produce a user-facing response rather than propagating an HTTP 500 or returning an empty string. Surfaced in the branch-wide final review.

**Snippet-level corrections applied above:**
- `MCPToolWrapper` field names: `mcp_client=` (not `client=`), `tool_server=` (not `server=`); `_build_args_schema(name, parameters)` takes two positional args.
- `RAGMatch` field names: `entity_id`, `domain`, `similarity_score` (not `source`, `score`).
- `route_after_rag` has four `return "plan"` sites (not two) — refactored via a shared `tool_path` local in the shipped code.
- Graph factory function is `create_agent_graph` (not `build_graph`); compiled-graph introspection is via `graph.get_graph().nodes`.

**Files that were modified but listed as "Not touched":**
- `src/agent/domains/tools.py` — received `create_mcp_tools_from_catalog`, a sibling of `create_domain_tools`.

**Test count:** the suite in `tests/test_tool_calling_loop.py` grew to 18 tests (8 core scenarios + 1 `tool_results` back-fill + 6 routing/graph-structure + 3 hardening).

**Commits on this branch:** `4a9d9d2` (flag) → `64b9da5` (prompts) → `1b3dcca`/`acbf91d` (tests + field-name fix) → `f8a5278` (node) → `b4ac03f` (prompts field-name fix) → `251def1` (tool_results back-fill) → `b8b9153` (graph wiring) → `de637d1`/`617f66c` (deprecation docs) → `92ae941` (pre-Phase-7 hardening).
