# Turn Timing Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record per-tool-call and per-model-call wall-clock timing from inside the agent run and write it to both Honeycomb spans and the turn report, plus store the turn's OTEL trace id on each `turn_reports` row.

**Architecture:** Timings are captured where the calls happen (the MCP tool wrapper, the doc-search tool, the loop's LLM callback) and carried to the report writer via the existing `turn_capture` ContextVar side-channel and the loop's node return. `MCPClient.call_tool` gains a span via the existing-but-unused `trace_mcp_call` helper. No new I/O on the response path — timing is clock reads; span export stays batched; the report write already runs off-path.

**Tech Stack:** Python 3.12, LangChain/LangGraph, OpenTelemetry SDK, SQLAlchemy, pytest (`uv run pytest`).

**Spec:** `../../../../access-agent-reporting/docs/specs/2026-06-12-turn-timing-and-charts-design.md`

**Repo / branch:** `access-agent`, new branch `feat/turn-timing` off up-to-date `main`.

**Background reading for the implementer (skim before starting):**
- `src/agent/turn_capture.py` — the ContextVar side-channel pattern. Key invariant: mutate the dict **in place**, never reassign the ContextVar, so writes from LangGraph's child asyncio tasks stay visible.
- `src/turn_reporter.py` — `_assemble_turn_report` is a pure function turning `final_state` + `turn_capture` into row dicts; `_migrate_columns` adds columns to existing deployments (no migration framework).
- `src/agent/nodes/tool_calling_loop.py` — `_build_tool_results` reconstructs `ToolResult` objects from LangChain ToolMessages after the loop ends; this is where `duration_ms` currently dies (the docstring says so).
- Lesson from a prior bug: a key returned by a LangGraph node is **silently dropped** unless it's declared as a channel in `AgentState` (`src/agent/state.py`). Task 5 declares `model_calls` — do not skip that file.

---

### Task 0: Branch setup

**Files:** none

- [ ] **Step 1: Create branch**

```bash
cd /Users/josephbacal/Projects/sweet-and-fizzy/access-ci/access-agent
git checkout main && git pull origin main
git checkout -b feat/turn-timing
```

- [ ] **Step 2: Verify clean test baseline**

Run: `uv run pytest tests/ -x -q -m "not e2e"`
Expected: all pass (CI runs this exact command).

---

### Task 1: `turn_capture` — tool timings + trace id

**Files:**
- Modify: `src/agent/turn_capture.py`
- Test: `tests/test_turn_capture.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_turn_capture.py` (it already imports `get_turn_capture`, `reset_turn_capture`; extend the import to add `record_tool_timing`, `record_trace_id`):

```python
def test_records_tool_timings_in_order():
    reset_turn_capture()
    record_tool_timing("list_things", 42)
    record_tool_timing("search_access_documents", 7)
    cap = get_turn_capture()
    assert cap["tool_timings"] == [
        {"tool_name": "list_things", "duration_ms": 42},
        {"tool_name": "search_access_documents", "duration_ms": 7},
    ]


def test_records_trace_id():
    reset_turn_capture()
    record_trace_id("ab" * 16)
    assert get_turn_capture()["trace_id"] == "ab" * 16


def test_tool_timing_safe_without_reset():
    async def _isolated():
        record_tool_timing("x", 1)
        record_trace_id("ff" * 16)
        return get_turn_capture()

    cap = asyncio.run(_isolated())
    assert cap["tool_timings"] == [] and cap["trace_id"] is None
```

Also update the existing `test_default_capture_is_safe_without_reset` — its exact-dict assertion must include the two new keys:

```python
    assert cap == {
        "searched": False,
        "chunks": [],
        "summarized": False,
        "tool_timings": [],
        "trace_id": None,
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_turn_capture.py -v`
Expected: FAIL — `ImportError: cannot import name 'record_tool_timing'`

- [ ] **Step 3: Implement**

In `src/agent/turn_capture.py`, replace `reset_turn_capture` and `get_turn_capture`, and add the two recorders:

```python
def reset_turn_capture() -> None:
    """Start a fresh capture for the current turn (call once per request)."""
    _turn_capture.set(
        {
            "searched": False,
            "chunks": [],
            "summarized": False,
            "tool_timings": [],
            "trace_id": None,
        }
    )


def record_tool_timing(tool_name: str, duration_ms: int) -> None:
    """Record one tool invocation's wall-clock duration.

    Appended in completion order; the loop pairs these back to reconstructed
    ToolResults FIFO per tool name (see _build_tool_results).
    """
    cap = _turn_capture.get()
    if cap is None:
        return
    cap.setdefault("tool_timings", []).append(
        {"tool_name": tool_name, "duration_ms": int(duration_ms)}
    )


def record_trace_id(trace_id: str) -> None:
    """Record the OTEL trace id (32-hex) of the turn's root span."""
    cap = _turn_capture.get()
    if cap is not None:
        cap["trace_id"] = trace_id
```

And in `get_turn_capture`, extend the safe default:

```python
def get_turn_capture() -> dict[str, Any]:
    """Read the current turn's capture (safe default if never reset)."""
    return _turn_capture.get() or {
        "searched": False,
        "chunks": [],
        "summarized": False,
        "tool_timings": [],
        "trace_id": None,
    }
```

Update the module docstring's first paragraph: it currently says the reporting layer needs "two facts"; reword to cover timings and trace id, e.g. append: "It also carries per-tool-call durations (recorded by the MCP tool wrapper and doc-search tool) and the turn's OTEL trace id (recorded at the root span)."

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_turn_capture.py -v`
Expected: PASS (all, including the updated default-shape test)

- [ ] **Step 5: Commit**

```bash
git add src/agent/turn_capture.py tests/test_turn_capture.py
git commit -m "feat: turn_capture carries tool timings and trace id"
```

---

### Task 2: MCP tool wrapper records timing

**Files:**
- Modify: `src/agent/domains/tools.py:83-99` (`MCPToolWrapper._arun`)
- Test: `tests/test_tool_timing.py` (new file)

- [ ] **Step 1: Write the failing test**

Create `tests/test_tool_timing.py`:

```python
"""Per-tool-call timing: capture at the call site, pairing in the loop."""

import asyncio

from src.agent.domains.tools import MCPToolWrapper
from src.agent.turn_capture import (
    get_turn_capture,
    record_tool_timing,
    reset_turn_capture,
)
from src.tools.mcp_client import MCPClient, MCPToolResult


def test_mcp_wrapper_records_tool_timing(monkeypatch):
    reset_turn_capture()
    client = MCPClient()

    async def fake_call_tool(**kwargs):
        return MCPToolResult(success=True, data={"ok": 1}, duration_ms=42)

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    wrapper = MCPToolWrapper(
        name="list_things",
        description="test tool",
        tool_server="test-server",
        mcp_client=client,
    )
    asyncio.run(wrapper._arun())
    assert get_turn_capture()["tool_timings"] == [
        {"tool_name": "list_things", "duration_ms": 42}
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_timing.py -v`
Expected: FAIL — `tool_timings == []` (nothing records yet)

- [ ] **Step 3: Implement**

In `src/agent/domains/tools.py`, add to the imports:

```python
from ..turn_capture import record_tool_timing
```

In `MCPToolWrapper._arun`, after the `await self.mcp_client.call_tool(...)` result is assigned and before the return:

```python
        record_tool_timing(self.name, result.duration_ms)
```

(`MCPToolResult.duration_ms` is already measured by `MCPClient.call_tool` — it was simply never carried out of the wrapper.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tool_timing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/domains/tools.py tests/test_tool_timing.py
git commit -m "feat: MCP tool wrapper records per-call duration into turn_capture"
```

---

### Task 3: Doc-search tool records timing

**Files:**
- Modify: `src/agent/tools/access_documents.py:95-159` (`_search_access_documents`)
- Test: `tests/test_tool_timing.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_tool_timing.py`:

```python
def test_doc_search_records_timing(monkeypatch):
    from src.agent.tools.access_documents import _search_access_documents

    reset_turn_capture()

    class _Retrieval:
        chunks = []

    class _Client:
        is_chatmcp_configured = True

        async def retrieve(self, query, rp_name=None):
            return _Retrieval()

    class _Registry:
        def enabled_rag_endpoints(self):
            return {"general", "xdmod"}

        def scoped_rag_enabled(self):
            return True

    monkeypatch.setattr(
        "src.agent.tools.access_documents.get_uky_client", lambda: _Client()
    )
    monkeypatch.setattr(
        "src.agent.tools.access_documents.get_capability_registry",
        lambda: _Registry(),
    )
    asyncio.run(_search_access_documents("how do I use globus"))
    timings = get_turn_capture()["tool_timings"]
    assert len(timings) == 1
    assert timings[0]["tool_name"] == "search_access_documents"
    assert timings[0]["duration_ms"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tool_timing.py::test_doc_search_records_timing -v`
Expected: FAIL — `len(timings) == 0`

- [ ] **Step 3: Implement**

In `src/agent/tools/access_documents.py`:

Add to imports:

```python
import time

from ..turn_capture import record_retrieved_chunks, record_tool_timing
```

(`record_retrieved_chunks` is already imported — extend that import line.)

Wrap the entire body of `_search_access_documents` so every invocation records exactly one timing entry (the 1:1 invariant is what lets the loop pair records back to ToolMessages; early "unavailable" returns record ~0ms, which is the truth):

```python
async def _search_access_documents(
    query: str,
    source: Literal["general", "xdmod"] = "general",
    rp_name: str | None = None,
) -> str:
    start = time.monotonic()
    try:
        return await _search_access_documents_inner(query, source, rp_name)
    finally:
        record_tool_timing(
            "search_access_documents", int((time.monotonic() - start) * 1000)
        )
```

Rename the existing function body to `_search_access_documents_inner` (same signature, keep the original docstring on the inner function). The `StructuredTool.from_function(coroutine=_search_access_documents, ...)` registration at the bottom stays pointed at the outer wrapper.

- [ ] **Step 4: Run tests — new one passes, existing doc-tool tests still pass**

Run: `uv run pytest tests/test_tool_timing.py tests/test_access_documents_tool.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/tools/access_documents.py tests/test_tool_timing.py
git commit -m "feat: doc-search tool records per-call duration into turn_capture"
```

---

### Task 4: Loop pairs timings onto reconstructed ToolResults

**Files:**
- Modify: `src/agent/nodes/tool_calling_loop.py:397-451` (`_build_tool_results`)
- Test: `tests/test_tool_timing.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tool_timing.py`:

```python
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.nodes.tool_calling_loop import _build_tool_results


def _thread_two_calls():
    ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "list_things", "args": {"a": 1}, "id": "c1"},
            {"name": "list_things", "args": {"a": 2}, "id": "c2"},
        ],
    )
    return [
        ai,
        ToolMessage(content="{}", tool_call_id="c1"),
        ToolMessage(content="{}", tool_call_id="c2"),
    ]


def test_build_tool_results_pairs_timings_fifo():
    reset_turn_capture()
    record_tool_timing("list_things", 42)
    record_tool_timing("list_things", 7)
    results, orphans = _build_tool_results(_thread_two_calls(), [])
    assert orphans == 0
    assert [r.duration_ms for r in results] == [42, 7]


def test_build_tool_results_without_timings_defaults_zero():
    reset_turn_capture()
    results, _ = _build_tool_results(_thread_two_calls(), [])
    assert [r.duration_ms for r in results] == [0, 0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tool_timing.py -v`
Expected: `test_build_tool_results_pairs_timings_fifo` FAILS with `[0, 0] != [42, 7]`; the defaults test passes (it documents current behavior).

- [ ] **Step 3: Implement**

In `src/agent/nodes/tool_calling_loop.py`:

Add to imports:

```python
from collections import deque
```

and extend the existing `..turn_capture` import:

```python
from ..turn_capture import get_turn_capture, mark_summarized
```

In `_build_tool_results`, before the `results: list[ToolResult] = []` line, build per-tool FIFO queues:

```python
    # Pair capture-recorded durations back to results, FIFO per tool name.
    # Limitation: two parallel calls to the SAME tool in one turn may swap
    # durations between them (records append in completion order).
    timing_queues: dict[str, deque[int]] = {}
    for rec in get_turn_capture().get("tool_timings", []):
        timing_queues.setdefault(rec["tool_name"], deque()).append(rec["duration_ms"])
```

In the result loop, replace:

```python
        results.append(_parse_tool_message(msg, tc_id, tool_name, server, tool_args))
```

with:

```python
        result = _parse_tool_message(msg, tc_id, tool_name, server, tool_args)
        queue = timing_queues.get(tool_name)
        if queue:
            result.duration_ms = queue.popleft()
        results.append(result)
```

Update the `_build_tool_results` docstring: delete the paragraph "duration_ms is left at 0 because the react loop doesn't track per-call timing; the eval scorer doesn't depend on it." and replace with: "duration_ms is paired from turn_capture's tool_timings (recorded at the call sites), FIFO per tool name; calls with no recorded timing keep 0."

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_timing.py tests/test_tool_calling_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/nodes/tool_calling_loop.py tests/test_tool_timing.py
git commit -m "feat: pair per-tool durations onto reconstructed ToolResults"
```

---

### Task 5: Model-call timing in the loop callback + state channel

**Files:**
- Modify: `src/agent/nodes/tool_calling_loop.py:202-222` (`_TokenUsageAccumulator`), `:313` (callback already registered), `:378-394` (node return)
- Modify: `src/agent/state.py:195` (new channel), `:244` (initial state)
- Test: `tests/test_tool_calling_loop_instrumentation.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tool_calling_loop_instrumentation.py` (it already defines `_resp(total)` and imports `_TokenUsageAccumulator`; `AgentState` / `create_initial_state` are imported for the existing channel tests — extend as needed):

```python
def test_accumulator_times_model_calls():
    acc = _TokenUsageAccumulator()
    rid = "run-1"
    asyncio.run(acc.on_chat_model_start({}, [], run_id=rid))
    asyncio.run(acc.on_llm_end(_resp(30), run_id=rid))
    assert len(acc.model_calls) == 1
    call = acc.model_calls[0]
    assert call["index"] == 0
    assert call["total_tokens"] == 30
    assert call["duration_ms"] >= 0


def test_accumulator_unmatched_run_id_records_zero_duration():
    acc = _TokenUsageAccumulator()
    asyncio.run(acc.on_llm_end(_resp(12), run_id="never-started"))
    assert acc.model_calls[0]["duration_ms"] == 0
    assert acc.total_tokens == 12


def test_model_calls_is_a_declared_state_channel():
    assert "model_calls" in AgentState.__annotations__


def test_create_initial_state_sets_model_calls():
    state = create_initial_state(
        query="q", session_id="s", question_id="qid", tool_catalog={}
    )
    assert state["model_calls"] is None
```

(Mirror the import style of the existing `test_total_tokens_is_a_declared_state_channel` / `test_create_initial_state_sets_total_tokens` at lines 69–75 of that file.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tool_calling_loop_instrumentation.py -v`
Expected: FAIL — `AttributeError: 'on_chat_model_start'` / missing `model_calls`

- [ ] **Step 3: Implement**

In `src/agent/nodes/tool_calling_loop.py`, add `import time` to the imports, then replace `_TokenUsageAccumulator` with:

```python
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

    async def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: Any = None, **kwargs: Any) -> None:
        del serialized, prompts, kwargs
        self._starts[run_id] = time.monotonic()

    async def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: Any = None, **kwargs: Any) -> None:
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
```

In the node's return dict (after `"total_tokens": token_accumulator.total_tokens,`):

```python
            "model_calls": token_accumulator.model_calls,
```

In `src/agent/state.py`, after the `total_tokens` channel declaration (line 195):

```python
    # Output (per-LLM-call timing/tokens recorded by the loop's usage callback).
    model_calls: Annotated[
        list[dict[str, Any]] | None,
        "Per-LLM-call {index, duration_ms, total_tokens} entries for this turn",
    ]
```

(`Any` is already imported in state.py; verify `dict` usage matches the file's existing `dict[str, Any]` style.) And in `create_initial_state`, after `total_tokens=None,`:

```python
        model_calls=None,
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_calling_loop_instrumentation.py tests/test_state.py tests/test_tool_calling_loop.py -v`
Expected: PASS — including the two pre-existing accumulator tests (they call `on_llm_end` without `run_id`, which now defaults to `None` → 0ms entry appended, `total_tokens` math unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/agent/nodes/tool_calling_loop.py src/agent/state.py tests/test_tool_calling_loop_instrumentation.py
git commit -m "feat: record per-model-call timing; declare model_calls state channel"
```

---

### Task 6: Capture trace id at the root span

**Files:**
- Modify: `src/agent/graph.py:84-92` (run_agent span), `:159-167` (stream_agent span)
- Test: `tests/test_tool_timing.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_tool_timing.py`:

```python
def test_record_current_trace_id_writes_32_hex():
    from opentelemetry.sdk.trace import TracerProvider

    from src.agent.graph import _record_current_trace_id

    reset_turn_capture()
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("agent.run") as span:
        _record_current_trace_id(span)
    tid = get_turn_capture()["trace_id"]
    assert tid is not None and len(tid) == 32
    assert int(tid, 16) != 0


def test_record_current_trace_id_noop_for_non_recording_span():
    from opentelemetry.trace import INVALID_SPAN

    from src.agent.graph import _record_current_trace_id

    reset_turn_capture()
    _record_current_trace_id(INVALID_SPAN)
    assert get_turn_capture()["trace_id"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tool_timing.py -v`
Expected: FAIL — `ImportError: cannot import name '_record_current_trace_id'`

- [ ] **Step 3: Implement**

In `src/agent/graph.py`:

Add to imports:

```python
from .turn_capture import record_trace_id
```

Add the helper below the imports:

```python
def _record_current_trace_id(span: Any) -> None:
    """Stash the root span's trace id (32-hex) in turn_capture.

    With telemetry disabled the span is non-recording and its trace_id is 0 —
    record nothing, so turn_reports.trace_id stays NULL rather than holding a
    dead link.
    """
    trace_id = span.get_span_context().trace_id
    if trace_id:
        record_trace_id(format(trace_id, "032x"))
```

In `run_agent`, immediately after the `with tracer.start_as_current_span("agent.run", ...) as root_span:` line:

```python
        _record_current_trace_id(root_span)
```

In `stream_agent`, the span context manager currently doesn't bind a name — change line 167's closing `):` to `) as root_span:` and add the same call immediately inside:

```python
        _record_current_trace_id(root_span)
```

(Order note: `api/routes.py` calls `reset_turn_capture()` before `stream_agent`, and `eval/runner.py` resets before `run_agent` — the capture dict exists by the time this records. No call-site changes needed.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tool_timing.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/agent/graph.py tests/test_tool_timing.py
git commit -m "feat: record turn trace id into turn_capture at the root span"
```

---

### Task 7: Reporter — `trace_id` column + `payload.model_calls`

**Files:**
- Modify: `src/turn_reporter.py:65` (column), `:266-279` (migration map), `:179-216` (`_assemble_turn_report`)
- Test: `tests/test_turn_reporter.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_turn_reporter.py`, following that file's existing pattern (its tests call `_assemble_turn_report` directly with keyword args, and the schema tests inspect the SQLAlchemy models — mirror `test_total_tokens_from_final_state` at line 133 for shape):

```python
    def test_trace_id_column_exists(self):
        assert "trace_id" in TurnReport.__table__.columns

    def test_trace_id_from_capture(self):
        report, _ = _assemble_turn_report(
            final_state={},
            session_id="s",
            turn_index=1,
            question_id="q",
            query_text="hello",
            duration_ms=10.0,
            acting_user=None,
            success=True,
            capabilities=[],
            turn_capture={"trace_id": "ab" * 16, "searched": False, "chunks": []},
        )
        assert report["trace_id"] == "ab" * 16

    def test_trace_id_absent_is_none(self):
        report, _ = _assemble_turn_report(
            final_state={},
            session_id="s",
            turn_index=1,
            question_id="q",
            query_text="hello",
            duration_ms=10.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        assert report["trace_id"] is None

    def test_payload_model_calls_from_final_state(self):
        calls = [{"index": 0, "duration_ms": 1200, "total_tokens": 900}]
        report, _ = _assemble_turn_report(
            final_state={"model_calls": calls},
            session_id="s",
            turn_index=1,
            question_id="q",
            query_text="hello",
            duration_ms=10.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        assert report["payload"]["model_calls"] == calls

    def test_tool_call_duration_flows_to_children(self):
        report, tool_calls = _assemble_turn_report(
            final_state={
                "tool_results": [
                    {
                        "step_id": "c1",
                        "tool_name": "list_things",
                        "server": "srv",
                        "success": True,
                        "arguments": {"a": 1},
                        "duration_ms": 42,
                    }
                ],
                "tools_used": ["list_things"],
            },
            session_id="s",
            turn_index=1,
            question_id="q",
            query_text="hello",
            duration_ms=10.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        assert tool_calls[0]["duration_ms"] == 42
```

Place these inside the existing test class (note the `self` parameter) and match its import list — `TurnReport` and `_assemble_turn_report` are already imported there.

- [ ] **Step 2: Run tests to verify failures**

Run: `uv run pytest tests/test_turn_reporter.py -v`
Expected: the `trace_id` and `model_calls` tests FAIL (`KeyError: 'trace_id'` / `'model_calls'`); `test_tool_call_duration_flows_to_children` PASSES already (the assembler has always copied `duration_ms` — it documents the now-live path).

- [ ] **Step 3: Implement**

In `src/turn_reporter.py`:

Column — after `env = Column(String(16), index=True)` (line 65):

```python
    trace_id = Column(String(32), index=True)  # OTEL trace id of the turn's root span
```

Migration map — add to the `migrations` dict in `_migrate_columns`:

```python
            "trace_id": "VARCHAR(32)",
```

`_assemble_turn_report` — in the `report` dict, after `"env": settings.DEPLOY_ENV or None,`:

```python
        "trace_id": capture.get("trace_id"),
```

and in the `payload` sub-dict, after `"node_trace": ...,`:

```python
            "model_calls": final_state.get("model_calls") or [],
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_turn_reporter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/turn_reporter.py tests/test_turn_reporter.py
git commit -m "feat: turn_reports.trace_id column + payload.model_calls"
```

---

### Task 8: Span per MCP tool call

**Files:**
- Modify: `src/tools/mcp_client.py:94-176` (`call_tool`)
- Test: `tests/test_mcp_client.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_mcp_client.py` (match its existing imports; add what's missing):

```python
def test_call_tool_emits_span(monkeypatch):
    import asyncio

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from src.tools.mcp_client import MCPClient

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(
        "src.telemetry.spans.get_tracer",
        lambda name="test": provider.get_tracer(name),
    )

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": '{"ok": true}'}]}

    class _Client:
        async def post(self, url, json=None, headers=None):
            return _Resp()

    monkeypatch.setattr(
        "src.tools.mcp_client.get_shared_client", lambda timeout: _Client()
    )
    monkeypatch.setattr(MCPClient, "get_server_url", lambda self, s: "http://x")

    result = asyncio.run(
        MCPClient().call_tool(server="srv", tool_name="list_things", arguments={"a": 1})
    )
    assert result.success

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "mcp.call_tool.list_things"
    assert spans[0].attributes["mcp.server"] == "srv"
    assert spans[0].attributes["mcp.success"] is True
    assert spans[0].attributes["mcp.duration_ms"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp_client.py::test_call_tool_emits_span -v`
Expected: FAIL — `len(spans) == 0`

- [ ] **Step 3: Implement**

In `src/tools/mcp_client.py`:

Add import:

```python
from ..telemetry.spans import trace_mcp_call
```

Restructure `call_tool` so the network attempt runs inside the span and every exit path stamps the span. Replace the body from `url = f"{server_url}/tools/{tool_name}"` to the end of the method with:

```python
        url = f"{server_url}/tools/{tool_name}"
        client = get_shared_client(self.timeout)

        logger.info(f"MCP call: {tool_name} with args: {arguments}")

        # Build headers
        headers = {"Content-Type": "application/json"}
        if acting_user:
            headers["X-Acting-User"] = acting_user
        if server in settings.mcp_servers_requiring_api_key and settings.MCP_API_KEY:
            headers["X-Api-Key"] = settings.MCP_API_KEY

        with trace_mcp_call(server, tool_name, arguments) as span:
            try:
                response = await client.post(
                    url,
                    json={"arguments": arguments},
                    headers=headers,
                )
                response.raise_for_status()

                data = response.json()

                # Parse MCP response format
                # Response: {"content": [{"type": "text", "text": "{...json...}"}]}
                parsed_data = self._parse_mcp_response(data)

                logger.info(
                    f"MCP response for {tool_name}: success, data keys: {list(parsed_data.keys()) if isinstance(parsed_data, dict) else type(parsed_data)}"
                )

                result = MCPToolResult(
                    success=True,
                    data=parsed_data,
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

            except httpx.HTTPStatusError as e:
                result = MCPToolResult(
                    success=False,
                    error=f"HTTP {e.response.status_code}: {e.response.text[:500]}",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
            except httpx.TimeoutException:
                result = MCPToolResult(
                    success=False,
                    error=f"Timeout calling {server}/{tool_name}",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )
            except Exception as e:
                result = MCPToolResult(
                    success=False,
                    error=f"{type(e).__name__}: {e}",
                    duration_ms=int(time.time() * 1000) - start_ms,
                )

            span.set_attribute("mcp.duration_ms", result.duration_ms)
            span.set_attribute("mcp.success", result.success)
            if result.error:
                span.set_attribute("mcp.error", result.error[:300])
            return result
```

(The early `ValueError` return for an unknown server stays outside the span, unchanged. `trace_mcp_call` already sets `mcp.server`, `mcp.tool`, and redacted `mcp.arguments` attributes and handles its own status; failures here surface as `mcp.success=false` attributes, since `call_tool` converts exceptions to error results rather than raising.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_client.py -v`
Expected: PASS (all — the restructure must not break existing call_tool tests)

- [ ] **Step 5: Commit**

```bash
git add src/tools/mcp_client.py tests/test_mcp_client.py
git commit -m "feat: emit an OTEL span per MCP tool call (wires existing trace_mcp_call)"
```

---

### Task 9: Full verification

**Files:** none

- [ ] **Step 1: Full suite + lint** (REQUIRED SUB-SKILL: superpowers:verification-before-completion)

```bash
uv run pytest tests/ -x -q -m "not e2e"
uv run ruff check src tests
```

Expected: all tests pass, no lint errors. Fix anything that surfaces before proceeding.

- [ ] **Step 2: End-to-end smoke (local, optional but recommended)**

If local Docker + a battery run is convenient: `docker compose up -d`, run one battery question through the eval CLI (see root `CLAUDE.md`, and note `reference_eval_docker_invocation` — eval must run *inside* the agent container), then check Postgres:

```sql
SELECT tool_name, duration_ms FROM report_tool_calls ORDER BY id DESC LIMIT 5;
SELECT trace_id, payload->'model_calls' FROM turn_reports ORDER BY id DESC LIMIT 1;
```

Expected: nonzero `duration_ms` on MCP/doc-search calls; 32-hex `trace_id` (NULL is correct if OTEL is disabled locally); `model_calls` array with per-call durations.

- [ ] **Step 3: Commit any fixes**

```bash
git add -A && git commit -m "test: fixes from full-suite verification"
```

(Skip if nothing changed.)

---

### Task 10: Update the data-contract doc (reporting repo)

**Files:**
- Modify: `/Users/josephbacal/Projects/sweet-and-fizzy/access-ci/access-agent-reporting/docs/reference/reporting-data-contract.md`

- [ ] **Step 1: Edit the contract doc**

Three precise edits (this is the doc the dashboard build reads — keep it mirroring reality):

1. **§2 table** — add a row after `env`:
   `| trace_id | varchar | OTEL trace id (32-hex) of the turn's root span; NULL when telemetry was off. Deep-link target for Honeycomb. |`
2. **§3 table** — replace the `duration_ms` row's meaning: `**real since the timing PR** — wall-clock ms measured at the call site (was always 0 before).` Also delete the matching bullet in §8 ("report_tool_calls.duration_ms is 0 …").
3. **§4 payload keys** — add: `- model_calls — per-LLM-call timing: [{index, duration_ms, total_tokens}].`

- [ ] **Step 2: Commit (in the reporting repo)**

```bash
cd /Users/josephbacal/Projects/sweet-and-fizzy/access-ci/access-agent-reporting
git add docs/reference/reporting-data-contract.md
git commit -m "docs: contract gains trace_id, real tool durations, payload.model_calls"
```

---

### Task 11: Push + PR

**Files:** none

- [ ] **Step 1: Push branch**

```bash
cd /Users/josephbacal/Projects/sweet-and-fizzy/access-ci/access-agent
git push -u origin feat/turn-timing
```

- [ ] **Step 2: Open PR** (REQUIRED SUB-SKILL: superpowers:requesting-code-review before this)

```bash
gh pr create --repo necyberteam/access-agent --title "feat: per-tool and per-model-call timing + trace id on turn reports" --body "$(cat <<'EOF'
## What

Records granular timing from inside the run and writes it to both Honeycomb spans and the turn report:

- **Per-MCP-tool-call spans** — wires the existing-but-unused `trace_mcp_call` helper into `MCPClient.call_tool` (`mcp.duration_ms`, `mcp.success`, `mcp.error` attributes).
- **`report_tool_calls.duration_ms` is now real** — call-site durations (MCP wrapper + doc-search tool) ride the `turn_capture` side-channel and pair onto reconstructed ToolResults in the loop (FIFO per tool name).
- **Per-model-call timing** — the loop's usage callback now also times each LLM call; written to `payload.model_calls` as `[{index, duration_ms, total_tokens}]` (new `model_calls` state channel).
- **`turn_reports.trace_id`** — new column (32-hex OTEL trace id; NULL when telemetry off) for "view this turn in Honeycomb" deep links.

## Why

Feeds the reporting dashboard's per-turn time breakdowns and battery charts (spec: access-agent-reporting `docs/specs/2026-06-12-turn-timing-and-charts-design.md`). Per the PR #75 data-source split, aggregates stay Honeycomb's job — this is row-level drill-in data.

## Performance

No response-path cost: timing is clock reads; span export stays batched/async; the report write already runs off the response path. No new I/O during the turn.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review notes (done at plan time)

- **Spec coverage:** tool-call spans (Task 8), model-call timing (Task 5), `duration_ms` fill (Tasks 2–4), `trace_id` (Tasks 6–7), zero response-path cost (design constraint stated in every relevant task), contract-doc mirror (Task 10). The dashboard charts and Honeycomb-board work are separate plans per the spec.
- **Known limitation accepted in spec discussion:** two parallel calls to the *same* tool may swap durations (FIFO pairing). Documented in code comment (Task 4).
- **Pre-existing accumulator tests** call `on_llm_end` without `run_id` — new signature keeps it keyword-optional (verified in Task 5 step 4).
