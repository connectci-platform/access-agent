# Streaming Agent Responses

**Date:** 2026-04-03
**Status:** Draft
**Repos affected:** access-agent, qa-bot-core, access-qa-bot

## Problem

The agent takes 3-15 seconds to respond depending on query complexity (classification, RAG, tool calls, synthesis). During this time the user sees bouncing dots with no indication of progress. Modern AI chat interfaces stream responses token-by-token with status updates, setting user expectations that our chatbot doesn't meet.

## Solution

Stream agent responses via Server-Sent Events (SSE). The agent emits status updates at each node boundary and streams LLM synthesis tokens as they're generated. The frontend displays status messages that transition into the streaming answer.

## Architecture

```
Frontend (qa-bot-core)              Agent (access-agent)
─────────────────────              ────────────────────

POST /query ──────────────────────▶ routes.py
  {query, session_id, ...}           │
                                     │ stream_agent()
◀─ SSE Stream ────────────────────   │
                                     ▼
event: status                      classify_node
  "Classifying query..."             writer({"status": "..."})
                                     │
event: status                      rag_answer_node
  "Searching documentation..."       writer({"status": "..."})
                                     │
event: status                      execute_node
  "Querying system status..."        writer({"status": "..."})
                                     │
event: status                      synthesize_node
  "Generating answer..."             writer({"status": "..."})
                                     │
event: token                       LLM tokens (automatic)
  "ACCESS has "                      stream_mode="messages"
event: token                         │
  "57 affinity "                     │
event: token                         │
  "groups..."                        │
                                     │
event: done                        Stream complete
  {metadata}                         Final state extracted
```

## Event Format

Three SSE event types:

### `status` — Progress updates from graph nodes

```
event: status
data: {"message": "Classifying query..."}
```

Human-readable status messages emitted by each node. The frontend shows these in the message bubble, replacing the previous status as new ones arrive.

### `token` — LLM synthesis output

```
event: token
data: {"content": "ACCESS has "}
```

Individual tokens from the synthesis LLM. When the first token arrives, the frontend transitions from showing status to streaming the answer. Subsequent tokens append.

### `done` — Response metadata

```
event: done
data: {"metadata": {"capability_id": "browse_affinity_groups", "is_final_response": true, "rating_target": "agent", "tools_used": ["search_affinity_groups"], "duration_ms": 3450, "question_id": "q_123"}, "success": true}
```

Sent after the stream completes. Contains the same metadata the current non-streaming response returns. The frontend uses this for rating buttons, analytics events, and usage tracking.

### Error event

```
event: error
data: {"message": "Failed to process query", "code": "agent_error"}
```

Sent if the agent encounters an unrecoverable error during streaming.

## Agent Implementation

### LangGraph Streaming

LangGraph supports multiple stream modes that can be combined. We use three:

- **`"custom"`** — Status messages emitted via `get_stream_writer()` from inside nodes
- **`"messages"`** — LLM tokens captured automatically from any `ChatOpenAI` call
- **`"updates"`** — State changes after each node (used to extract final metadata)

```python
async for chunk in graph.astream(
    initial_state,
    stream_mode=["custom", "messages", "updates"],
    config={},
):
    # chunk has {"type": "custom"|"messages"|"updates", "data": ...}
```

Token streaming happens automatically — LangGraph intercepts `ChatOpenAI`'s internal streaming even though nodes use `ainvoke()`. No changes to LLM call patterns needed.

### New `stream_agent()` function

`graph.py` gets `stream_agent()` alongside the existing `run_agent()`. It's an async generator that yields events:

```python
async def stream_agent(query, session_id, question_id, tool_catalog, ...):
    graph = create_agent_graph()
    initial_state = create_initial_state(...)

    async for chunk in graph.astream(
        initial_state,
        stream_mode=["custom", "messages", "updates"],
    ):
        yield chunk
```

`run_agent()` remains unchanged for internal use (eval pipeline, tests).

### Status messages from nodes

Each user-facing node calls `get_stream_writer()` to emit status:

| Node | Status message |
|------|---------------|
| classify | "Classifying query..." |
| rag_answer | "Searching ACCESS documentation..." |
| rag_and_plan | "Searching documentation and planning..." |
| plan | "Planning tool calls..." |
| execute | "Querying {tool_name}..." (dynamic, per tool) |
| evaluate | (none — internal quality check) |
| recover | (none — internal retry logic) |
| synthesize | "Generating answer..." |
| domain_agent | "Starting {domain} workflow..." |

The execute node emits per-tool status with the actual tool name for specificity ("Querying search_affinity_groups..." or a human-friendly version).

### SSE endpoint

`routes.py` replaces the current JSON response with SSE for agent queries:

```python
@router.post("/query")
async def query_agent(request: QueryRequest, ...):
    # Discovery short-circuit still returns JSON (instant, no streaming needed)
    discovery = _check_capability_discovery(...)
    if discovery:
        return discovery

    # Agent queries stream via SSE
    return StreamingResponse(
        _stream_events(request, ...),
        media_type="text/event-stream",
    )
```

The `_stream_events` async generator:
1. Yields `status` events from custom stream chunks
2. Yields `token` events from message stream chunks (filtering to synthesis node only)
3. Collects final state from update chunks
4. Yields a `done` event with metadata after the stream completes
5. Logs usage asynchronously after done

### Token filtering

`stream_mode="messages"` captures tokens from ALL LLM calls — classification, planning, evaluation, synthesis. We only want to stream synthesis tokens to the user. Filter by checking `metadata["langgraph_node"]` on message chunks:

```python
if chunk["type"] == "messages":
    msg, metadata = chunk["data"]
    if metadata.get("langgraph_node") == "synthesize":
        yield sse_event("token", {"content": msg.content})
```

This ensures classification and planning LLM calls don't leak internal reasoning to the user.

## Frontend Implementation

### Transport

Standard `EventSource` only supports GET. Since we POST the query body, use `fetch()` with readable stream:

```typescript
const response = await fetch(endpoint, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ query, session_id, question_id }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
// Parse SSE events from chunks
```

Alternatively, use `@microsoft/fetch-event-source` which handles POST + SSE parsing.

### Message display flow

Using react-chatbotify's `streamMessage()`:

1. **Status phase**: On each `status` event, call `streamMessage(statusText)`. Each new status replaces the previous one in the same message bubble.
2. **Token phase**: On the first `token` event, call `streamMessage(tokenContent)` which replaces the last status. Subsequent tokens append via additional `streamMessage()` calls.
3. **Done phase**: On `done` event, call `endStreamMessage()`. Store metadata for rating buttons and analytics.

### Rating buttons

Rating buttons appear after `endStreamMessage()` based on `done` event metadata:
- `is_final_response: true` AND `rating_target` is not null → show buttons
- Otherwise → suppress buttons

### Error handling

On `error` event or stream failure:
- Display the error message in the chat
- End the stream cleanly
- Allow the user to try again

## What Doesn't Change

- Graph structure (nodes, edges, routing logic)
- LLM calls inside nodes (still `ainvoke()` — LangGraph handles the streaming interception)
- Capability discovery short-circuit (instant JSON, no SSE)
- Checkpointing
- The eval pipeline (uses `run_agent()` directly, not HTTP)
- MCP tool calls (execute node still calls tools synchronously)

## Discovery Responses

Capability discovery (`_check_capability_discovery`) returns regular JSON responses, not SSE. These are instant (no LLM, no tools) and don't benefit from streaming.

The frontend detects which response type it received from the `Content-Type` header:
- `text/event-stream` → read as SSE stream (agent queries)
- `application/json` → parse as JSON (discovery, Turnstile challenges, errors)

Both go through the same `/query` endpoint. The frontend must handle both content types from a single `fetch()` call.

## Usage Logging

Usage logging happens after the stream completes, in the `done` event handler on the server side. The `_stream_events` generator collects the final state from `updates` chunks and logs to `usage_logs` after yielding the `done` event.

## Scope

| Repo | Changes |
|------|---------|
| access-agent | `stream_agent()` in graph.py. `get_stream_writer()` in each node. SSE `StreamingResponse` in routes.py. |
| qa-bot-core | SSE reader + `streamMessage()` in qa-flow.tsx. Add fetch-event-source or manual SSE parsing. |
| access-qa-bot | Minimal — transport changes are in qa-bot-core. |
