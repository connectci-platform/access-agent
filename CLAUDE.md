# access-agent

The central orchestrator of the ACCESS-CI AI assistant: a Python/LangGraph service that runs a
**single-node tool-calling loop**. A reasoning LLM is given the full tool catalog (document search +
MCP tools) and decides, turn by turn, which tools to call and when to answer. No classifier, no
router, no separate synthesis node.

## Where this sits

Calls UKY's RAG (via `chat-mcp`) for documents and the `access-mcp` servers for live data; serves
`qa-bot-core` / MCP clients. Canonical architecture and the "why" behind it:

- **Architecture / current state** → `access-qa-planning/active/01-agent-architecture.md`
- **Decisions (ADRs)** → `access-qa-planning/decisions/` (esp. 008 loop, 009 Qwen, 010 context mgmt, 011 chat-mcp, 012 MCP envelope)

## Run / test

See `README.md` for full setup. Day-to-day:

```bash
uv sync                                   # install
uv run python -m src.main                 # run API (localhost:8000)
uv run pytest                             # tests
uv run mypy src/ && uv run ruff check src/  # types + lint
docker-compose up --build                 # full local stack
```

## Conventions & gotchas

- **The loop is the only graph node.** `src/agent/nodes/tool_calling_loop.py` (built with
  `create_agent` + `SummarizationMiddleware`). `src/agent/graph.py` is `START → loop → END`.
- **LLM provider:** code default is OpenAI `gpt-4o` for local dev; **production sets `LLM_PROVIDER`
  to the UKY vLLM (Qwen) endpoint.** The default is not the production model (ADR 009).
- **Document retrieval is a tool** — `src/agent/tools/access_documents.py`. `source="general"` →
  UKY `chat-mcp` (raw chunks, agent synthesizes); `source="xdmod"` → legacy `/ask`.
- **pgvector / access-qa-service is NOT wired in.** The agent retrieves only from UKY. Ignore stale
  references suggesting otherwise.
- **Qwen reasoning trace** is stripped at the LLM client layer (`src/llm/providers.py`), not in a
  node. The stream is `…reasoning…</think>answer` (no opening `<think>`).
- **Eval CLI** runs inside the agent container (`docker compose exec`), not the host venv; battery
  files must be copied in. Full eval reference: `eval/README.md`; report rendering:
  `src/eval/html_report/README.md`.

## Key paths

`src/agent/` (graph, nodes, prompts, tools) · `src/llm/providers.py` (provider abstraction) ·
`src/config.py` (env config) · `src/eval/` (eval pipeline) · `src/api/routes.py` (FastAPI).
