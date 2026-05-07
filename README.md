# ACCESS Documentation Agent (LangGraph)

A LangGraph-powered documentation agent for ACCESS-CI with RAG-primary architecture.

## Architecture

```
Query → tool_calling_loop → Answer
```

Single-node design. The LLM inside the loop sees the full MCP catalog
plus a `search_access_documents` tool wrapping ACCESS-CI documentation
retrieval, and decides on each turn whether to look something up, call
a live tool, or emit a final answer. No upstream classifier, no domain
router, no separate plan→execute→synthesize chain.

- **Documentation questions**: the LLM calls `search_access_documents`,
  reads the result, decides whether to follow up.
- **Live data**: the LLM calls the relevant MCP tool (allocations,
  system status, events, etc.) directly.
- **Mixed**: the LLM interleaves doc lookups and tool calls inside one
  loop, synthesizing across both as it goes.

## Components

```
┌─────────────────┐
│  access-agent   │ ← This repo
│  (LangGraph)    │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐     ┌─────────────────┐
│  QA Service     │     │  MCP Servers    │
│  (FastAPI)      │     │  (TypeScript)   │
│  pgvector RAG   │     │  10 servers     │
└─────────────────┘     └─────────────────┘
```

- **access-qa-service**: RAG retrieval from verified Q&A pairs
- **MCP servers**: Real-time ACCESS data (allocations, resources, status, etc.)

## Quick Start

### Prerequisites

- Python 3.11+
- Access to MCP servers (or local catalog file)
- OpenAI API key

### Installation

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Edit .env with your settings
```

### Running Locally

```bash
# Start the API server
uv run python -m src.main

# Or with uvicorn directly
uv run uvicorn src.main:app --reload
```

### API Usage

```bash
# Query the agent
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What GPU resources are available on Delta?"}'

# Health check
curl http://localhost:8000/api/v1/health

# List available tools
curl http://localhost:8000/api/v1/tools
```

## Configuration

Environment variables (see `.env.example`):

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | local, docker, production | local |
| `LLM_PROVIDER` | openai, vllm, access_ai | openai |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `OPENAI_MODEL` | Model name | gpt-4o |
| `MCP_CATALOG_URL` | URL to fetch tool catalog | - |
| `MCP_CATALOG_PATH` | Path to local catalog file | - |
| `MAX_TOKENS_LOOP` | max_tokens for the react agent inside the loop | 6000 |

## Development

```bash
# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
uv run ruff format src/
```

## Docker

```bash
# Development
docker-compose up --build

# Production (uses GHCR image)
docker-compose -f docker-compose.prod.yml up -d
```

## Observability

The agent includes comprehensive OpenTelemetry tracing for debugging and performance analysis.

### Setup

Tracing is enabled by setting environment variables:

```bash
OTEL_ENABLED=true
OTEL_EXPORTER_OTLP_ENDPOINT=https://your-otlp-endpoint
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic xxx
```

### What's Traced

- **Agent flow**: the `tool_calling_loop` span captures one execution
- **LLM calls**: Model, tokens (input/output/cached), latency (via `opentelemetry-instrumentation-langchain`)
- **MCP tool calls**: Server, tool name, arguments, results
- **Doc retrieval**: `search_access_documents` calls (query, matches)

### Viewing Traces

Traces are exported to Honeycomb (or any OTLP-compatible backend). View them in:
- Honeycomb UI: https://ui.honeycomb.io → Dataset: `access-ci`
- Filter by `service.component = "access-agent"` to see agent traces

The Honeycomb board in `observability/honeycomb.tf` provides pre-built dashboards for monitoring.

## Project Structure

```
src/
├── main.py              # FastAPI entry point
├── config.py            # Environment configuration
├── agent/
│   ├── graph.py         # LangGraph definition (single-node loop)
│   ├── state.py         # State schema
│   ├── nodes/
│   │   └── tool_calling_loop.py  # The only node — react-style tool loop
│   ├── prompts/
│   │   └── no_classify.py        # System prompt for the loop
│   └── tools/
│       └── access_documents.py   # search_access_documents — doc-retrieval tool
├── services/
│   └── uky_client.py    # HTTP client for UKY RAG endpoints
├── tools/
│   ├── mcp_client.py    # MCP HTTP client
│   └── catalog_aggregator.py
├── llm/
│   └── providers.py     # LLM abstraction (OpenAI/vLLM)
└── api/
    └── routes.py        # FastAPI routes
```

## Related Repos

- [access-qa-service](https://github.com/necyberteam/access-qa-service) - RAG retrieval service
- [access_mcp](https://github.com/necyberteam/access_mcp) - MCP servers for ACCESS data
