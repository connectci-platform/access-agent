# ACCESS Documentation Agent (LangGraph)

A LangGraph-powered documentation agent for ACCESS-CI with RAG-primary architecture.

## Architecture

```
Query → Classify → Route
                    ├─ Static/Combined → RAG (QA Service) → [tools if needed] → Synthesize
                    └─ Dynamic → Plan → Execute (MCP Tools) → Synthesize
```

**RAG-Primary Design:**
- **Static queries**: Answered directly from verified Q&A pairs (87% accuracy vs 58% with fine-tuning)
- **Dynamic queries**: Real-time data from MCP tools (allocations, system status, etc.)
- **Combined queries**: RAG knowledge + tool results synthesized together

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
- Access to QA service (or skip RAG with `QA_SERVICE_URL=""`)
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
| `QA_SERVICE_URL` | RAG service URL | http://localhost:8001 |
| `RAG_THRESHOLD_STATIC` | Similarity threshold for static queries | 0.85 |
| `RAG_THRESHOLD_COMBINED` | Similarity threshold for combined queries | 0.75 |
| `MCP_CATALOG_URL` | URL to fetch tool catalog | - |
| `MCP_CATALOG_PATH` | Path to local catalog file | - |

## Query Classification

The agent classifies queries into three types:

| Type | Description | Path |
|------|-------------|------|
| **static** | Factual questions about resources | RAG only |
| **dynamic** | User-specific or real-time data | MCP tools only |
| **combined** | Both static knowledge + live data | RAG + MCP tools |

Examples:
- Static: "What GPUs does Delta have?" → RAG
- Dynamic: "What are my current allocations?" → MCP tools
- Combined: "What GPUs does Delta have and is it operational?" → RAG + tools

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

## Project Structure

```
src/
├── main.py              # FastAPI entry point
├── config.py            # Environment configuration
├── agent/
│   ├── graph.py         # LangGraph definition
│   ├── state.py         # State schema (includes RAGMatch)
│   └── nodes/
│       ├── classify.py  # Query classification
│       ├── rag_answer.py # RAG retrieval from QA service
│       ├── plan.py      # Tool selection
│       ├── execute.py   # MCP tool execution
│       ├── evaluate.py  # Result quality check
│       ├── recover.py   # Error recovery
│       └── synthesize.py # Answer generation (RAG-aware)
├── services/
│   └── qa_client.py     # HTTP client for QA service
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
