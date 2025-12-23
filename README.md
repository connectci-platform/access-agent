# ACCESS Documentation Agent (LangGraph)

A LangGraph-powered documentation agent for ACCESS-CI, migrated from n8n workflows for better code-first development experience.

## Architecture

```
Query → Plan → Execute → Synthesize → Answer
         ↓
    (MCP Tools)
```

- **Plan Node**: LLM analyzes query and selects MCP tools to call
- **Execute Node**: Calls MCP tools in parallel with dependency resolution
- **Synthesize Node**: Generates natural language answer from tool results

## Quick Start

### Prerequisites

- Python 3.11+
- Access to MCP servers (or local catalog file)
- OpenAI API key (or vLLM server)

### Installation

```bash
cd /Users/drew/Sites/connectci/access-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Copy environment file
cp .env.example .env
# Edit .env with your API keys
```

### Running Locally

```bash
# Start the API server
python -m src.main

# Or with uvicorn directly
uvicorn src.main:app --reload
```

### API Usage

```bash
# Query the agent
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What GPU resources are available?"}'

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

## Development

```bash
# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## Docker

```bash
# Build and run
docker-compose up --build

# Access API at http://localhost:8000
```

## Project Structure

```
src/
├── main.py              # FastAPI entry point
├── config.py            # Environment configuration
├── agent/
│   ├── graph.py         # LangGraph definition
│   ├── state.py         # State schema
│   └── nodes/           # Graph nodes (plan, execute, synthesize)
├── tools/
│   ├── mcp_client.py    # MCP HTTP client
│   └── registry.py      # Tool catalog loader
├── llm/
│   └── providers.py     # LLM abstraction (OpenAI/vLLM)
└── api/
    └── routes.py        # FastAPI routes
```

## Future Enhancements

- [ ] Quality evaluation loop (retry if result unhelpful)
- [ ] Error recovery with LLM-driven replanning
- [ ] PostgreSQL checkpointing for durability
- [ ] Trigger.dev integration for production hardening
- [ ] vLLM integration for self-hosted models
