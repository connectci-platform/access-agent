#!/usr/bin/env python3
"""Quick test script for the ACCESS Documentation Agent.

Usage:
    python test_agent.py                    # Default query, no checkpointing
    python test_agent.py "your question"    # Custom query
    python test_agent.py --checkpoint "q"   # With checkpointing

Reads configuration from .env file.
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file manually if dotenv not available
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from src.agent.graph import run_agent
from src.config import settings
from src.tools import ToolRegistry


async def main():
    # Parse args
    use_checkpoint = "--checkpoint" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--checkpoint"]
    query = args[0] if args else "What GPU resources are available on ACCESS?"

    print(f"Environment: {settings.ENVIRONMENT}")
    print(f"LLM Provider: {settings.LLM_PROVIDER}")
    print(f"MCP Catalog: {settings.MCP_CATALOG_PATH or settings.MCP_CATALOG_URL}")
    print(f"Checkpointing: {'enabled' if use_checkpoint else 'disabled'}")
    if use_checkpoint:
        print(f"Database: {settings.DATABASE_URL}")
    print()

    # Load catalog
    print("Loading tool catalog...")
    registry = ToolRegistry()
    await registry.load()
    print(f"Loaded {registry.tool_count} tools")
    print()

    print(f"Query: {query}")
    print("-" * 50)

    result = await run_agent(
        query=query,
        session_id="test_session",
        question_id="test_question",
        tool_catalog=registry.catalog,
        use_checkpointing=use_checkpoint,
        db_uri=settings.DATABASE_URL if use_checkpoint else None,
    )

    print()
    print("=== RESULT ===")
    print(f"Tools used: {result.get('tools_used', [])}")
    print(f"Strategy: {result.get('execution_strategy')}")
    print()
    print("Answer:")
    print(result.get("final_answer", "No answer"))


if __name__ == "__main__":
    asyncio.run(main())
