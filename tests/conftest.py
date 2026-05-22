"""Pytest configuration and fixtures."""

import os
from pathlib import Path

import pytest

# Load .env file for tests if it exists
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with env_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


@pytest.fixture(autouse=True)
def _reset_async_singletons():
    """Drop module-global cached httpx clients between tests.

    ``uky_client._client`` and ``mcp_client._shared_client`` each cache an
    ``httpx.AsyncClient`` bound to whichever event loop was live when it was
    first created. pytest-asyncio gives each test a fresh function-scoped
    loop, so a client created in one test is reused in the next against a
    closed loop -- surfacing as ``RuntimeError: Event loop is closed``.
    ``is_closed`` does not catch this (a dead loop does not close the
    client), so resetting the globals is the reliable fix: each test
    rebuilds its client on its own loop.
    """
    yield
    from src.services import uky_client
    from src.tools import mcp_client

    uky_client._client = None
    mcp_client._shared_client = None


@pytest.fixture
def sample_catalog():
    """Sample MCP tool catalog for testing."""
    return {
        "quick_lookup": {
            "search_resources": {"server": "compute-resources"},
            "get_current_outages": {"server": "system-status"},
            "search_software": {"server": "software-discovery"},
        },
        "tools": [
            {
                "name": "search_resources",
                "description": "Search for compute resources",
                "parameters": [
                    {"name": "has_gpu", "type": "boolean", "required": False},
                    {"name": "resource_type", "type": "string", "required": False},
                ],
            },
            {
                "name": "get_current_outages",
                "description": "Get current system outages",
                "parameters": [
                    {"name": "resource_filter", "type": "string", "required": False},
                ],
            },
            {
                "name": "search_software",
                "description": "Search for software packages",
                "parameters": [
                    {"name": "query", "type": "string", "required": True},
                    {"name": "resource", "type": "string", "required": False},
                ],
            },
        ],
    }


@pytest.fixture
def sample_query():
    """Sample user query."""
    return "What GPU resources are available on ACCESS?"
