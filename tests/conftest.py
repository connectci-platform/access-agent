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
