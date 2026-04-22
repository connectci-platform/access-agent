"""End-to-end tests for the ACCESS Documentation Agent.

These tests run actual queries through the full agent pipeline against
live MCP servers. They verify that:
1. The planner selects appropriate tools with valid parameters
2. MCP tools execute successfully
3. The agent produces helpful responses

Test cases are defined in e2e_test_cases.csv for easy maintenance.

Run with: MCP_SERVER_HOST=45.79.215.140 uv run pytest tests/test_e2e.py -v -m e2e

These tests require:
- OPENAI_API_KEY set in environment
- MCP servers running (locally or specify MCP_SERVER_HOST)
"""

import csv
import os
from pathlib import Path

import pytest

# Mark all tests in this module as e2e and skip if OPENAI_API_KEY is not set
pytestmark = [
    pytest.mark.e2e,  # Requires live MCP servers + real OpenAI; runs nightly
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for e2e tests",
    ),
]

TEST_CASES_FILE = Path(__file__).parent / "e2e_test_cases.csv"


def load_test_cases():
    """Load test cases from CSV file."""
    with TEST_CASES_FILE.open() as f:
        return list(csv.DictReader(f))


def get_test_id(case):
    """Generate test ID from case description."""
    return case.get("description", "unknown").replace(" ", "_").lower()


@pytest.fixture
async def tool_catalog():
    """Fetch live tool catalog from MCP servers."""
    from src.tools import CatalogAggregator

    aggregator = CatalogAggregator(timeout=15.0)
    return await aggregator.fetch_catalog()


@pytest.fixture
def run_query(tool_catalog):
    """Factory fixture to run queries through the agent."""

    async def _run_query(query: str) -> dict:
        from src.agent.graph import run_agent

        return await run_agent(
            query=query,
            session_id="e2e_test",
            question_id="e2e_test",
            tool_catalog=tool_catalog,
            use_checkpointing=False,
        )

    return _run_query


@pytest.mark.parametrize("case", load_test_cases(), ids=get_test_id)
async def test_e2e_query(case, run_query):
    """Run a single e2e test case from CSV."""
    query = case["query"]
    expected_tool = case.get("expected_tool", "").strip()
    must_contain = case.get("must_contain", "").strip()
    must_not_contain = case.get("must_not_contain", "").strip()

    # Run the query
    result = await run_query(query)

    # Always check we got an answer
    assert result.get("final_answer"), f"Should produce an answer for: {query}"

    answer = result.get("final_answer", "").lower()
    tools_used = result.get("tools_used", [])

    # Check expected tool was used
    if expected_tool:
        assert expected_tool in tools_used, (
            f"Expected tool '{expected_tool}' not used. Got: {tools_used}"
        )

    # Check must_contain words (pipe-separated)
    if must_contain:
        words = [w.strip().lower() for w in must_contain.split("|")]
        assert any(word in answer for word in words), (
            f"Answer should contain one of {words}. Got: {answer[:200]}"
        )

    # Check must_not_contain words (pipe-separated)
    if must_not_contain:
        words = [w.strip().lower() for w in must_not_contain.split("|")]
        for word in words:
            assert word not in answer, f"Answer should not contain '{word}'. Got: {answer[:200]}"
