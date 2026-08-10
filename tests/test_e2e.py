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

# Mark all tests in this module as e2e and skip if the configured LLM
# provider has no credentials. The nightly runs the production provider
# (LLM_PROVIDER=vllm); local runs default to openai.
_PROVIDER = os.environ.get("LLM_PROVIDER", "openai")
_PROVIDER_KEY = "VLLM_API_KEY" if _PROVIDER == "vllm" else "OPENAI_API_KEY"

pytestmark = [
    pytest.mark.e2e,  # Requires live MCP servers + a live LLM; runs nightly
    pytest.mark.skipif(
        not os.environ.get(_PROVIDER_KEY),
        reason=f"{_PROVIDER_KEY} required for e2e tests (LLM_PROVIDER={_PROVIDER})",
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


_catalog_cache = None

# The production model intermittently routes these usage-stats questions to
# other live tools instead of get_chart_data; membership flickers run to run.
# Tracked in #173 (tool-description fix in access-mcp). Remove entries as the
# fix lands and the nightly stays green.
XFAIL_CHART_SELECTION = {
    "xdmod_most_used_resources",
    "xdmod_active_pis",
    "xdmod_jobs_by_gateway",
    "xdmod_project_count",
    "xdmod_active_allocations_trend",
    "xdmod_gpu_utilization",
    "xdmod_job_count_by_field_of_science",
    "xdmod_aces_allocated",
    "xdmod_filter_with_hyphen",
}


@pytest.fixture
async def tool_catalog():
    """Fetch the live tool catalog once per session, mirroring production.

    The deployed loop aggregates the catalog once at startup; fetching per
    test hammered every server's /tools endpoint 34 times per run, and any
    transient fetch failure silently shrank that test's catalog."""
    global _catalog_cache
    if _catalog_cache is None:
        import asyncio

        from src.config import settings
        from src.tools import CatalogAggregator

        aggregator = CatalogAggregator(timeout=15.0)
        expected = len(settings.mcp_server_urls)
        best: dict = {}
        for attempt in range(1, 4):
            catalog = await aggregator.fetch_catalog(force_refresh=True)
            got = catalog.get("servers_available", 0)
            print(f"[catalog] attempt {attempt}: {got}/{expected} servers available")
            if got > best.get("servers_available", -1):
                best = catalog
            if got == expected:
                break
            await asyncio.sleep(5)
        _catalog_cache = best
    return _catalog_cache


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
async def test_e2e_query(case, run_query, request):
    """Run a single e2e test case from CSV."""
    if get_test_id(case) in XFAIL_CHART_SELECTION:
        request.applymarker(
            pytest.mark.xfail(
                reason="model under-selects get_chart_data; see #173",
                strict=False,
            )
        )
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

    # Check expected tool was used (pipe-separated = any-of, like must_contain)
    if expected_tool:
        accepted = [t.strip() for t in expected_tool.split("|")]
        assert any(t in tools_used for t in accepted), (
            f"Expected one of tools {accepted} not used. Got: {tools_used}"
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
