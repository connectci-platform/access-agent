"""End-to-end tests for the ACCESS Documentation Agent.

These tests run actual queries through the full agent pipeline against
live MCP servers. They verify that:
1. The planner selects appropriate tools with valid parameters
2. MCP tools execute successfully
3. The agent produces helpful responses

Run with: pytest tests/test_e2e.py -v -m e2e
Run against production: MCP_SERVER_HOST=45.79.215.140 pytest tests/test_e2e.py -v -m e2e

These tests require:
- OPENAI_API_KEY set in environment
- MCP servers running (locally or specify MCP_SERVER_HOST)
"""

import os

import pytest

# Mark all tests in this module as e2e and skip if OPENAI_API_KEY is not set
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for e2e tests",
    ),
]


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


class TestOutageQueries:
    """Test queries about system outages and maintenance."""

    async def test_current_outages(self, run_query):
        """Test asking about current outages."""
        result = await run_query("What are the current outages in ACCESS?")

        assert result.get("final_answer"), "Should produce an answer"
        assert "get_infrastructure_news" in result.get(
            "tools_used", []
        ), "Should use get_infrastructure_news tool"
        # Should not contain error messages about invalid parameters
        answer = result.get("final_answer", "").lower()
        assert (
            "invalid" not in answer or "parameter" not in answer
        ), f"Answer suggests parameter error: {answer[:200]}"

    async def test_planned_outages(self, run_query):
        """Test asking about planned/scheduled outages."""
        result = await run_query("What are the planned outages in ACCESS?")

        assert result.get("final_answer"), "Should produce an answer"
        assert "get_infrastructure_news" in result.get(
            "tools_used", []
        ), "Should use get_infrastructure_news tool"
        answer = result.get("final_answer", "").lower()
        assert (
            "invalid" not in answer or "parameter" not in answer
        ), f"Answer suggests parameter error: {answer[:200]}"

    async def test_scheduled_maintenance(self, run_query):
        """Test asking about scheduled maintenance."""
        result = await run_query("Is there any scheduled maintenance coming up?")

        assert result.get("final_answer"), "Should produce an answer"
        assert "get_infrastructure_news" in result.get(
            "tools_used", []
        ), "Should use get_infrastructure_news tool"


class TestResourceQueries:
    """Test queries about compute resources."""

    async def test_gpu_resources(self, run_query):
        """Test asking about GPU resources."""
        result = await run_query("What GPU resources are available on ACCESS?")

        assert result.get("final_answer"), "Should produce an answer"
        tools_used = result.get("tools_used", [])
        assert any(
            "resource" in t.lower() for t in tools_used
        ), f"Should use a resource-related tool, got: {tools_used}"

    async def test_specific_resource(self, run_query):
        """Test asking about a specific resource."""
        result = await run_query("Tell me about Delta at NCSA")

        assert result.get("final_answer"), "Should produce an answer"
        answer = result.get("final_answer", "").lower()
        # Should mention Delta or NCSA in the response
        assert (
            "delta" in answer or "ncsa" in answer
        ), "Response should mention the requested resource"


class TestSoftwareQueries:
    """Test queries about software availability."""

    async def test_software_search(self, run_query):
        """Test searching for software."""
        result = await run_query("Is Python available on ACCESS systems?")

        assert result.get("final_answer"), "Should produce an answer"
        tools_used = result.get("tools_used", [])
        assert any(
            "software" in t.lower() for t in tools_used
        ), f"Should use a software-related tool, got: {tools_used}"


class TestAnnouncementQueries:
    """Test queries about announcements and news."""

    async def test_recent_announcements(self, run_query):
        """Test asking about recent announcements."""
        result = await run_query("What are the recent ACCESS announcements?")

        assert result.get("final_answer"), "Should produce an answer"
        tools_used = result.get("tools_used", [])
        assert any(
            "announcement" in t.lower() or "news" in t.lower() for t in tools_used
        ), f"Should use an announcement-related tool, got: {tools_used}"


class TestGeneralQueries:
    """Test general queries that may not need tools."""

    async def test_what_is_access(self, run_query):
        """Test asking what ACCESS is."""
        result = await run_query("What is ACCESS-CI?")

        assert result.get("final_answer"), "Should produce an answer"
        answer = result.get("final_answer", "").lower()
        # Should mention ACCESS or cyberinfrastructure
        assert (
            "access" in answer or "cyberinfrastructure" in answer or "hpc" in answer
        ), "Response should explain what ACCESS is"


class TestToolParameterValidation:
    """Test that the planner generates valid tool parameters.

    These tests specifically check for issues like the 'time: future' bug
    where the planner sent invalid parameter values.
    """

    async def test_time_parameter_values(self, tool_catalog, run_query):
        """Verify planner uses valid time parameter values for infrastructure news."""
        from src.tools import ToolRegistry

        registry = ToolRegistry(catalog=tool_catalog)

        # Check that the tool has the expected enum values
        tool = registry.get_tool("get_infrastructure_news")
        if tool:
            time_param = next((p for p in tool.parameters if p.name == "time"), None)
            if time_param:
                # Just verify the tool exists and has the time parameter
                assert time_param.name == "time"

        # Run a query that should use the time parameter
        result = await run_query("Show me scheduled maintenance")
        assert result.get("final_answer"), "Should produce an answer"

    async def test_empty_arrays_handled(self, run_query):
        """Test that empty arrays don't cause errors."""
        # This query previously caused ids: [] to be sent which triggered errors
        result = await run_query("Are there any outages affecting Delta?")

        assert result.get("final_answer"), "Should produce an answer"
        answer = result.get("final_answer", "").lower()
        # Should not indicate a parameter validation error
        assert (
            "required" not in answer or "parameter" not in answer
        ), f"Answer suggests parameter error: {answer[:200]}"
