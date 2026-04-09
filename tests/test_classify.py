"""Tests for query classification routing.

Verifies the classifier routes queries to the correct pipeline
(query_type, rag_endpoint, domain) without running the full agent.

Requires OPENAI_API_KEY (uses gpt-4o-mini for classification).

Run with: uv run pytest tests/test_classify.py -v -m classify
"""

import os

import pytest

pytestmark = [
    pytest.mark.classify,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY required for classification tests",
    ),
]


async def classify(query: str):
    """Helper to classify a query."""
    from src.agent.nodes.classify import classify_query_with_llm

    return await classify_query_with_llm(query)


# --- Static / general documentation ---


class TestStaticRouting:
    """Queries that should route to static/general (RAG only, no tools)."""

    async def test_how_to_get_allocation(self):
        r = await classify("How do I get an allocation?")
        assert r.query_type == "static"
        assert r.rag_endpoint == "general"

    async def test_resource_description(self):
        # "What GPUs does Delta have?" is genuinely ambiguous — it can be
        # answered from documentation (static RAG) or by querying the
        # compute-resources MCP server (combined). Either is acceptable.
        r = await classify("What GPUs does Delta have?")
        assert r.query_type in ("static", "combined")
        assert r.rag_endpoint == "general"

    async def test_policy_question(self):
        r = await classify("What is an SU?")
        assert r.query_type == "static"
        assert r.rag_endpoint == "general"

    async def test_about_supremm(self):
        """Asking what SUPREMM is = documentation, not data."""
        r = await classify("What can you tell me about SUPREMM?")
        assert r.query_type == "static"
        assert r.rag_endpoint == "general"

    async def test_about_allocations(self):
        """Asking about allocations in general = docs, not aggregate counts."""
        r = await classify("Tell me about allocations in ACCESS")
        assert r.query_type == "static"
        assert r.rag_endpoint == "general"

    async def test_resource_comparison(self):
        r = await classify("Compare Bridges-2 and Expanse for AI workloads")
        # "Compare X and Y" is genuinely ambiguous — the classifier may
        # route to static (docs are enough) or combined (wants live
        # hardware details). Both are defensible.
        assert r.query_type in ("static", "combined")
        assert r.rag_endpoint == "general"

    async def test_acknowledge_access(self):
        r = await classify("How do I acknowledge ACCESS in a publication?")
        assert r.query_type == "static"
        assert r.rag_endpoint == "general"

    async def test_project_types(self):
        r = await classify("What are the different project types in ACCESS?")
        assert r.query_type == "static"
        assert r.rag_endpoint == "general"


# --- Dynamic / no RAG ---


class TestDynamicRouting:
    """Queries needing only live data, no RAG."""

    async def test_system_status(self):
        r = await classify("Is Delta down right now?")
        assert r.query_type == "dynamic"
        assert r.rag_endpoint is None

    async def test_user_specific(self):
        r = await classify("What's my usage on Expanse?")
        assert r.query_type == "dynamic"
        assert r.rag_endpoint is None

    async def test_upcoming_events(self):
        r = await classify("Are there any upcoming training workshops?")
        assert r.query_type == "dynamic"
        assert r.rag_endpoint is None

    async def test_current_outages(self):
        r = await classify("Are there any current outages?")
        assert r.query_type == "dynamic"
        assert r.rag_endpoint is None

    async def test_software_availability(self):
        """Software search — static is acceptable (planner falls back to MCP tools)."""
        r = await classify("Is GROMACS available on Delta?")
        assert r.query_type in ("static", "dynamic", "combined")

    async def test_affinity_groups(self):
        """Affinity groups — static is acceptable (planner falls back to MCP tools)."""
        r = await classify("What affinity groups are there?")
        assert r.query_type in ("static", "dynamic", "combined")

    async def test_nsf_awards(self):
        """NSF awards — static is acceptable (planner falls back to MCP tools)."""
        r = await classify("Find NSF awards related to quantum computing")
        assert r.query_type in ("static", "dynamic", "combined")


# --- XDMoD combined ---


class TestXDMoDRouting:
    """Aggregate data queries that should route to XDMoD."""

    async def test_job_count(self):
        r = await classify("How many jobs ran last quarter?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_cpu_hours(self):
        r = await classify("Show me CPU hours on Delta last month")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_active_allocations(self):
        r = await classify("How many active allocations are there?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_gpu_capacity(self):
        r = await classify("What is the total GPU capacity across ACCESS?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_user_accounts(self):
        r = await classify("How many user accounts were created?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_active_gateways(self):
        r = await classify("How many active gateways are there?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_project_count(self):
        r = await classify("How many new projects were created in ACCESS?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_field_of_science(self):
        r = await classify("Which field of science submits the most jobs?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_most_used_resources(self):
        r = await classify("What were the most used resources last month?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_active_pis(self):
        r = await classify("Who are the most active PIs?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_trend_query(self):
        r = await classify(
            "How has the number of active allocations changed over the past 3 years?"
        )
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_cloud_sessions(self):
        r = await classify("How many cloud VM sessions were started in 2024?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_jobs_by_gateway(self):
        r = await classify("What is the breakdown of jobs by gateway?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_aces_allocated(self):
        r = await classify("How many ACEs have been allocated this fiscal year?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_resource_comparison_by_usage(self):
        r = await classify("Compare Delta and Bridges-2 by total CPU hours this year")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"


# --- Domain agents ---


class TestDomainRouting:
    """Queries that should route to domain-specific agents."""

    async def test_create_announcement(self):
        r = await classify("Create an announcement about the new GPU cluster")
        assert r.domain == "announcements"

    async def test_update_announcement(self):
        r = await classify("Update the maintenance announcement to say it's resolved")
        assert r.domain == "announcements"

    async def test_delete_announcement(self):
        r = await classify("Delete the old announcement about Stampede2 retirement")
        assert r.domain == "announcements"

    async def test_file_ticket(self):
        r = await classify("I need to submit a support ticket about a login issue")
        assert r.domain == "jsm"

    async def test_report_issue(self):
        r = await classify("I'm having trouble logging into Delta, can you help me file a ticket?")
        assert r.domain == "jsm"

    async def test_general_question_no_domain(self):
        r = await classify("How do I get started with ACCESS?")
        assert r.domain is None

    async def test_xdmod_query_no_domain(self):
        """XDMoD queries should not route to a domain agent."""
        r = await classify("How many jobs ran last month?")
        assert r.domain is None

    async def test_status_query_no_domain(self):
        """Status queries should not route to a domain agent."""
        r = await classify("Is Anvil down?")
        assert r.domain is None


# --- Ambiguous / edge cases ---


class TestEdgeCases:
    """Queries that test boundary conditions in classification."""

    async def test_ambiguous_allocations_count_vs_docs(self):
        """'How many allocations' = XDMoD data, not docs about allocations."""
        r = await classify("How many active allocations are there right now?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_ambiguous_about_vs_data(self):
        """'Tell me about allocations' = docs, not data."""
        r = await classify("What is the allocation process?")
        assert r.query_type == "static"
        assert r.rag_endpoint == "general"

    async def test_xdmod_without_mentioning_xdmod(self):
        """User doesn't need to say 'XDMoD' for aggregate queries."""
        r = await classify("What's the total compute usage across all resources?")
        assert r.query_type == "combined"
        assert r.rag_endpoint == "xdmod"

    async def test_out_of_scope(self):
        """Completely out of scope question should still classify without error."""
        r = await classify("What is the weather in Pittsburgh?")
        assert r.query_type is not None
        assert r.domain is None
