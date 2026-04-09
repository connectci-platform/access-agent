"""Tests for ToolRegistry capability-aware filtering.

The registry drops tools whose server is not owned by an enabled
capability, so the planner never sees tools that shouldn't be callable.
"""

from unittest.mock import patch

import pytest

from src.tools.registry import ToolRegistry


def _sample_catalog() -> dict:
    """Catalog with tools from several MCP servers used in tests."""
    return {
        "tools": [
            {"name": "allocations__search_projects", "description": "Search projects"},
            {"name": "allocations__get_stats", "description": "Get allocation stats"},
            {
                "name": "software-discovery__search_software",
                "description": "Search software",
            },
            {"name": "events__search_events", "description": "Search events"},
            {"name": "jsm__create_support_ticket", "description": "Create a ticket"},
            {
                "name": "announcements__search_announcements",
                "description": "Search announcements",
            },
        ],
        "quick_lookup": {
            "allocations__search_projects": {"server": "allocations"},
            "allocations__get_stats": {"server": "allocations"},
            "software-discovery__search_software": {"server": "software-discovery"},
            "events__search_events": {"server": "events"},
            "jsm__create_support_ticket": {"server": "jsm"},
            "announcements__search_announcements": {"server": "announcements"},
        },
    }


class TestCapabilityFilter:
    """The tool catalog should respect the capability registry's MCP server set."""

    def _build(self, enabled: str = "", disabled: str = "") -> ToolRegistry:
        """Build a ToolRegistry with the sample catalog under given env vars.

        Resets the capability registry singleton so env-var changes take
        effect before the filter runs.
        """
        from src.agent.domains import capabilities as cap_module
        from src.config import settings

        cap_module._registry = None  # force a fresh build
        with (
            patch.object(settings, "ENABLED_CAPABILITIES", enabled),
            patch.object(settings, "DISABLED_CAPABILITIES", disabled),
        ):
            return ToolRegistry(catalog=_sample_catalog())

    def test_default_keeps_all_known_servers(self):
        reg = self._build()
        # All sample servers are owned by default-enabled capabilities
        assert "allocations__search_projects" in reg.tools
        assert "allocations__get_stats" in reg.tools
        assert "software-discovery__search_software" in reg.tools
        assert "events__search_events" in reg.tools
        assert "jsm__create_support_ticket" in reg.tools
        assert "announcements__search_announcements" in reg.tools

    def test_disabling_check_allocations_drops_allocations_tools(self):
        reg = self._build(disabled="check_allocations")
        assert "allocations__search_projects" not in reg.tools
        assert "allocations__get_stats" not in reg.tools
        # Other servers unaffected
        assert "software-discovery__search_software" in reg.tools
        assert "events__search_events" in reg.tools

    def test_disabling_announcements_domain_drops_announcements_tools(self):
        reg = self._build(disabled="search_announcements,manage_announcements")
        assert "announcements__search_announcements" not in reg.tools

    def test_rag_only_eval_empties_catalog(self):
        """ENABLED_CAPABILITIES=ask_question is the 'agent-without-MCP' config."""
        reg = self._build(enabled="ask_question")
        assert len(reg.tools) == 0
        assert reg.tool_count == 0

    def test_rag_only_eval_also_clears_quick_lookup(self):
        """Filtered tools must not leak back via the quick_lookup fallback."""
        reg = self._build(enabled="ask_question")
        # get_server_for_tool should not resurrect filtered tools
        assert reg.get_server_for_tool("allocations__search_projects") is None

    def test_enabled_subset_keeps_matching_tools(self):
        reg = self._build(enabled="ask_question,check_allocations,browse_events")
        assert "allocations__search_projects" in reg.tools
        assert "events__search_events" in reg.tools
        # Not in the enabled set
        assert "software-discovery__search_software" not in reg.tools
        assert "jsm__create_support_ticket" not in reg.tools

    def test_deny_wins_in_tool_filter(self):
        """Consistent with the capability registry: deny always wins."""
        reg = self._build(
            enabled="ask_question,check_allocations,browse_events",
            disabled="check_allocations",
        )
        assert "allocations__search_projects" not in reg.tools
        assert "events__search_events" in reg.tools

    @pytest.fixture(autouse=True)
    def _reset_singleton(self):
        """Ensure every test starts and ends with a clean capability registry."""
        from src.agent.domains import capabilities as cap_module

        cap_module._registry = None
        yield
        cap_module._registry = None
