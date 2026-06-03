"""Tests for capability registry and rating logic."""

from unittest.mock import patch

import pytest

from src.agent.domains.capabilities import (
    CATEGORIES,
    GENERAL_CAPABILITIES,
    CapabilityRegistry,
    _build_registry,
)
from src.agent.domains.config import Capability


@pytest.fixture
def registry():
    """Registry with general + a few domain capabilities."""
    domain_caps = [
        Capability(
            "open_ticket",
            "Open a help ticket",
            "Create a support ticket",
            "support",
            requires_auth=False,
        ),
        Capability(
            "search_announcements",
            "Search announcements",
            "Find ACCESS news",
            "explore",
            requires_auth=False,
        ),
        Capability(
            "manage_announcements",
            "Manage announcements",
            "Create and edit announcements",
            "content",
            requires_auth=True,
        ),
    ]
    return CapabilityRegistry(
        capabilities=list(GENERAL_CAPABILITIES) + domain_caps,
        categories=CATEGORIES,
    )


class TestCapabilityRegistry:
    def test_all_capabilities_loaded(self, registry):
        all_caps = registry.get_all()
        assert len(all_caps) == len(GENERAL_CAPABILITIES) + 3

    def test_get_by_id(self, registry):
        cap = registry.get_by_id("ask_question")
        assert cap is not None
        assert cap.label == "Ask a question"

    def test_get_by_id_missing(self, registry):
        assert registry.get_by_id("nonexistent") is None

    def test_disabled_capabilities_excluded(self):
        # The caller applies ENABLED/DISABLED filtering before constructing
        # the registry. _build_registry is the canonical path; this test
        # exercises the constructor directly with a pre-filtered list.
        filtered = [
            c for c in GENERAL_CAPABILITIES if c.id not in {"check_usage", "search_nsf_awards"}
        ]
        registry = CapabilityRegistry(
            capabilities=filtered,
            categories=CATEGORIES,
        )
        assert registry.get_by_id("check_usage") is None
        assert registry.get_by_id("search_nsf_awards") is None
        assert registry.get_by_id("ask_question") is not None

    def test_categories_only_includes_active(self, registry):
        cats = registry.get_categories()
        cat_ids = {c.id for c in cats}
        # All our test caps span these categories
        assert "general" in cat_ids
        assert "support" in cat_ids
        assert "explore" in cat_ids

    def test_categories_sorted_by_order(self, registry):
        cats = registry.get_categories()
        orders = [c.order for c in cats]
        assert orders == sorted(orders)


class TestCapabilityAuth:
    def test_anonymous_sees_locked_on_auth_required(self, registry):
        result = registry.get_by_category(authenticated=False)
        # Find manage_announcements (requires_auth=True — the only locked capability)
        for cat in result:
            for cap in cat["capabilities"]:
                if cap["id"] == "manage_announcements":
                    assert cap.get("locked") is True
                    return
        pytest.fail("manage_announcements not found")

    def test_anonymous_sees_unlocked_on_public(self, registry):
        result = registry.get_by_category(authenticated=False)
        for cat in result:
            for cap in cat["capabilities"]:
                if cap["id"] == "check_system_status":
                    assert "locked" not in cap
                    return
        pytest.fail("check_system_status not found")

    def test_authenticated_sees_nothing_locked(self, registry):
        result = registry.get_by_category(authenticated=True)
        for cat in result:
            for cap in cat["capabilities"]:
                assert "locked" not in cap, f"{cap['id']} should not be locked for auth users"

    def test_system_prompt_excludes_auth_caps_for_anon(self, registry):
        prompt = registry.get_system_prompt_section(authenticated=False)
        assert "Check system status" in prompt
        assert "Browse events" in prompt
        assert "Check usage" in prompt  # XDMoD is public
        # Auth-required caps should not appear
        assert "Manage your announcements" not in prompt

    def test_system_prompt_includes_all_for_auth(self, registry):
        prompt = registry.get_system_prompt_section(authenticated=True)
        assert "Check usage" in prompt
        assert "Check system status" in prompt


class TestCapabilityInference:
    def test_infer_from_domain(self, registry):
        # These assertions depend on the real domain registry: the JSM and
        # announcements configs in src/agent/domains/{jsm,announcements}.py
        # declare open_ticket and search_announcements as their first
        # capability respectively. If those files change order, this test
        # will fail — which is intentional: domain attribution should
        # prefer reads, so reads should stay declared first.
        assert registry.infer_capability_id("jsm") == "open_ticket"
        assert registry.infer_capability_id("announcements") == "search_announcements"

    def test_infer_from_tools(self, registry):
        assert (
            registry.infer_capability_id(None, ["system-status__get_infrastructure_news"])
            == "check_system_status"
        )
        assert registry.infer_capability_id(None, ["events__search_events"]) == "browse_events"

    def test_infer_prefers_read_over_write_on_shared_server(self):
        # search_announcements and manage_announcements share the
        # 'announcements' server. Registration order (read-before-write)
        # determines which one wins attribution. Uses the real
        # _build_registry so the domain capabilities' McpBackend entries
        # are present (unlike the shared fixture).
        reg = _build_registry()
        assert (
            reg.infer_capability_id(None, ["announcements__create_announcement"])
            == "search_announcements"
        )

    def test_infer_defaults_to_ask_question(self, registry):
        assert registry.infer_capability_id(None) == "ask_question"
        assert registry.infer_capability_id(None, ["unknown_tool"]) == "ask_question"

    def test_infer_fallback_order_when_ask_question_disabled(self):
        # With ask_question excluded, the fallback chain should move to
        # the next preferred capability (ask_xdmod_question).
        filtered = [c for c in GENERAL_CAPABILITIES if c.id != "ask_question"]
        reg = CapabilityRegistry(capabilities=filtered, categories=CATEGORIES)
        assert reg.infer_capability_id(None) == "ask_xdmod_question"

    def test_infer_returns_unknown_sentinel_when_all_fallbacks_disabled(self):
        # Extreme case: every capability in the fallback chain is off.
        fallback_ids = {
            "ask_question",
            "ask_xdmod_question",
            "ask_about_resource",
            "check_allocations",
            "check_system_status",
        }
        filtered = [c for c in GENERAL_CAPABILITIES if c.id not in fallback_ids]
        reg = CapabilityRegistry(capabilities=filtered, categories=CATEGORIES)
        assert reg.infer_capability_id(None) == "unknown"


# ── Backend-derived query methods ───────────────────────────────────


class TestBackendQueries:
    """Tests for enabled_mcp_servers/enabled_rag_endpoints/scoped_rag_enabled."""

    def _registry_from_general(self) -> CapabilityRegistry:
        """Build a registry from the real GENERAL_CAPABILITIES list."""
        return CapabilityRegistry(
            capabilities=list(GENERAL_CAPABILITIES),
            categories=CATEGORIES,
        )

    def test_enabled_mcp_servers_default(self):
        reg = self._registry_from_general()
        servers = reg.enabled_mcp_servers()
        # Every MCP-backed capability in GENERAL_CAPABILITIES contributes
        assert "allocations" in servers
        assert "software-discovery" in servers
        assert "system-status" in servers
        assert "events" in servers
        assert "affinity-groups" in servers
        assert "xdmod" in servers
        assert "xdmod-data" in servers
        assert "nsf-awards" in servers

    def test_enabled_mcp_servers_respects_filter(self):
        # Only keep ask_question and check_allocations
        caps = [c for c in GENERAL_CAPABILITIES if c.id in {"ask_question", "check_allocations"}]
        reg = CapabilityRegistry(capabilities=caps, categories=CATEGORIES)
        assert reg.enabled_mcp_servers() == {"allocations"}

    def test_enabled_rag_endpoints_default(self):
        reg = self._registry_from_general()
        endpoints = reg.enabled_rag_endpoints()
        assert endpoints == {"general", "xdmod"}

    def test_enabled_rag_endpoints_only_general(self):
        caps = [c for c in GENERAL_CAPABILITIES if c.id == "ask_question"]
        reg = CapabilityRegistry(capabilities=caps, categories=CATEGORIES)
        assert reg.enabled_rag_endpoints() == {"general"}
        # No XDMoD RAG, no scoped
        assert not reg.scoped_rag_enabled()

    def test_scoped_rag_enabled_default(self):
        reg = self._registry_from_general()
        assert reg.scoped_rag_enabled() is True

    def test_scoped_rag_disabled_when_ask_about_resource_removed(self):
        caps = [c for c in GENERAL_CAPABILITIES if c.id != "ask_about_resource"]
        reg = CapabilityRegistry(capabilities=caps, categories=CATEGORIES)
        assert reg.scoped_rag_enabled() is False

    def test_capability_for_server(self):
        reg = self._registry_from_general()
        cap = reg.capability_for_server("allocations")
        assert cap is not None
        assert cap.id == "check_allocations"
        # The no-token charting server and the token-gated data server are
        # separate capabilities so xdmod-data can be gated off on its own.
        cap = reg.capability_for_server("xdmod")
        assert cap is not None
        assert cap.id == "check_usage"
        cap = reg.capability_for_server("xdmod-data")
        assert cap is not None
        assert cap.id == "extract_xdmod_data"
        assert reg.capability_for_server("nonexistent") is None

    def test_is_domain_enabled_default(self):
        reg = _build_registry()
        # Both announcements and jsm domains should be enabled by default
        assert reg.is_domain_enabled("announcements")
        assert reg.is_domain_enabled("jsm")
        assert not reg.is_domain_enabled("nonexistent_domain")


# ── ENABLED/DISABLED env var filter (deny-wins semantics) ───────────


class TestEnvVarFilter:
    """Tests for _build_registry with ENABLED_CAPABILITIES and DISABLED_CAPABILITIES."""

    def _build(self, enabled: str = "", disabled: str = "") -> CapabilityRegistry:
        # _build_registry does `from ...config import settings` at call time,
        # so patch the attributes on the real settings object.
        from src.config import settings

        with (
            patch.object(settings, "ENABLED_CAPABILITIES", enabled),
            patch.object(settings, "DISABLED_CAPABILITIES", disabled),
        ):
            return _build_registry()

    def test_no_filter_includes_all(self):
        reg = self._build()
        # 11 general + 2 announcements + 3 jsm = 16
        assert len(reg.get_all()) >= 11  # general caps definitely present
        assert reg.get_by_id("ask_question") is not None
        assert reg.get_by_id("check_allocations") is not None
        assert reg.get_by_id("manage_announcements") is not None

    def test_enabled_only_restricts_set(self):
        reg = self._build(enabled="ask_question,check_allocations")
        assert reg.get_by_id("ask_question") is not None
        assert reg.get_by_id("check_allocations") is not None
        assert reg.get_by_id("search_software") is None
        assert reg.get_by_id("manage_announcements") is None

    def test_disabled_removes_capability(self):
        reg = self._build(disabled="check_allocations")
        assert reg.get_by_id("ask_question") is not None
        assert reg.get_by_id("check_allocations") is None
        # Other capabilities still present
        assert reg.get_by_id("search_software") is not None

    def test_deny_wins_over_allow(self):
        """If a capability is in both ENABLED and DISABLED, it stays disabled."""
        reg = self._build(
            enabled="ask_question,check_allocations,search_software",
            disabled="check_allocations",
        )
        assert reg.get_by_id("ask_question") is not None
        assert reg.get_by_id("search_software") is not None
        assert reg.get_by_id("check_allocations") is None

    def test_rag_only_eval_config(self):
        """The 'agent-without-MCP' eval configuration."""
        reg = self._build(enabled="ask_question")
        assert reg.get_by_id("ask_question") is not None
        assert reg.enabled_rag_endpoints() == {"general"}
        assert reg.enabled_mcp_servers() == set()  # no MCP
        assert not reg.is_domain_enabled("announcements")
        assert not reg.is_domain_enabled("jsm")

    def test_disabling_all_rag_logs_warning(self, caplog):
        """When no RAG capability is enabled, operator gets a warning."""
        import logging

        caplog.set_level(logging.WARNING)
        reg = self._build(disabled="ask_question,ask_xdmod_question,ask_about_resource")
        assert reg.enabled_rag_endpoints() == set()
        assert any("No RAG capabilities enabled" in rec.message for rec in caplog.records)

    def test_disabling_domain_caps_disables_domain(self):
        reg = self._build(disabled="search_announcements,manage_announcements")
        assert not reg.is_domain_enabled("announcements")
        # jsm still enabled
        assert reg.is_domain_enabled("jsm")

    def test_empty_enabled_string_treated_as_unset(self):
        """Empty string should mean 'all', not 'none'."""
        reg = self._build(enabled="")
        assert len(reg.get_all()) > 5  # lots of capabilities

    def test_whitespace_stripped(self):
        reg = self._build(enabled="ask_question, check_allocations ")
        assert reg.get_by_id("ask_question") is not None
        assert reg.get_by_id("check_allocations") is not None
