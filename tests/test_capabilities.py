"""Tests for capability registry and rating logic."""

import pytest

from src.agent.domains.capabilities import (
    CATEGORIES,
    GENERAL_CAPABILITIES,
    CapabilityRegistry,
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
        registry = CapabilityRegistry(
            capabilities=list(GENERAL_CAPABILITIES),
            categories=CATEGORIES,
            disabled_ids={"check_usage", "search_nsf_awards"},
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
        # Find ask_question (requires_auth=True)
        for cat in result:
            for cap in cat["capabilities"]:
                if cap["id"] == "ask_question":
                    assert cap.get("locked") is True
                    return
        pytest.fail("ask_question not found")

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
        # Auth-required caps should not appear
        assert "Check usage" not in prompt

    def test_system_prompt_includes_all_for_auth(self, registry):
        prompt = registry.get_system_prompt_section(authenticated=True)
        assert "Check usage" in prompt
        assert "Check system status" in prompt


class TestCapabilityInference:
    def test_infer_from_domain(self, registry):
        assert registry.infer_capability_id("jsm") == "open_ticket"
        assert registry.infer_capability_id("announcements") == "search_announcements"

    def test_infer_from_tools(self, registry):
        assert (
            registry.infer_capability_id(None, ["system-status__get_infrastructure_news"])
            == "check_system_status"
        )
        assert registry.infer_capability_id(None, ["events__search_events"]) == "browse_events"

    def test_infer_defaults_to_ask_question(self, registry):
        assert registry.infer_capability_id(None) == "ask_question"
        assert registry.infer_capability_id(None, ["unknown_tool"]) == "ask_question"
