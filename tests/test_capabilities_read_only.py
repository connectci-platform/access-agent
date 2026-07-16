"""Tests for the READ_ONLY capability guard (launch Phase 1).

The guard adds every write-capable capability ID into the disabled set
at registry build time when settings.READ_ONLY is True. Verifies that:

1. Without READ_ONLY, write capabilities remain enabled (baseline).
2. With READ_ONLY=true, write capabilities are disabled in the registry.
3. With READ_ONLY=true, read-only capabilities are NOT affected.
4. The WRITE_CAPABILITY_IDS constant lists exactly the known write caps.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def fresh_registry(monkeypatch):
    """Rebuild the capability registry from scratch under the current settings.

    The registry is a module-level singleton cached in _registry. We reset
    it before each test so settings changes are picked up.
    """
    from src.agent.domains import capabilities as caps_mod

    # Reset the module-level singleton before each test.
    monkeypatch.setattr(caps_mod, "_registry", None, raising=False)

    def _build():
        # Reset again to be extra-safe against test interleaving.
        monkeypatch.setattr(caps_mod, "_registry", None, raising=False)
        return caps_mod.get_capability_registry()

    return _build


def test_write_capability_ids_constant_matches_known_writes():
    """The WRITE_CAPABILITY_IDS set must list exactly the known write capabilities."""
    from src.agent.domains.capabilities import WRITE_CAPABILITY_IDS

    assert (
        frozenset(
            {
                "manage_announcements",
                "open_ticket",
                "report_login_problem",
                "report_security",
            }
        )
        == WRITE_CAPABILITY_IDS
    )


def test_write_mcp_tool_names_covers_all_known_write_tools():
    """The tool-name deny-list gates the loop and stamps invoked_write —
    every write-capable MCP tool must be listed exactly."""
    from src.agent.domains.capabilities import WRITE_MCP_TOOL_NAMES

    assert (
        frozenset(
            {
                "create_announcement",
                "update_announcement",
                "delete_announcement",
                "create_support_ticket",
                "create_login_ticket",
                "report_security_incident",
                "cancel_registration",
                "register_for_event",
            }
        )
        == WRITE_MCP_TOOL_NAMES
    )


def test_baseline_write_caps_enabled_without_read_only(monkeypatch, fresh_registry):
    """Sanity check: with READ_ONLY=false and no ENABLED/DISABLED env, writes are on."""
    monkeypatch.setattr("src.config.settings.READ_ONLY", False, raising=False)
    monkeypatch.setattr("src.config.settings.ENABLED_CAPABILITIES", "", raising=False)
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    registry = fresh_registry()

    for cap_id in (
        "manage_announcements",
        "open_ticket",
        "report_login_problem",
        "report_security",
    ):
        assert registry.get_by_id(cap_id) is not None, f"{cap_id} should be enabled"


def test_read_only_disables_all_write_capabilities(monkeypatch, fresh_registry):
    """With READ_ONLY=true, every write capability is absent from the registry."""
    monkeypatch.setattr("src.config.settings.READ_ONLY", True, raising=False)
    monkeypatch.setattr("src.config.settings.ENABLED_CAPABILITIES", "", raising=False)
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    registry = fresh_registry()

    for cap_id in (
        "manage_announcements",
        "open_ticket",
        "report_login_problem",
        "report_security",
    ):
        assert registry.get_by_id(cap_id) is None, f"{cap_id} must be disabled when READ_ONLY=true"


def test_read_only_does_not_affect_read_capabilities(monkeypatch, fresh_registry):
    """Read-only capabilities stay enabled when READ_ONLY=true."""
    monkeypatch.setattr("src.config.settings.READ_ONLY", True, raising=False)
    monkeypatch.setattr("src.config.settings.ENABLED_CAPABILITIES", "", raising=False)
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    registry = fresh_registry()

    # A sampling of read-only capabilities that should NOT be touched.
    for cap_id in (
        "ask_question",
        "check_allocations",
        "search_software",
        "check_system_status",
        "browse_events",
    ):
        assert registry.get_by_id(cap_id) is not None, (
            f"{cap_id} is read-only and should stay enabled under READ_ONLY=true"
        )


def test_read_only_overrides_explicit_enabled_list(monkeypatch, fresh_registry):
    """If READ_ONLY=true, even ENABLED_CAPABILITIES naming a write cap can't resurrect it."""
    monkeypatch.setattr("src.config.settings.READ_ONLY", True, raising=False)
    monkeypatch.setattr(
        "src.config.settings.ENABLED_CAPABILITIES",
        "ask_question,manage_announcements,open_ticket",
        raising=False,
    )
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    registry = fresh_registry()

    assert registry.get_by_id("ask_question") is not None  # allowed read
    assert registry.get_by_id("manage_announcements") is None  # blocked by READ_ONLY
    assert registry.get_by_id("open_ticket") is None  # blocked by READ_ONLY


def test_read_only_logged_at_startup(monkeypatch, fresh_registry, caplog):
    """READ_ONLY=true must produce a prominent log line at registry build time."""
    import logging

    monkeypatch.setattr("src.config.settings.READ_ONLY", True, raising=False)
    monkeypatch.setattr("src.config.settings.ENABLED_CAPABILITIES", "", raising=False)
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    with caplog.at_level(logging.WARNING):
        fresh_registry()

    assert any(
        "READ_ONLY" in rec.message and "active" in rec.message.lower() for rec in caplog.records
    ), "Expected a WARNING-level log line announcing READ_ONLY=true at startup"
