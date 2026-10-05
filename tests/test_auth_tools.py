"""Unit tests for the shared tool-needs-identity helper."""

from __future__ import annotations

from src.agent.auth_tools import ToolIdentityClass, tool_identity_class, tool_needs_identity


class TestToolNeedsIdentity:
    def test_write_tool_needs_identity(self):
        assert tool_needs_identity("register_for_event") is True

    def test_auth_read_by_prefix_needs_identity(self):
        assert tool_needs_identity("get_my_rp_accounts") is True

    def test_auth_read_by_explicit_set_needs_identity(self):
        assert tool_needs_identity("get_rp_account") is True

    def test_unauth_read_does_not_need_identity(self):
        assert tool_needs_identity("search_resources") is False

    def test_compositional_get_tool_does_not_need_identity(self):
        # get_raw_data is COMPOSITIONAL (XDMoD plumbing), not auth-read — proves
        # tool_needs_identity is "write OR auth-read", not "not a read".
        assert tool_needs_identity("get_raw_data") is False


class TestToolIdentityClass:
    def test_write_classification(self):
        assert tool_identity_class("register_for_event") is ToolIdentityClass.WRITE

    def test_auth_read_classification(self):
        assert tool_identity_class("get_rp_account") is ToolIdentityClass.AUTH_READ

    def test_no_identity_classification_for_unauth_read(self):
        assert tool_identity_class("search_resources") is ToolIdentityClass.NONE

    def test_no_identity_classification_for_compositional(self):
        assert tool_identity_class("get_raw_data") is ToolIdentityClass.NONE
