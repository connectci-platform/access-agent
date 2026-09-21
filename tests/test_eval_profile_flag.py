"""Tests for the `--profile-resource` eval CLI flag.

Pattern of tests/test_coverage_handler.py: build_parser() for argparse wiring;
direct unit calls for the sync/async split in _profile_from_args.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError


def _parser():
    from src.eval.__main__ import build_parser

    return build_parser()


class TestRunParserProfileResource:
    def test_run_parser_accepts_repeated_profile_resource(self):
        args = _parser().parse_args(
            [
                "run",
                "--profile-resource",
                "Delta GPU=delta",
                "--profile-resource",
                "Delta Storage",
            ]
        )
        assert args.profile_resource == ["Delta GPU=delta", "Delta Storage"]

    def test_default_is_none_or_empty(self):
        args = _parser().parse_args(["run"])
        assert not args.profile_resource


class TestProfileFromArgs:
    def test_profile_resource_without_slug_is_ungrouped(self):
        from src.eval.__main__ import _profile_from_args

        profile = _profile_from_args(["Delta Storage"])
        assert profile is not None
        resource = profile.allocated_resources[0]
        assert resource.name == "Delta Storage"
        assert resource.rp_slug is None
        assert resource.resource_id is None

    def test_profile_resource_with_slug_uses_it_verbatim(self):
        from src.eval.__main__ import _profile_from_args

        mock_cache = MagicMock()
        mock_cache.ensure_loaded = AsyncMock()
        mock_cache.list_slugs.return_value = ["delta"]

        with patch("src.eval.__main__.get_rp_cache", return_value=mock_cache):
            profile = _profile_from_args(["Delta GPU=delta"])

        resource = profile.allocated_resources[0]
        assert resource.rp_slug == "delta"

    def test_profile_from_args_unknown_slug_fails_fast(self):
        from src.eval.__main__ import ProfileArgError, _profile_from_args

        mock_cache = MagicMock()
        mock_cache.ensure_loaded = AsyncMock()
        mock_cache.list_slugs.return_value = [
            "aces",
            "anvil",
            "bridges2",
            "cloudbank",
            "delta",
            "deltaai",
            "derecho",
            "expanse",
            "fabric",
            "granite",
            "jetstream2",
            "kyric",
            "launch",
            "neocortex",
            "osg",
            "osn",
            "ranch",
            "repacss",
            "stampede3",
            "voyager",
        ]

        with (
            patch("src.eval.__main__.get_rp_cache", return_value=mock_cache),
            pytest.raises(ProfileArgError) as exc_info,
        ):
            _profile_from_args(["Delta GPU=deltagpu"])

        assert "deltagpu" in str(exc_info.value)
        assert "delta" in str(exc_info.value)

    def test_profile_from_args_reports_vocabulary_unavailable(self):
        from src.eval.__main__ import ProfileArgError, _profile_from_args

        mock_cache = MagicMock()
        mock_cache.ensure_loaded = AsyncMock()
        mock_cache.list_slugs.return_value = []

        with (
            patch("src.eval.__main__.get_rp_cache", return_value=mock_cache),
            pytest.raises(ProfileArgError) as exc_info,
        ):
            _profile_from_args(["Delta GPU=delta"])

        assert "unavailable" in str(exc_info.value)
        assert "unknown" not in str(exc_info.value).lower()

    def test_profile_from_args_skips_network_when_no_slugs_supplied(self):
        from src.eval.__main__ import _profile_from_args

        with patch("src.eval.__main__.get_rp_cache") as mock_get_cache:
            _profile_from_args(["Delta GPU", "Delta Storage"])

        mock_get_cache.assert_not_called()

    def test_profile_from_args_builds_mixed_profile(self):
        from src.eval.__main__ import _profile_from_args

        mock_cache = MagicMock()
        mock_cache.ensure_loaded = AsyncMock()
        mock_cache.list_slugs.return_value = ["delta"]

        with patch("src.eval.__main__.get_rp_cache", return_value=mock_cache):
            profile = _profile_from_args(["Delta GPU=delta", "Delta Storage"])

        assert [r.name for r in profile.allocated_resources] == ["Delta GPU", "Delta Storage"]
        assert profile.allocated_resources[0].rp_slug == "delta"
        assert profile.allocated_resources[1].rp_slug is None

    def test_profile_from_args_returns_none_without_flag(self):
        from src.eval.__main__ import _profile_from_args

        assert _profile_from_args([]) is None
        assert _profile_from_args(None) is None

    def test_handle_run_rejects_invalid_resource_name(self):
        from src.eval.__main__ import _profile_from_args

        with pytest.raises(ValidationError):
            _profile_from_args(["Delta\n#x"])


class TestHandleRun:
    def test_handle_run_resolves_profile_and_forwards_it(self):
        """_handle_run builds the profile from args and threads it to run_eval."""
        import argparse

        from src.eval.__main__ import _handle_run

        args = argparse.Namespace(
            questions="eval/questions/friendly_battery.json",
            system="agent_full",
            judge_model=None,
            profile_resource=["Delta Storage"],
        )

        with (
            patch("src.eval.scorer.run_eval", new_callable=AsyncMock) as mock_run_eval,
            patch("src.eval.report.print_run_summary") as mock_print,
        ):
            mock_run_eval.return_value = {"run_id": "r1"}
            _handle_run(args)

        profile = mock_run_eval.call_args.kwargs["profile"]
        assert profile.allocated_resources[0].name == "Delta Storage"
        mock_print.assert_called_once_with({"run_id": "r1"})

    def test_handle_run_rejects_invalid_resource_name_before_run_eval(self):
        import argparse

        from src.eval.__main__ import _handle_run

        args = argparse.Namespace(
            questions="eval/questions/friendly_battery.json",
            system="agent_full",
            judge_model=None,
            profile_resource=["Delta\n#x"],
        )

        with (
            patch("src.eval.scorer.run_eval", new_callable=AsyncMock) as mock_run_eval,
            pytest.raises(ValidationError),
        ):
            _handle_run(args)

        mock_run_eval.assert_not_called()

    def test_handle_run_logs_resolved_profile(self, caplog):
        """A forgotten '=slug' must be visible in the run output, not only
        discoverable later in eval_runs.metadata."""
        import argparse
        import logging

        from src.eval.__main__ import _handle_run

        args = argparse.Namespace(
            questions="eval/questions/friendly_battery.json",
            system="agent_full",
            judge_model=None,
            profile_resource=["Delta GPU=delta", "Delta Storage"],
        )

        mock_cache = MagicMock()
        mock_cache.ensure_loaded = AsyncMock()
        mock_cache.list_slugs.return_value = ["delta"]

        with (
            patch("src.eval.__main__.get_rp_cache", return_value=mock_cache),
            patch("src.eval.scorer.run_eval", new_callable=AsyncMock) as mock_run_eval,
            patch("src.eval.report.print_run_summary"),
            caplog.at_level(logging.INFO, logger="src.eval.__main__"),
        ):
            mock_run_eval.return_value = {"run_id": "r1"}
            _handle_run(args)

        messages = [r.message for r in caplog.records]
        assert any("Delta GPU: scoped (rp_name=delta)" in m for m in messages)
        assert any("Delta Storage: ungrouped (no rp_name)" in m for m in messages)
