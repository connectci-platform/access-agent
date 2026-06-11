"""get_git_info falls back to build-time env stamps inside the container."""

import subprocess

from src.eval.runner import get_git_info


def _no_git(*args, **kwargs):
    raise FileNotFoundError("git not available")


class TestGetGitInfo:
    def test_env_fallback_when_git_unavailable(self, monkeypatch):
        monkeypatch.setattr(subprocess, "check_output", _no_git)
        monkeypatch.setenv("GIT_COMMIT", "abc1234def5678")
        monkeypatch.setenv("GIT_BRANCH", "main")
        info = get_git_info()
        assert info["commit"] == "abc1234def5678"
        assert info["branch"] == "main"

    def test_unknown_when_git_and_env_absent(self, monkeypatch):
        monkeypatch.setattr(subprocess, "check_output", _no_git)
        monkeypatch.delenv("GIT_COMMIT", raising=False)
        monkeypatch.delenv("GIT_BRANCH", raising=False)
        monkeypatch.setattr("src.eval.runner.settings.AGENT_VERSION", "")
        info = get_git_info()
        assert info["commit"] == "unknown"
        assert info["branch"] == "unknown"

    def test_agent_version_is_last_resort_for_commit(self, monkeypatch):
        monkeypatch.setattr(subprocess, "check_output", _no_git)
        monkeypatch.delenv("GIT_COMMIT", raising=False)
        monkeypatch.delenv("GIT_BRANCH", raising=False)
        monkeypatch.setattr("src.eval.runner.settings.AGENT_VERSION", "v1.2.3-4-gabc123")
        info = get_git_info()
        assert info["commit"] == "v1.2.3-4-gabc123"

    def test_real_git_preferred_over_env(self, monkeypatch):
        # In this repo git IS available — env stamps must not override it.
        monkeypatch.setenv("GIT_COMMIT", "should-not-be-used")
        info = get_git_info()
        assert info["commit"] != "should-not-be-used"
        assert len(info["commit"]) == 40  # a real SHA
