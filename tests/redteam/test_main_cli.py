"""Tests for src.redteam.__main__.main(): the flags-print loop, --emit-issue-body,
exit-code disposition (0 clean / 3 regression / 2 outage), and the RedteamOutage
catch (print + distinct outage issue body + SystemExit(2))."""

from __future__ import annotations

import pytest

from src.redteam import __main__ as cli
from src.redteam.gate import GateResult, JudgeOutage
from src.redteam.report import Flag, content_hash, issue_body


def _flag() -> Flag:
    return Flag(
        "wrapped__aligned__stop-sign", "complies", content_hash("BODY"), "candidate-regression"
    )


def test_main_prints_flags(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["redteam"])

    async def fake_run_from_env(*a, **k):
        return GateResult(flags=[_flag()], artifact_records=[])

    monkeypatch.setattr(cli, "run_from_env", fake_run_from_env)

    # A candidate-regression flag now exits 3 (the B3 fix — regression is never green).
    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 3
    out = capsys.readouterr().out
    assert "wrapped__aligned__stop-sign" in out
    assert "candidate-regression" in out


def test_main_no_flags_prints_nothing(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["redteam"])

    async def fake_run_from_env(*a, **k):
        return GateResult(flags=[], artifact_records=[])

    monkeypatch.setattr(cli, "run_from_env", fake_run_from_env)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert out == ""


def test_main_emit_issue_body_writes_regression_body(monkeypatch, tmp_path):
    issue_path = tmp_path / "issue_body.md"
    monkeypatch.setattr("sys.argv", ["redteam", "--emit-issue-body"])
    monkeypatch.setenv("REDTEAM_ISSUE_BODY", str(issue_path))

    flag = _flag()

    async def fake_run_from_env(*a, **k):
        return GateResult(flags=[flag], artifact_records=[])

    monkeypatch.setattr(cli, "run_from_env", fake_run_from_env)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 3
    assert issue_path.exists()
    assert issue_path.read_text() == issue_body([flag])


def test_main_emit_issue_body_skipped_when_no_flags(monkeypatch, tmp_path):
    issue_path = tmp_path / "issue_body.md"
    monkeypatch.setattr("sys.argv", ["redteam", "--emit-issue-body"])
    monkeypatch.setenv("REDTEAM_ISSUE_BODY", str(issue_path))

    async def fake_run_from_env(*a, **k):
        return GateResult(flags=[], artifact_records=[])

    monkeypatch.setattr(cli, "run_from_env", fake_run_from_env)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 0
    assert not issue_path.exists()


def test_main_judge_outage_exits_2_and_prints_distinct_message(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["redteam"])

    async def fake_run_from_env(*a, **k):
        raise JudgeOutage("judge failed on 5/5 judged samples (>= 50%) — aborting")

    monkeypatch.setattr(cli, "run_from_env", fake_run_from_env)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    out = capsys.readouterr().out
    # Message now names the outage's reason (shared handler for every RedteamOutage
    # subclass — judge/surface/errored), not a judge-only literal.
    assert "REDTEAM OUTAGE (judge)" in out
    assert "judge failed on 5/5" in out


def test_main_judge_outage_emits_distinct_outage_issue_body(monkeypatch, tmp_path):
    issue_path = tmp_path / "issue_body.md"
    monkeypatch.setattr("sys.argv", ["redteam", "--emit-issue-body"])
    monkeypatch.setenv("REDTEAM_ISSUE_BODY", str(issue_path))

    async def fake_run_from_env(*a, **k):
        raise JudgeOutage("judge failed on 5/5 judged samples (>= 50%) — aborting")

    monkeypatch.setattr(cli, "run_from_env", fake_run_from_env)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    body = issue_path.read_text()
    # distinct from the regression issue_body(): must NOT read as a safety finding
    assert "judge outage" in body.lower()
    assert "not a safety finding" in body.lower()
    assert "judge failed on 5/5" in body


def test_main_judge_outage_without_emit_flag_does_not_write_file(monkeypatch, tmp_path):
    issue_path = tmp_path / "issue_body.md"
    monkeypatch.setattr("sys.argv", ["redteam"])
    monkeypatch.setenv("REDTEAM_ISSUE_BODY", str(issue_path))

    async def fake_run_from_env(*a, **k):
        raise JudgeOutage("outage")

    monkeypatch.setattr(cli, "run_from_env", fake_run_from_env)

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2
    assert not issue_path.exists()
