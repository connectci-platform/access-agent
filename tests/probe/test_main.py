"""Tests for src.probe.__main__: per-tool pass/fail summary printing and the
exit-code disposition (0 all-pass / 1 any-backend-error / 1 defense-in-depth
catch when the injected _run itself raises)."""

from __future__ import annotations

import pytest

from src.probe import __main__ as cli
from src.probe.runner import ProbeResult


def test_main_exits_nonzero_on_any_backend_error(capsys):
    async def fake_run(client, table):
        return [
            ProbeResult("a", True, None),
            ProbeResult("b", False, "HTTP 400"),
        ]

    with pytest.raises(SystemExit) as exc_info:
        cli.main_argv([], _run=fake_run)

    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "OK: a" in out
    assert "FAIL: b — HTTP 400" in out


def test_main_exits_zero_when_all_pass(capsys):
    async def fake_run(client, table):
        return [
            ProbeResult("a", True, None),
            ProbeResult("b", True, None),
        ]

    with pytest.raises(SystemExit) as exc_info:
        cli.main_argv([], _run=fake_run)

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "OK: a" in out
    assert "OK: b" in out


def test_main_defense_in_depth_catches_run_exception(capsys):
    async def fake_run(client, table):
        raise RuntimeError("network exploded")

    with pytest.raises(SystemExit) as exc_info:
        cli.main_argv([], _run=fake_run)

    assert exc_info.value.code != 0
    out = capsys.readouterr().out
    assert "probe run failed to execute" in out
    assert "network exploded" in out


def test_main_entrypoint_delegates_to_main_argv(monkeypatch):
    # `main()` is the `python -m src.probe` entrypoint; it just delegates to
    # main_argv. Stub main_argv so this touches no network and only exercises
    # the entrypoint wrapper.
    called = False

    def fake_main_argv():
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "main_argv", fake_main_argv)
    cli.main()
    assert called
