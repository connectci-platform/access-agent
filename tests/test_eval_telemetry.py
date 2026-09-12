"""The eval CLI must emit telemetry like the API server does.

The eval runs as its own process, so it gets nothing from src/main.py's
lifespan hooks. Before this wiring, eval spans went to a no-op provider and
exported nothing — a production eval was invisible in Honeycomb. Investigating
a 24% question failure rate on 2026-09-11 turned up no traces at all, and the
diagnosis had to fall back to log greps.
"""

from unittest.mock import patch

import pytest

import src.eval.__main__ as em


def _run_cli(argv: list[str]) -> None:
    with patch.object(em.sys, "argv", ["prog", *argv]):
        em.main()


def test_run_initializes_and_shuts_down_telemetry():
    calls: list[str] = []
    with (
        patch.object(em, "init_telemetry", lambda **_: calls.append("init")),
        patch.object(em, "shutdown_telemetry", lambda: calls.append("shutdown")),
        patch.object(em, "_handle_coverage", lambda _a: calls.append("handler")),
    ):
        _run_cli(["coverage"])
    assert calls == ["init", "handler", "shutdown"]


def test_telemetry_flushes_even_when_the_handler_raises():
    # The run worth tracing is the one that crashed. Spans sit in a batch
    # processor, so without a flush on the failure path the CLI exits and takes
    # the evidence with it.
    calls: list[str] = []

    def _boom(_args: object) -> None:
        calls.append("handler")
        raise RuntimeError("eval blew up")

    with (
        patch.object(em, "init_telemetry", lambda **_: calls.append("init")),
        patch.object(em, "shutdown_telemetry", lambda: calls.append("shutdown")),
        patch.object(em, "_handle_coverage", _boom),
        pytest.raises(RuntimeError, match="eval blew up"),
    ):
        _run_cli(["coverage"])
    assert calls == ["init", "handler", "shutdown"]


def test_unknown_command_exits_without_initializing_telemetry():
    # A usage error should not spin up an exporter just to tear it down.
    calls: list[str] = []
    with (
        patch.object(em, "init_telemetry", lambda **_: calls.append("init")),
        patch.object(em, "shutdown_telemetry", lambda: calls.append("shutdown")),
        pytest.raises(SystemExit),
    ):
        _run_cli([])
    assert calls == []


def test_eval_traces_are_tagged_as_eval_not_live_traffic():
    # service.name is the Honeycomb dataset by design, so the component
    # attribute is what separates an eval run from real user requests.
    seen: dict[str, object] = {}
    with (
        patch.object(em, "init_telemetry", lambda **kw: seen.update(kw)),
        patch.object(em, "shutdown_telemetry", lambda: None),
        patch.object(em, "_handle_coverage", lambda _a: None),
    ):
        _run_cli(["coverage"])
    assert seen["service_name"] == "access-agent-eval"
