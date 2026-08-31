"""Synthetic per-tool health probe CLI. No LLM: builds an MCPClient the way the
app does (prod parity from env — MCP_SERVER_HOST etc., see src/config.py) and
replays PROBE_TABLE against the live stack. Exits non-zero on any backend error
so a nightly step can key off the exit code alone."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.tools.mcp_client import MCPClient

from .runner import ProbeResult, run_probe
from .table import PROBE_TABLE

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


def main_argv(
    argv: list[str] | None = None,
    _run: Callable[..., Coroutine[object, object, list[ProbeResult]]] = run_probe,
) -> None:
    del argv  # no CLI flags yet; parameter kept for interface parity with redteam's main_argv
    try:
        client = MCPClient()
        results: list[ProbeResult] = asyncio.run(_run(client, PROBE_TABLE))
    except Exception as exc:  # defense in depth: never a bare traceback for a nightly step
        print(f"probe run failed to execute: {exc}")
        raise SystemExit(1) from exc

    for result in results:
        if result.success:
            print(f"OK: {result.tool_name}")
        else:
            print(f"FAIL: {result.tool_name} — {result.error}")

    if any(not result.success for result in results):
        raise SystemExit(1)
    raise SystemExit(0)


def main() -> None:  # entrypoint keeps its name for `python -m src.probe`
    main_argv()


if __name__ == "__main__":
    main()
