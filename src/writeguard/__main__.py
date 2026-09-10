"""Write-tool deny-list drift guard CLI.

Fetches the live MCP catalog (no LLM) and reports any tool whose name looks
like a write but is missing from ``WRITE_MCP_TOOL_NAMES`` — a tool that would
NOT be stripped under READ_ONLY. Exits non-zero on such a finding so a nightly
step can key off the exit code alone. See issue #200.

Run inside the agent container or on a host with MCP access:
    python -m src.writeguard
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from src.agent.domains.capabilities import WRITE_MCP_TOOL_NAMES
from src.tools import get_catalog_aggregator

from . import HEURISTIC_EXCEPTIONS, SERVERS_OWNING_WRITE_TOOLS
from .check import check_drift

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


async def _fetch_catalog() -> dict[str, Any]:  # pragma: no cover - live MCP fetch
    """The live aggregated catalog.

    Excluded from coverage: this is the network boundary and nothing more.
    ``main_argv`` takes it as an injected ``_fetch``, and every line that
    *interprets* the result — including the coverage checks below, where the
    dangerous bugs live — is tested against synthetic catalogs.
    """
    catalog: dict[str, Any] = await get_catalog_aggregator().fetch_catalog(force_refresh=True)
    return catalog


def _tool_names(catalog: dict[str, Any]) -> set[str]:
    """Every tool name the catalog advertises."""
    return {
        tool.get("name", "")
        for server in catalog.get("servers", [])
        for tool in server.get("tools", [])
        if tool.get("name")
    }


def _unavailable_servers(catalog: dict[str, Any]) -> tuple[str, ...]:
    """Servers the aggregator could not reach.

    ``fetch_catalog`` does not fail on a per-server error — it records
    ``status: "unavailable"`` with an empty tool list and carries on. A tool
    on an unreachable server is therefore ABSENT from the catalog, which to a
    name-based check is indistinguishable from "correctly deny-listed". That
    is how this guard could otherwise report a confident green while having
    verified almost nothing.
    """
    return tuple(
        sorted(
            server.get("server", "?")
            for server in catalog.get("servers", [])
            if server.get("status") != "available"
        )
    )


def _unseen_denied_servers(catalog: dict[str, Any]) -> tuple[str, ...]:
    """Write-owning servers with no entry in the catalog at all.

    Distinct from ``_unavailable_servers``: those are present-but-unreachable.
    These are absent entirely, which is what the capability filter does when
    ``DISABLED_CAPABILITIES`` drops a server (``jsm`` and ``announcements``
    are owned solely by write capabilities, so disabling those drops the
    server and hides its write tools from this check).
    """
    present = {server.get("server", "") for server in catalog.get("servers", [])}
    return tuple(sorted(SERVERS_OWNING_WRITE_TOOLS - present))


def main_argv(
    argv: list[str] | None = None,
    _fetch: Callable[[], Coroutine[object, object, dict[str, Any]]] = _fetch_catalog,
) -> None:
    del argv  # no CLI flags; parameter kept for parity with probe/redteam main_argv
    try:
        catalog = asyncio.run(_fetch())
    except Exception as exc:  # never a bare traceback for a nightly step
        print(f"write-guard failed to fetch the catalog: {exc}")
        raise SystemExit(1) from exc

    tool_names = _tool_names(catalog)

    if not tool_names:
        # An empty catalog would make every check vacuously pass — refuse to
        # report a green guard when nothing was actually checked.
        print("write-guard: catalog returned no tools; cannot verify the deny-list")
        raise SystemExit(1)

    # A missing tool and a deny-listed tool look identical to a name check, so
    # the guard must prove it could SEE everything before it may report green.
    # Both failure modes below would otherwise produce a confident "OK" over an
    # unverified deny-list — the exact fail-open this guard exists to catch.
    if unreachable := _unavailable_servers(catalog):
        print(f"write-guard: servers unreachable: {', '.join(unreachable)}")
        print("Their tools are absent from the catalog, so the deny-list cannot be verified.")
        raise SystemExit(1)

    if unseen := _unseen_denied_servers(catalog):
        print(f"write-guard: no catalog entry for server(s): {', '.join(unseen)}")
        print(
            "A deny-listed tool lives there, so it cannot be checked. This happens when "
            "DISABLED_CAPABILITIES drops a server the deny-list still covers (jsm and "
            "announcements are owned solely by write capabilities). Run the guard with "
            "those capabilities enabled."
        )
        raise SystemExit(1)

    report = check_drift(tool_names, WRITE_MCP_TOOL_NAMES)

    print(f"checked {len(tool_names)} live tools against {len(WRITE_MCP_TOOL_NAMES)} deny-listed")
    for name in report.waived:
        print(f"WAIVED: {name} — {HEURISTIC_EXCEPTIONS[name]}")
    for name in report.stale:
        print(f"STALE: {name} is deny-listed but not in the live catalog (renamed or retired?)")

    if report.ok:
        print("OK: every write-looking tool is stripped under READ_ONLY")
        raise SystemExit(0)

    for name in report.unguarded:
        print(f"UNGUARDED: {name} looks like a write but is not in WRITE_MCP_TOOL_NAMES")
    print(
        "\nThese tools are NOT stripped under READ_ONLY. Add each to "
        "WRITE_MCP_TOOL_NAMES in src/agent/domains/capabilities.py, or — if it "
        "is genuinely read-only — to HEURISTIC_EXCEPTIONS in src/writeguard "
        "with a reason."
    )
    raise SystemExit(1)


def main() -> None:  # entrypoint keeps its name for `python -m src.writeguard`
    main_argv()


if __name__ == "__main__":
    main()
