"""Pure comparison between a live tool catalog and the write deny-list."""

from __future__ import annotations

from dataclasses import dataclass

from . import HEURISTIC_EXCEPTIONS, looks_like_write


@dataclass(frozen=True)
class DriftReport:
    """Findings from comparing catalog tool names against the deny-list.

    ``unguarded`` is the fail-open case and the reason this guard exists: a
    live tool that looks like a write but is not stripped under READ_ONLY.

    ``stale`` is informational — a deny-list entry no longer in the catalog.
    It is not a safety problem (stripping a tool that does not exist is
    harmless), but it signals a renamed or retired tool worth cleaning up,
    and a rename usually means its replacement is now unguarded.

    ``waived`` records exceptions that were actually exercised, so a waiver
    that stops applying can be noticed and removed.
    """

    unguarded: tuple[str, ...] = ()
    stale: tuple[str, ...] = ()
    waived: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """True when nothing is unguarded. Stale entries do not fail the run."""
        return not self.unguarded


def check_drift(catalog_tool_names: set[str], deny_list: frozenset[str]) -> DriftReport:
    """Compare live tool names against the write deny-list.

    Args:
        catalog_tool_names: Every tool name the live catalog advertises.
        deny_list: ``WRITE_MCP_TOOL_NAMES`` — the names stripped under READ_ONLY.

    Returns:
        A ``DriftReport``. ``unguarded`` names are write-looking tools missing
        from the deny-list; each is a fail-open hole.
    """
    suspected = {name for name in catalog_tool_names if looks_like_write(name)}
    waived = suspected & set(HEURISTIC_EXCEPTIONS)
    unguarded = suspected - deny_list - waived
    stale = deny_list - catalog_tool_names

    return DriftReport(
        unguarded=tuple(sorted(unguarded)),
        stale=tuple(sorted(stale)),
        waived=tuple(sorted(waived)),
    )
