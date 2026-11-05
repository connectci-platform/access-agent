"""Pure comparison between a live tool catalog and the write deny-list."""

from __future__ import annotations

from dataclasses import dataclass

from src.agent.auth_tools import ToolIdentityClass, tool_identity_class

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


@dataclass(frozen=True)
class HintReport:
    """Findings from comparing each tool's readOnlyHint with the name rules.

    ``unhinted``: tools whose server published no readOnlyHint, so the agent
    falls back to names for them. Every access-mcp server publishes one, so
    this means a server that is out of date or broken.

    ``contradicted``: tools published as read-only whose name the agent treats
    as a write. The agent keeps treating them as writes (a hint never removes
    WRITE), so this is a server marking a write wrongly, not a hole.

    ``unlisted``: tools published as making changes that are missing from
    ``WRITE_MCP_TOOL_NAMES``. The gate, the wrapper and READ_ONLY already
    handle them by the hint, but the eval coverage audit classifies by name
    only, so it lags until the name is added.
    """

    unhinted: tuple[str, ...] = ()
    contradicted: tuple[str, ...] = ()
    unlisted: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not (self.unhinted or self.contradicted or self.unlisted)


def check_hints(hints: dict[str, bool | None], deny_list: frozenset[str]) -> HintReport:
    """Compare each tool's published readOnlyHint with the agent's name rules.

    Args:
        hints: Tool name to its readOnlyHint (None when the server sent none).
        deny_list: ``WRITE_MCP_TOOL_NAMES``.
    """
    unhinted = {name for name, hint in hints.items() if hint is None}
    contradicted = {
        name
        for name, hint in hints.items()
        if hint is True and tool_identity_class(name) is ToolIdentityClass.WRITE
    }
    unlisted = {name for name, hint in hints.items() if hint is False and name not in deny_list}
    return HintReport(
        unhinted=tuple(sorted(unhinted)),
        contradicted=tuple(sorted(contradicted)),
        unlisted=tuple(sorted(unlisted)),
    )
