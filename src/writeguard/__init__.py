"""Drift guard for the READ_ONLY write-tool deny-list.

``WRITE_MCP_TOOL_NAMES`` (src/agent/domains/capabilities.py) is the sole guard
that strips write-capable MCP tools under ``READ_ONLY``. It is hand-maintained,
so it silently falls out of sync whenever a new write tool ships in access_mcp
— and a missing name means the tool is NOT stripped. The failure mode is
fail-open, and nothing today notices (issue #200).

The MCP catalog carries no machine-readable write/read annotation (tools expose
only name/description/parameters), so write-ness cannot be derived. This guard
therefore uses tool NAMES as a suspicion signal — deliberately as an alarm, not
as enforcement:

* Enforcement stays with the explicit deny-list. A name heuristic used to strip
  tools would over-block silently (a read tool called ``add_bookmark`` would
  vanish from READ_ONLY runs with no error) and would still miss writes named
  outside the convention (``submit_x``, ``publish_y``).
* This guard only *reports*: any live tool whose name looks like a write but is
  absent from the deny-list is a finding. That converts a silent fail-open into
  a loud nightly failure.

Reviewed read-only tools that trip the heuristic go in ``HEURISTIC_EXCEPTIONS``
with a reason, so a waiver is a deliberate, reviewable act.

**Known limit — read this before trusting a green run.** A write tool named
outside ``WRITE_NAME_PREFIXES`` (``approve_x``, ``set_y``, ``join_z``) ships
unguarded *and* unflagged, and this guard reports OK. It does NOT detect a
break in the naming convention; it detects new tools that follow it. The
prefix list is therefore a floor, not a proof, and every addition to it should
be treated as widening a net that starts with known holes. What the guard does
buy: the 2026-08-15 events incident (nine write tools shipped at once, all
matching the convention) would have been caught the next morning instead of by
hand.

The durable fix is upstream: MCP servers should declare their own write-ness
(an annotation in access_mcp), at which point this heuristic is replaced by
the declaration and the hole closes. A complementary check belongs in
access_mcp's own CI, where a new write tool can be caught at the moment it is
introduced rather than a night later in a different repo.
"""

from __future__ import annotations

# Verb prefixes that suggest a mutating operation. The first group is observed
# in the 17 currently-deny-listed tools; the rest are plausible next writes for
# this domain (allocations, registrations, memberships, moderation). Note the
# observed group is not evidence the heuristic works — those prefixes were read
# off the very tools they now match. Recall against names nobody has written
# yet is unmeasured, which is why the module docstring calls this a floor.
WRITE_NAME_PREFIXES: tuple[str, ...] = (
    # observed in the current deny-list
    "add_",
    "cancel_",
    "create_",
    "delete_",
    "edit_",
    "register_",
    "report_",
    "restore_",
    "send_",
    "update_",
    # plausible, not yet observed
    "acknowledge_",
    "approve_",
    "archive_",
    "assign_",
    "attach_",
    "bulk_",
    "clone_",
    "confirm_",
    "disable_",
    "dismiss_",
    "duplicate_",
    "enable_",
    "exchange_",
    "flag_",
    "import_",
    "invite_",
    "join_",
    "leave_",
    "link_",
    "merge_",
    "move_",
    "post_",
    "publish_",
    "remove_",
    "rename_",
    "renew_",
    "request_",
    "reset_",
    "revoke_",
    "schedule_",
    "set_",
    "submit_",
    "subscribe_",
    "sync_",
    "transfer_",
    "unpublish_",
    "unschedule_",
    "unsubscribe_",
    "upload_",
)

# MCP servers that own at least one deny-listed write tool. The guard asserts
# the catalog actually contains each of these before it may report green: a
# tool on a missing server is absent from the catalog, and absence is
# indistinguishable from "correctly deny-listed" to a name-based check.
#
# Not derivable from the catalog itself — quick_lookup only maps tools that are
# PRESENT, which is circular for detecting a missing server. Kept here, small
# and asserted: tests/test_writeguard.py checks it against the deny-list's real
# owners so this map cannot quietly fall behind either.
SERVERS_OWNING_WRITE_TOOLS: frozenset[str] = frozenset(
    {
        "announcements",
        "events",
        "jsm",
    }
)

# Tools whose names trip WRITE_NAME_PREFIXES but which are genuinely read-only.
# Each entry is a reviewed waiver: map tool name -> why it is not a write.
# Empty today; add here (with a reason) rather than loosening the prefixes.
HEURISTIC_EXCEPTIONS: dict[str, str] = {}


def looks_like_write(tool_name: str) -> bool:
    """True if ``tool_name`` matches the mutating-verb naming convention."""
    return tool_name.startswith(WRITE_NAME_PREFIXES)
