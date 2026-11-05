"""Shared runtime predicate: does a tool require a resolved identity to call safely?

Single-sourced from the same write/auth-read name sets and regex fallbacks that
``src.eval.coverage.structural_class`` uses for its eval-only coverage audit.
This module has no dependency on ``src.eval`` (and must not gain one) — it is
the runtime-safe home so both the eval coverage audit and runtime middleware
(the requires-identity tool gate) import the same predicates instead of each
maintaining its own copy.

Classification mirrors ``structural_class``'s write and auth-read checks
exactly, but is deliberately narrower: ``structural_class`` has four classes
(write / composition / auth-read / unauth-read) and checks composition BEFORE
auth-read, so a handful of ``get_*`` tools (``get_raw_data``,
``get_dimension_values``, ``get_chart_link``, ...) are compositional XDMoD
plumbing, not auth-read, and must NOT be treated as needing identity.
``tool_needs_identity`` is therefore "write OR auth-read" only — it is NOT
"not a read." Composition is intentionally excluded.

Known limit: the WRITE/AUTH_READ name sets (plus their regex fallbacks) are
hand-maintained in ``src.agent.domains.capabilities``. They are a floor, not a
proof — a newly added write or private-data tool that doesn't match an
existing name or prefix pattern is served unguarded. For an anonymous caller
the MCP server still refuses an identity-gated call itself. For a signed-in
caller whose token lacks ``agent:write``, this classification IS the
boundary: the agent sends ``X-Acting-User``, so a write tool classified as
NONE runs as that user. The access-mcp servers now publish each tool's
``readOnlyHint``, and ``tool_identity_class`` treats ``readOnlyHint: false`` as
a write, which closes that gap for hinted tools; the nightly writeguard check
reports any tool without a hint.
"""

from __future__ import annotations

import re
from enum import Enum

from src.agent.domains.capabilities import (
    AUTH_READ_MCP_TOOL_NAMES,
    WRITE_MCP_TOOL_NAMES,
)

# Mirrors src/eval/coverage.py's _AUTH_READ_RE / _WRITE_PREFIX_RE exactly — kept
# as separate compiled patterns (rather than importing coverage.py's) so this
# module has zero dependency on src.eval; coverage.py imports FROM here instead.
_AUTH_READ_RE = re.compile(r"^(get_my_|authenticate$|complete_authentication$)")
_WRITE_PREFIX_RE = re.compile(r"^(create_|update_|delete_|register_|cancel_|report_)")


class ToolIdentityClass(Enum):
    """Why (if at all) a tool needs a resolved identity before it can be called."""

    WRITE = "write"
    AUTH_READ = "auth_read"
    NONE = "none"


def tool_identity_class(name: str, read_only: bool | None = None) -> ToolIdentityClass:
    """Classify a tool as WRITE, AUTH_READ, or NONE for identity-gating purposes.

    ``read_only`` is the tool server's own ``annotations.readOnlyHint``, when
    the catalog carries one. It can only add WRITE: ``read_only=False`` makes a
    tool a write whatever its name, which catches a write the name rules miss.
    ``read_only=True`` never removes WRITE, so one wrong hint can't turn a
    known write into a read, and the names stay authoritative for every tool
    they cover. ``None`` means no hint, and the name rules alone decide.

    Write is checked before auth-read (a tool can't be both; WRITE_MCP_TOOL_NAMES
    and AUTH_READ_MCP_TOOL_NAMES are disjoint in practice, but write takes
    priority if that ever changes). Composition and unauth-read both collapse to
    NONE here — this classifier doesn't need to distinguish them, unlike
    ``structural_class`` which reports four classes for the eval coverage audit.
    """
    if name in WRITE_MCP_TOOL_NAMES or _WRITE_PREFIX_RE.match(name) or read_only is False:
        return ToolIdentityClass.WRITE
    if name in AUTH_READ_MCP_TOOL_NAMES or _AUTH_READ_RE.match(name):
        return ToolIdentityClass.AUTH_READ
    return ToolIdentityClass.NONE


def tool_needs_identity(name: str) -> bool:
    """True iff calling tool `name` requires a resolved identity.

    True for write tools and auth-read tools; False for everything else,
    including compositional reads like ``get_raw_data`` that are plumbing, not
    user-identity-scoped. Thin bool wrapper over ``tool_identity_class`` — callers
    that also need the write-vs-auth-read distinction (e.g. to pick a coarse
    "write_action" vs "private_data" reason string) should call
    ``tool_identity_class`` directly instead of re-deriving it from this bool.
    """
    return tool_identity_class(name) is not ToolIdentityClass.NONE
