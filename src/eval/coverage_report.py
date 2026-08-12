"""Rendering for the per-tool coverage audit (table / md / csv / json).

Caption honesty (per the design's adversarial review):
- facts columns are a question-attributed UPPER BOUND (a question invoking two
  tools attributes all its facts to both), so we never print a total row.
- distinct_arg_shapes is an ordinal exercise-variety COUNT, never a coverage %
  (the argument-mode denominator is not in the eval DB).
- write / auth-read tools sit near zero BY CONSTRUCTION: battery runs execute
  with acting_user=None, so they are structurally unreachable, not neglected.
"""

from __future__ import annotations

import csv
import io
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .coverage import ToolCoverage

_TIERS = [
    ("UNCOVERED-TESTABLE", "actionable gap: served user-facing read tool with no battery question"),
    ("EXERCISED-NOT-EVALUATED", "the loop calls it but no fact checks its output"),
    ("EVALUATED", "invoked and checked by at least one required fact"),
    (
        "UNCOVERED-COMPOSITIONAL",
        "helper/plumbing: covered via a parent capability, no standalone question",
    ),
    ("UNCOVERED-STRUCTURAL", "write / auth-read: unreachable under acting_user=None"),
]


def _summary(
    coverage: list[ToolCoverage], run_id: str, battery: str | None, model: str | None
) -> str:
    served = sum(1 for c in coverage if c.in_snapshot)
    invoked = sum(1 for c in coverage if c.invocations > 0)
    evaluated = sum(1 for c in coverage if c.invocations > 0 and c.facts_realized > 0)
    testable = sum(1 for c in coverage if c.tier == "UNCOVERED-TESTABLE")
    compositional = sum(1 for c in coverage if c.tier == "UNCOVERED-COMPOSITIONAL")
    structural = sum(1 for c in coverage if c.tier == "UNCOVERED-STRUCTURAL")
    return (
        f"{served} served -> {invoked} invoked -> {evaluated} evaluated "
        f"(facts_realized>0); {testable} uncovered-testable (actionable gaps); "
        f"{compositional} compositional (covered via parent); "
        f"{structural} structural (write/auth-read, acting_user=None). "
        f"run {run_id} | battery {battery or '?'} | model {model or '?'}"
    )


def _rows(coverage: list[ToolCoverage]) -> list[dict[str, object]]:
    out = []
    for c in coverage:
        enum = "; ".join(f"{k}={'/'.join(sorted(v))}" for k, v in sorted(c.enum_values.items()))
        out.append(
            {
                "tool": c.tool + ("" if c.in_snapshot else " (NOT-IN-SNAPSHOT)"),
                "class": c.structural,
                "tier": c.tier,
                "invocations": c.invocations,
                "total_calls": c.total_calls,
                "arg_shapes": c.distinct_arg_shapes,
                "enum_values_hit": enum,
                "facts_realized_ub": c.facts_realized,
            }
        )
    return out


def render_coverage(
    coverage: list[ToolCoverage],
    *,
    run_id: str,
    battery: str | None,
    model: str | None,
    warnings: list[str],
    fmt: str = "table",
) -> str:
    rows = _rows(coverage)
    summary = _summary(coverage, run_id, battery, model)

    if fmt == "json":
        return json.dumps({"summary": summary, "warnings": warnings, "tools": rows}, indent=2)

    if fmt == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
        return f"# {summary}\n" + buf.getvalue()

    # table / md: group by tier
    lines: list[str] = [summary, ""]
    if warnings:
        lines.append(f"parser warnings ({len(warnings)}):")
        lines.extend("  " + w for w in warnings)
        lines.append("")

    cols = [
        "tool",
        "class",
        "invocations",
        "total_calls",
        "arg_shapes",
        "facts_realized_ub",
        "enum_values_hit",
    ]
    header_note = "facts_realized_ub = required facts on questions that invoked the tool (UPPER BOUND; never summed)"

    for tier, desc in _TIERS:
        tier_rows = [r for r in rows if r["tier"] == tier]
        if not tier_rows:
            continue
        lines.append(f"## {tier} — {desc}" if fmt == "md" else f"[{tier}] {desc}")
        if fmt == "md":
            lines.append("| " + " | ".join(cols) + " |")
            lines.append("|" + "|".join("---" for _ in cols) + "|")
            for r in tier_rows:
                lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        else:
            widths = {c: max(len(c), *(len(str(r[c])) for r in tier_rows)) for c in cols}
            lines.append("  ".join(c.ljust(widths[c]) for c in cols))
            for r in tier_rows:
                lines.append("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
        lines.append("")

    lines.append(header_note)
    return "\n".join(lines)
