"""Per-tool evaluation-depth coverage audit.

Adding an MCP server is cheap, so new capabilities ship with no battery
coverage: the release gate stays green because nothing tests the new tool. This
module diffs the tools a run's catalog *served* against what its questions
*actually exercised*, and reports, per tool, how deeply it is evaluated.

Grounding (verified against the code, not assumed):
- Tools invoked per question live in ``context["node_trace"][*]["tools_called"]``.
  We reuse the exact traversal ``report._extract_tools`` already uses.
- Argument shapes survive only in the ``context["tool_results"]`` markdown
  string (``### Tool call: <name>`` blocks joined by ``\\n\\n---\\n\\n``, each
  with a ``- arguments: <json>`` line). ``tools_called`` dedups by name, so
  distinct call-shapes of one tool exist only in that string.
- The served-tool universe for a run is ``eval_runs.tool_catalog["tools"]``
  (names only; no per-tool params, so no argument-mode denominator here).
- There is no fact->tool link anywhere; facts attribute to a question, and a
  question's facts attribute to the tools it actually invoked ("realized").

What this does NOT do: re-run anything, add persistence, compute argument-mode
*ratios* (no denominator in the DB), or per-fact tool attribution.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.agent.domains.capabilities import (
    AUTH_READ_MCP_TOOL_NAMES,
    COMPOSITIONAL_MCP_TOOL_NAMES,
    WRITE_MCP_TOOL_NAMES,
)

# --- structural classification (name/registry-based; the DB carries no auth flag) ---

_AUTH_READ_RE = re.compile(r"^(get_my_|authenticate$|complete_authentication$)")
_WRITE_PREFIX_RE = re.compile(r"^(create_|update_|delete_|register_|cancel_|report_)")


def structural_class(tool: str) -> str:
    """Classify a tool as unauth-read / auth-read / write / composition.

    WRITE_MCP_TOOL_NAMES membership is the primary write signal; the name-prefix
    is a fallback so a newly-added write tool not yet in the set is still caught.
    Compositional tools (XDMoD plumbing, authoring helpers) are read-only but
    never user-facing, so they get their own class: a zero-invocation count for
    them is not an actionable battery gap, unlike a user-facing read tool.
    """
    if tool in WRITE_MCP_TOOL_NAMES or _WRITE_PREFIX_RE.match(tool):
        return "write"
    if tool in COMPOSITIONAL_MCP_TOOL_NAMES:
        return "composition"
    if tool in AUTH_READ_MCP_TOOL_NAMES or _AUTH_READ_RE.match(tool):
        return "auth-read"
    return "unauth-read"


# --- tool_results markdown parsing (the one unavoidable string parse) ---

_HEADER_RE = re.compile(r"^### Tool call:\s*(.+?)\s*$")
_ARGS_RE = re.compile(r"^- arguments:\s*(.*)$")


def parse_tool_result_calls(tool_results: str | None) -> list[tuple[str, dict[str, Any]]]:
    """Extract (tool_name, arguments) per call from the tool_results markdown.

    Blocks are joined by ``\\n\\n---\\n\\n`` but that delimiter (and the ``### ``
    header) can also appear inside a stringified ``- data:`` payload, so we do
    not trust a naive split: a block counts only when a ``### Tool call:`` header
    is immediately followed (within the block) by a ``- arguments:`` line.
    Unparseable arguments degrade to ``{}`` rather than dropping the call.
    """
    if not tool_results:
        return []
    calls: list[tuple[str, dict[str, Any]]] = []
    for block in tool_results.split("\n\n---\n\n"):
        lines = block.splitlines()
        name: str | None = None
        args: dict[str, Any] = {}
        for i, line in enumerate(lines):
            m = _HEADER_RE.match(line)
            if m:
                name = m.group(1)
                # arguments line must follow the header within this block
                for later in lines[i + 1 :]:
                    a = _ARGS_RE.match(later)
                    if a:
                        raw = a.group(1)
                        try:
                            parsed = json.loads(raw)
                            if isinstance(parsed, dict):
                                args = parsed
                        except (json.JSONDecodeError, TypeError):
                            args = {}
                        break
                break  # one header per block
        if name is not None:
            calls.append((name, args))
    return calls


def _extract_invoked_tools(node_trace: Any) -> list[str]:
    """Tools invoked in a question, from node_trace[*].tools_called.

    Mirrors ``report._extract_tools`` (kept local to avoid importing the report
    module's rendering deps). node_trace may be a JSON string or already a list.
    """
    if not node_trace:
        return []
    try:
        trace = json.loads(node_trace) if isinstance(node_trace, str) else node_trace
    except (json.JSONDecodeError, TypeError):
        return []
    tools: list[str] = []
    for node in trace or []:
        called = node.get("tools_called") if isinstance(node, dict) else None
        if called:
            tools.extend(called)
    return tools


# --- per-tool aggregation ---

_ENUM_LIKE_PARAMS = {"time", "source", "realm", "date", "when", "status", "type"}


@dataclass
class ToolCoverage:
    tool: str
    structural: str
    invocations: int = 0  # distinct questions that invoked it
    total_calls: int = 0  # total call blocks (>= invocations)
    arg_shapes: set[frozenset[str]] = field(default_factory=set)
    enum_values: dict[str, set[str]] = field(default_factory=dict)
    facts_realized: int = 0  # required_facts on questions that invoked it (UPPER BOUND)
    facts_passed: int = 0
    in_snapshot: bool = True  # was it in the run's served tool_catalog

    @property
    def distinct_arg_shapes(self) -> int:
        return len(self.arg_shapes)

    @property
    def tier(self) -> str:
        if self.invocations > 0 and self.facts_realized > 0:
            return "EVALUATED"
        if self.invocations > 0:
            return "EXERCISED-NOT-EVALUATED"
        if self.structural in ("write", "auth-read"):
            return "UNCOVERED-STRUCTURAL"
        if self.structural == "composition":
            return "UNCOVERED-COMPOSITIONAL"
        return "UNCOVERED-TESTABLE"


def build_coverage(
    served_tools: list[str],
    scores: list[dict[str, Any]],
) -> list[ToolCoverage]:
    """Aggregate per-tool coverage from a run's served catalog + judge scores.

    ``scores`` is a list of dicts each with ``context`` (node_trace, tool_results,
    required_facts, fact_verdicts). Question-level: a question that invoked a
    tool attributes ALL its facts to that tool (this over-counts facts_realized —
    it is an upper bound, never summed).
    """
    cov: dict[str, ToolCoverage] = {
        t: ToolCoverage(tool=t, structural=structural_class(t)) for t in served_tools
    }

    for score in scores:
        ctx = score.get("context") or {}
        invoked = set(_extract_invoked_tools(ctx.get("node_trace")))
        calls = parse_tool_result_calls(ctx.get("tool_results"))
        facts = ctx.get("required_facts") or []
        n_facts = len(facts) if isinstance(facts, list) else 0
        verdicts = ctx.get("fact_verdicts") or []
        n_pass = sum(
            1 for v in verdicts if isinstance(v, dict) and v.get("verdict") in ("yes", "pass", True)
        )

        # a tool invoked here but not in the served snapshot: track it, flagged
        for t in invoked:
            if t not in cov:
                cov[t] = ToolCoverage(tool=t, structural=structural_class(t), in_snapshot=False)

        for t in invoked:
            c = cov[t]
            c.invocations += 1
            c.facts_realized = max(c.facts_realized, n_facts)  # upper bound per tool
            c.facts_passed = max(c.facts_passed, n_pass)

        for name, args in calls:
            if name not in cov:
                cov[name] = ToolCoverage(
                    tool=name, structural=structural_class(name), in_snapshot=False
                )
            c = cov[name]
            c.total_calls += 1
            c.arg_shapes.add(frozenset(args.keys()))
            for k, v in args.items():
                if k in _ENUM_LIKE_PARAMS and isinstance(v, (str, int, bool)):
                    c.enum_values.setdefault(k, set()).add(str(v))

    return sorted(
        cov.values(),
        key=lambda c: (
            [
                "UNCOVERED-TESTABLE",
                "EXERCISED-NOT-EVALUATED",
                "EVALUATED",
                "UNCOVERED-COMPOSITIONAL",
                "UNCOVERED-STRUCTURAL",
            ].index(c.tier),
            c.tool,
        ),
    )


def reconcile_warnings(scores: list[dict[str, Any]]) -> list[str]:
    """Warn only when node_trace and tool_results disagree on the NAME SET.

    Counts legitimately differ (tools_called dedups; tool_results does not), so a
    count mismatch is not a bug. A name-set mismatch is real parser drift.
    """
    warnings: list[str] = []
    for i, score in enumerate(scores):
        ctx = score.get("context") or {}
        trace_names = set(_extract_invoked_tools(ctx.get("node_trace")))
        result_names = {n for n, _ in parse_tool_result_calls(ctx.get("tool_results"))}
        if trace_names and result_names and trace_names != result_names:
            only_trace = trace_names - result_names
            only_results = result_names - trace_names
            warnings.append(
                f"score[{i}] {score.get('question_id', '?')}: name-set drift "
                f"(only in node_trace: {sorted(only_trace)}; "
                f"only in tool_results: {sorted(only_results)})"
            )
    return warnings
