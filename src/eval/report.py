"""Print eval run reports to the terminal."""

import json
from datetime import UTC, datetime
from typing import Any

from .rubric import DIMENSION_MAX, DIMENSION_NAMES


def print_run_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 60)
    print(f"  Eval Run: {summary['run_id']}")
    print(f"  Branch:   {summary.get('agent_branch', 'unknown')}")
    print(f"  Commit:   {summary.get('agent_commit', 'unknown')}")
    print("=" * 60)
    print()
    print(f"  Questions: {summary['questions']}")
    print(f"  Scored:    {summary['scored']}")
    print(f"  Skipped:   {summary['skipped']}")
    print()
    print(f"  Composite Score: {summary['composite_score']:.2f} / 1.00")
    print()
    dims = summary.get("per_dimension", {})
    if dims:
        print("  Per Dimension:")
        # Bar is a fixed display width filled proportionally to score / the
        # dimension's own max (v2 dimensions max at 2, hedging at 1), so a
        # perfect score renders full regardless of the per-dimension scale.
        bar_width = 5
        for name in DIMENSION_NAMES:
            score = dims.get(name, 0.0)
            filled = (
                round(bar_width * score / DIMENSION_MAX[name]) if DIMENSION_MAX.get(name) else 0
            )
            filled = max(0, min(bar_width, filled))
            bar = "█" * filled + "░" * (bar_width - filled)
            print(f"    {name:20s} {score:.2f}  {bar}")
    print()
    print("=" * 60)


COMPOSITE_INCOMMENSURABLE_NOTE = "(macro, Fair-only — not comparable to micro)"


def print_comparison(
    run_a: dict[str, Any], run_b: dict[str, Any], *, composite_commensurable: bool = True
) -> None:
    """Print a two-run comparison table.

    When ``composite_commensurable`` is False the two runs' composites are built
    under different semantics (multiturn macro over Fair-only thread composites
    vs. single-turn micro average). The composite delta is then meaningless, so
    the row is annotated rather than given a number — the per-dimension rows,
    which ARE commensurable, carry the comparison.
    """
    print()
    print("=" * 70)
    print("  Eval Run Comparison")
    print("=" * 70)
    print(f"  {'':20s} {'Run A':>10s}  {'Run B':>10s}  {'Delta':>10s}")
    print(f"  {'':20s} {'─' * 10:>10s}  {'─' * 10:>10s}  {'─' * 10:>10s}")
    a_dims = run_a.get("per_dimension", {})
    b_dims = run_b.get("per_dimension", {})
    for name in DIMENSION_NAMES:
        a_val = a_dims.get(name, 0.0)
        b_val = b_dims.get(name, 0.0)
        delta = b_val - a_val
        sign = "+" if delta > 0 else ""
        print(f"  {name:20s} {a_val:10.2f}  {b_val:10.2f}  {sign}{delta:9.2f}")
    a_comp = run_a.get("composite_score", 0.0)
    b_comp = run_b.get("composite_score", 0.0)
    print(f"  {'─' * 50}")
    if composite_commensurable:
        delta = b_comp - a_comp
        sign = "+" if delta > 0 else ""
        print(f"  {'COMPOSITE':20s} {a_comp:10.2f}  {b_comp:10.2f}  {sign}{delta:9.2f}")
    else:
        print(f"  {'COMPOSITE':20s} {a_comp:10.2f}  {b_comp:10.2f}  {'n/a':>10s}")
        print(f"  {'':20s} {COMPOSITE_INCOMMENSURABLE_NOTE}")
    print()
    print(f"  Run A: {run_a.get('run_id', '?')} ({run_a.get('agent_branch', '?')})")
    print(f"  Run B: {run_b.get('run_id', '?')} ({run_b.get('agent_branch', '?')})")
    print("=" * 70)


def generate_team_report(data: dict[str, Any]) -> str:
    """Generate weekly team report as markdown."""
    lines = [
        f"# Agent Quality Report — {data['period']}",
        "",
        f"**Composite Score: {data['composite_score']:.2f} / 1.00**",
        f"Answers scored: {data['total_scored']} | Human reviewed: {data['human_coverage']:.0%}",
    ]
    if data.get("judge_human_agreement") is not None:
        lines.append(f"Judge-human agreement: {data['judge_human_agreement']:.0%}")
    dims = data.get("per_dimension", {})
    if dims:
        lines.extend(["", "## Per Dimension", "", "| Dimension | Score |", "|-----------|-------|"])
        for name in DIMENSION_NAMES:
            score = dims.get(name, 0.0)
            lines.append(f"| {name} | {score:.2f} |")
    worst = data.get("worst_answers", [])
    if worst:
        lines.extend(["", "## Lowest Scoring Answers", ""])
        for w in worst[:10]:
            lines.append(f"- **{w['composite']:.1f}** — {w['question']}")
    gaps = data.get("capability_gaps", [])
    if gaps:
        lines.extend(["", "## Capability Gaps", ""])
        for g in gaps:
            lines.append(f"- **{g['area']}**: avg {g['avg_score']:.2f} ({g['count']} questions)")
    if not dims and not worst:
        lines.append("\nNo scores available for this period.")
    return "\n".join(lines)


def generate_leadership_report(data: dict[str, Any]) -> str:
    """Generate monthly leadership summary as markdown."""
    lines = [
        f"# Agent Quality Summary — {data['period']}",
        "",
        f"**Composite Score: {data['composite_score']:.2f} / 1.00**",
    ]
    prev = data.get("previous_composite")
    if prev is not None:
        delta = data["composite_score"] - prev
        sign = "+" if delta > 0 else ""
        trend = "improved" if delta > 0 else "declined" if delta < 0 else "unchanged"
        lines.append(f"Trend: {sign}{delta:.2f} ({trend} from {prev:.2f})")
    lines.extend(
        [
            "",
            f"- Queries handled: {data.get('total_queries', 0):,}",
            f"- Answers scored: {data.get('total_scored', 0):,}",
            f"- Human reviewed: {data.get('human_reviewed', 0):,}",
        ]
    )
    breakdown = data.get("capability_breakdown", [])
    if breakdown:
        lines.extend(
            [
                "",
                "## Quality by Capability Area",
                "",
                "| Area | Score | Questions |",
                "|------|-------|-----------|",
            ]
        )
        for b in sorted(breakdown, key=lambda x: x["score"]):
            lines.append(f"| {b['area']} | {b['score']:.2f} | {b['count']} |")
    return "\n".join(lines)


def generate_resource_report(data: dict[str, Any]) -> str:
    """Generate per-resource report as markdown."""
    resource = data.get("resource", "Unknown")
    lines = [
        f"# {resource} — Agent Answer Quality",
        "",
        f"**Period:** {data['period']}",
        f"**Composite Score: {data['composite_score']:.2f} / 1.00**",
        f"**Answers scored:** {data['total_scored']}",
    ]
    dims = data.get("per_dimension", {})
    if dims:
        lines.extend(
            ["", "## Dimension Scores", "", "| Dimension | Score |", "|-----------|-------|"]
        )
        for name in DIMENSION_NAMES:
            score = dims.get(name, 0.0)
            lines.append(f"| {name} | {score:.2f} |")
    worst = data.get("worst_answers", [])
    if worst:
        lines.extend(["", "## Lowest Scoring Answers", ""])
        for w in worst[:5]:
            lines.append(f"- **{w['composite']:.1f}** — {w['question']}")
    return "\n".join(lines)


def generate_comparison_report(  # noqa: PLR0912, PLR0915
    run_pairs: list[dict[str, Any]],
    title: str = "Production Baseline Comparison",
) -> str:
    """Generate A/B comparison report as markdown.

    Args:
        run_pairs: List of dicts, each with keys:
            battery: str (e.g. "friendly_battery")
            baseline: dict with run_id, system, composite, scores_summary, scores (list)
            candidate: dict with same shape
        title: Report title.

    Returns:
        Markdown string.
    """
    now = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    lines = [
        f"# {title}",
        "",
        f"**Date:** {now}",
        "",
    ]

    # Collect all scores for overall stats
    all_baseline_composites = []
    all_candidate_composites = []

    for pair in run_pairs:
        for s in pair["baseline"]["scores"]:
            if s["composite_score"] is not None:
                all_baseline_composites.append(s["composite_score"])
        for s in pair["candidate"]["scores"]:
            if s["composite_score"] is not None:
                all_candidate_composites.append(s["composite_score"])

    baseline_avg = (
        sum(all_baseline_composites) / len(all_baseline_composites)
        if all_baseline_composites
        else 0
    )
    candidate_avg = (
        sum(all_candidate_composites) / len(all_candidate_composites)
        if all_candidate_composites
        else 0
    )
    delta = candidate_avg - baseline_avg
    sign = "+" if delta > 0 else ""
    total_q = len(all_baseline_composites)

    b_sys = run_pairs[0]["baseline"]["system"] if run_pairs else "baseline"
    c_sys = run_pairs[0]["candidate"]["system"] if run_pairs else "candidate"

    lines.extend(
        [
            "## Overall",
            "",
            f"| Metric | {b_sys} | {c_sys} | Delta |",
            "|--------|---------|---------|-------|",
            f"| **Composite** | **{baseline_avg:.2f}** | **{candidate_avg:.2f}** | **{sign}{delta:.2f}** |",
            f"| Questions scored | {total_q} | {total_q} | |",
            "",
        ]
    )

    # Per-battery summary
    lines.extend(
        [
            "## Per Battery",
            "",
            f"| Battery | {b_sys} | {c_sys} | Delta | Questions |",
            "|---------|---------|---------|-------|-----------|",
        ]
    )
    for pair in run_pairs:
        b = pair["baseline"]
        c = pair["candidate"]
        d = (c["composite"] or 0) - (b["composite"] or 0)
        s = "+" if d > 0 else ""
        battery_name = pair["battery"].replace("_battery", "").replace("_", " ")
        lines.append(
            f"| {battery_name} | {b['composite']:.2f} | {c['composite']:.2f} "
            f"| {s}{d:.2f} | {len(b['scores'])} |"
        )
    lines.append("")

    # Per-dimension comparison (aggregated)
    lines.extend(
        [
            "## Per Dimension",
            "",
            f"| Dimension | {b_sys} | {c_sys} | Delta |",
            "|-----------|---------|---------|-------|",
        ]
    )
    for dim in DIMENSION_NAMES:
        b_vals = [
            s[dim]
            for pair in run_pairs
            for s in pair["baseline"]["scores"]
            if s.get(dim) is not None
        ]
        c_vals = [
            s[dim]
            for pair in run_pairs
            for s in pair["candidate"]["scores"]
            if s.get(dim) is not None
        ]
        b_avg = sum(b_vals) / len(b_vals) if b_vals else 0
        c_avg = sum(c_vals) / len(c_vals) if c_vals else 0
        d = c_avg - b_avg
        s = "+" if d > 0 else ""
        lines.append(f"| {dim} | {b_avg:.2f} | {c_avg:.2f} | {s}{d:.2f} |")
    lines.append("")

    # Per-battery detail with question-level results
    for pair in run_pairs:
        battery_name = pair["battery"].replace("_battery", "").replace("_", " ")
        lines.extend(
            [
                f"## {battery_name.title()} Battery — Question Detail",
                "",
            ]
        )

        # Build lookup by question_id
        b_by_q = {s["question_id"]: s for s in pair["baseline"]["scores"]}
        c_by_q = {s["question_id"]: s for s in pair["candidate"]["scores"]}
        all_qids = list(
            dict.fromkeys(
                [s["question_id"] for s in pair["candidate"]["scores"]]
                + [s["question_id"] for s in pair["baseline"]["scores"]]
            )
        )

        # Find questions where agent used tools
        tool_questions = []
        tie_questions = []
        agent_wins = []
        agent_losses = []

        for qid in all_qids:
            b_score = b_by_q.get(qid, {})
            c_score = c_by_q.get(qid, {})
            bc = b_score.get("composite_score", 0) or 0
            cc = c_score.get("composite_score", 0) or 0
            d = cc - bc
            tools = _extract_tools(c_score)

            entry = {
                "qid": qid,
                "question": (c_score.get("question_text") or b_score.get("question_text", ""))[
                    :100
                ],
                "baseline_comp": bc,
                "candidate_comp": cc,
                "delta": d,
                "tools": tools,
            }

            if tools:
                tool_questions.append(entry)
            if abs(d) < 0.1:
                tie_questions.append(entry)
            elif d > 0:
                agent_wins.append(entry)
            else:
                agent_losses.append(entry)

        # Tool usage summary
        if tool_questions:
            lines.extend(
                [
                    f"### Tool Usage ({len(tool_questions)}/{len(all_qids)} questions)",
                    "",
                    f"| Question | {b_sys} | {c_sys} | Delta | Tools |",
                    "|----------|---------|---------|-------|-------|",
                ]
            )
            for e in sorted(tool_questions, key=lambda x: -x["delta"]):
                s = "+" if e["delta"] > 0 else ""
                tools_str = ", ".join(e["tools"][:3])
                if len(e["tools"]) > 3:
                    tools_str += f" +{len(e['tools']) - 3}"
                lines.append(
                    f"| {e['qid']}: {e['question'][:60]} "
                    f"| {e['baseline_comp']:.1f} | {e['candidate_comp']:.1f} "
                    f"| {s}{e['delta']:.1f} | {tools_str} |"
                )
            lines.append("")

        # Agent wins
        if agent_wins:
            lines.extend(
                [
                    f"### Agent Wins ({len(agent_wins)} questions)",
                    "",
                    f"| Question | {b_sys} | {c_sys} | Delta |",
                    "|----------|---------|---------|-------|",
                ]
            )
            for e in sorted(agent_wins, key=lambda x: -x["delta"]):
                lines.append(
                    f"| {e['qid']}: {e['question'][:70]} "
                    f"| {e['baseline_comp']:.1f} | {e['candidate_comp']:.1f} "
                    f"| +{e['delta']:.1f} |"
                )
            lines.append("")

        # Agent losses
        if agent_losses:
            lines.extend(
                [
                    f"### Agent Losses ({len(agent_losses)} questions)",
                    "",
                    f"| Question | {b_sys} | {c_sys} | Delta |",
                    "|----------|---------|---------|-------|",
                ]
            )
            for e in sorted(agent_losses, key=lambda x: x["delta"]):
                lines.append(
                    f"| {e['qid']}: {e['question'][:70]} "
                    f"| {e['baseline_comp']:.1f} | {e['candidate_comp']:.1f} "
                    f"| {e['delta']:.1f} |"
                )
            lines.append("")

    return "\n".join(lines)


def _extract_tools(score: dict[str, Any]) -> list[str]:
    """Extract tool names from a score's node_trace context."""
    ctx = score.get("context") or {}
    trace_str = ctx.get("node_trace")
    if not trace_str:
        return []
    try:
        trace = json.loads(trace_str) if isinstance(trace_str, str) else trace_str
    except (json.JSONDecodeError, TypeError):
        return []
    tools = []
    for t in trace or []:
        if t.get("tools_called"):
            tools.extend(t["tools_called"])
    return tools
