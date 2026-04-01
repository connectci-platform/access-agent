"""Print eval run reports to the terminal."""

from typing import Any

from .rubric import DIMENSION_NAMES


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
    print(f"  Composite Score: {summary['composite_score']:.2f} / 5.00")
    print()
    dims = summary.get("per_dimension", {})
    if dims:
        print("  Per Dimension:")
        for name in DIMENSION_NAMES:
            score = dims.get(name, 0.0)
            bar = "█" * int(score) + "░" * (5 - int(score))
            print(f"    {name:20s} {score:.2f}  {bar}")
    print()
    print("=" * 60)


def print_comparison(run_a: dict[str, Any], run_b: dict[str, Any]) -> None:
    print()
    print("=" * 70)
    print("  Eval Run Comparison")
    print("=" * 70)
    print(f"  {'':20s} {'Run A':>10s}  {'Run B':>10s}  {'Delta':>10s}")
    print(f"  {'':20s} {'─'*10:>10s}  {'─'*10:>10s}  {'─'*10:>10s}")
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
    delta = b_comp - a_comp
    sign = "+" if delta > 0 else ""
    print(f"  {'─'*50}")
    print(f"  {'COMPOSITE':20s} {a_comp:10.2f}  {b_comp:10.2f}  {sign}{delta:9.2f}")
    print()
    print(f"  Run A: {run_a.get('run_id', '?')} ({run_a.get('agent_branch', '?')})")
    print(f"  Run B: {run_b.get('run_id', '?')} ({run_b.get('agent_branch', '?')})")
    print("=" * 70)


def generate_team_report(data: dict[str, Any]) -> str:
    """Generate weekly team report as markdown."""
    lines = [
        f"# Agent Quality Report — {data['period']}",
        "",
        f"**Composite Score: {data['composite_score']:.2f} / 5.00**",
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
        f"**Composite Score: {data['composite_score']:.2f} / 5.00**",
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
        f"**Composite Score: {data['composite_score']:.2f} / 5.00**",
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
