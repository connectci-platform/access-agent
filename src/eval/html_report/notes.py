"""Human-written prose for the HTML report.

Everything in this file is editorial — it describes what a battery is for,
what the reader should take away, and so on. The numbers and per-question
data come from Postgres; the prose comes from here.

Edit freely — the report runner just interpolates these values into the
template. See README.md in this directory for the full list of knobs.
"""

from __future__ import annotations

from typing import TypedDict


class BatteryInfo(TypedDict):
    """Metadata + prose for one question battery."""

    name: str  # Display name in section headers
    count: int  # Informational — the runner also reports the observed count
    what: str  # One-sentence description of what's in the battery
    why: str  # One-sentence rationale for why the battery exists


# Keyed by the question_set basename *without* .json extension.
# The runner tolerates keys with or without the extension.
BATTERY_INFO: dict[str, BatteryInfo] = {
    "friendly_battery": {
        "name": "Friendly Battery",
        "count": 50,
        "what": (
            "Clean, well-documented questions the system should nail: resource specs, "
            '"how do I…" procedures, single-topic lookups.'
        ),
        "why": "Baseline competence — if either system stumbles here, nothing else matters.",
    },
    "real_user_battery": {
        "name": "Real User Battery",
        "count": 50,
        "what": (
            "Actual queries pulled from real users: messy phrasing, typos, terse fragments, "
            "multi-part asks."
        ),
        "why": "Tests robustness to how humans actually write when they want help.",
    },
    "combined_battery": {
        "name": "Combined Battery",
        "count": 30,
        "what": (
            "Questions designed to require multiple tools or data sources in one answer "
            "(resource + software + events, NSF awards + affinity groups, etc.)."
        ),
        "why": "This is where the agent should pull ahead of raw RAG — synthesis across live sources.",
    },
    "mcp_coverage_battery": {
        "name": "MCP Coverage Battery",
        "count": 21,
        "what": (
            "One question per MCP tool, crafted to trigger that tool and no other. "
            "Verifies every tool works end-to-end."
        ),
        "why": (
            "Smoke test for the tool surface — also the one place raw RAG is at a "
            "structural disadvantage (no live data)."
        ),
    },
}

# Order batteries appear in the report
BATTERY_ORDER: list[str] = [
    "friendly_battery",
    "real_user_battery",
    "combined_battery",
    "mcp_coverage_battery",
]


# Observation bullets shown at the bottom of the report. Each entry is a
# dict with "text" (HTML allowed for <strong>, <code>, <em>). Rewrite or
# reorder freely between eval cycles.
OBSERVATIONS: list[dict[str, str]] = [
    {
        "text": (
            "<strong>Agent matches or beats raw RAG on every battery.</strong> "
            "Most divergences are small; the ties are dominated by questions where both "
            "systems produce an acceptable paraphrase of the same underlying documentation."
        ),
    },
    {
        "text": (
            "<strong>Agent's largest margin is on the Combined battery,</strong> "
            "where questions require synthesizing multiple live data sources in a single "
            "answer — the shape raw RAG structurally can't handle."
        ),
    },
    {
        "text": (
            "<strong>One known regression: <code>mcp-cov-010</code> (webinar question).</strong> "
            "The agent correctly routed to the events MCP tool but mishandled an empty result — "
            'it asserted "no upcoming webinars" instead of falling back to static documentation. '
            "Fix is on a branch (<code>fix/synthesis-empty-tool-defers-to-rag</code>)."
        ),
    },
    {
        "text": (
            "<strong>Agent is slower</strong> because it runs plan → tool-selection → "
            "tool-execution → synth per query. The gap widens on multi-tool questions. "
            "Target is sub-5s for single-tool queries."
        ),
    },
    {
        "text": (
            "<strong>Verdicts come from a pairwise comparison judge,</strong> not from "
            "individual per-answer scores. The comparison judge sees both answers side by side "
            "and picks a winner with a margin and a why — that's the narrative surfaced in "
            "this report. Next calibration step: push disputed verdicts to Argilla for human "
            "review."
        ),
    },
]

# System ID → display label. Used by the renderer to translate the eval_runs
# system identifier (e.g. "raw_rag", "agent_full") into human-facing text in
# report copy, pills, and narrative substitution ("System A"/"System B" →
# label). Unknown IDs fall back to the raw ID.
SYSTEM_LABELS: dict[str, str] = {
    "raw_rag": "Raw RAG",
    "agent_full": "Agent",
}


def label_for_system(system_id: str | None) -> str:
    if not system_id:
        return "(unknown)"
    return SYSTEM_LABELS.get(system_id, system_id)


# Subtitle shown under the main title
REPORT_SUBTITLE: str = (
    "Raw UKY RAG vs. full ACCESS agent — pairwise verdicts from the "
    "comparison judge across four question batteries."
)
