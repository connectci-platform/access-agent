"""Human-written prose for the HTML report.

Everything in this file is editorial — it describes what a battery is for,
what the reader should take away, and so on. The numbers and per-question
data come from Postgres; the prose comes from here.

Prose is organized by **preset**. A preset bundles the subtitle, per-battery
descriptions, and observation bullets appropriate to one kind of comparison.
Use the `--preset` CLI flag to pick the right one; fall back to `grand-prix`
for historical compatibility.

Edit freely — the report runner just interpolates these values into the
template. See README.md in this directory for the full list of knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict


class BatteryInfo(TypedDict):
    """Metadata + prose for one question battery."""

    name: str  # Display name in section headers
    count: int  # Informational — the runner also reports the observed count
    what: str  # One-sentence description of what's in the battery
    why: str  # One-sentence rationale for why the battery exists


@dataclass(frozen=True)
class Preset:
    """Bundle of narrative prose for one kind of comparison report."""

    report_subtitle: str
    battery_info: dict[str, BatteryInfo]
    observations: list[dict[str, str]] = field(default_factory=list)


# Order batteries appear in the report. Preset-independent — presets share
# the same battery identity, they just describe them differently.
BATTERY_ORDER: list[str] = [
    "friendly_battery",
    "real_user_battery",
    "combined_battery",
    "mcp_coverage_battery",
]


# System ID → display label. Used by the renderer to translate the eval_runs
# system identifier (e.g. "raw_rag", "agent_full") into human-facing text in
# report copy, pills, and narrative substitution ("System A"/"System B" →
# label). Unknown IDs fall back to the raw ID. Preset-independent.
SYSTEM_LABELS: dict[str, str] = {
    "raw_rag": "Raw RAG",
    "agent_full": "Agent",
}


def label_for_system(system_id: str | None) -> str:
    if not system_id:
        return "(unknown)"
    return SYSTEM_LABELS.get(system_id, system_id)


# --- Preset registry ---------------------------------------------------------
#
# Each preset carries the full editorial surface: subtitle, per-battery
# descriptions, and observation bullets. To add a new preset, append an entry
# to PRESETS and expose it via the CLI's --preset choices.

PRESETS: dict[str, Preset] = {
    "grand-prix": Preset(
        report_subtitle="Raw UKY RAG vs full ACCESS agent",
        battery_info={
            "friendly_battery": {
                "name": "Friendly Battery",
                "count": 50,
                "what": (
                    "Clean, well-documented questions the system should nail: resource specs, "
                    '"how do I…" procedures, single-topic lookups.'
                ),
                "why": (
                    "Baseline competence — if either system stumbles here, nothing else matters."
                ),
            },
            "real_user_battery": {
                "name": "Real User Battery",
                "count": 50,
                "what": (
                    "Actual queries pulled from real users: messy phrasing, typos, terse "
                    "fragments, multi-part asks."
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
                "why": (
                    "This is where the agent should pull ahead of raw RAG — synthesis across "
                    "live sources."
                ),
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
        },
        # Observations intentionally empty — the report presents evidence
        # (answers, traces, required-facts checks) and lets the reader draw
        # conclusions. No editorialized summary bullets.
        observations=[],
    ),
}


def get_preset(name: str) -> Preset:
    """Look up a narrative preset by name.

    Raises ValueError with the list of valid choices if the name isn't known.
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choices: {sorted(PRESETS)}")
    return PRESETS[name]


# --- Backward-compatible module-level aliases --------------------------------
#
# Pre-existing call sites imported REPORT_SUBTITLE / BATTERY_INFO / OBSERVATIONS
# directly. Keep those names alive and pointed at the default (`grand-prix`)
# preset so nothing downstream of this file needs to change unless it wants to
# take advantage of the new --preset flag.

REPORT_SUBTITLE: str = PRESETS["grand-prix"].report_subtitle
BATTERY_INFO: dict[str, BatteryInfo] = PRESETS["grand-prix"].battery_info
OBSERVATIONS: list[dict[str, str]] = PRESETS["grand-prix"].observations
