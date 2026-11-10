"""The synthetic per-tool health probe's declarative tool -> server -> args table.

Args are grounded in the eval batteries' real provenance (capability_review_battery.yaml,
gapfill_battery.yaml, multiturn_support_battery.yaml), not invented. Some originally came
from tool_coverage_battery.yaml, retired 2026-09-18 — capability_review carries its kept
questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProbeCase:
    server: str
    tool_name: str
    args: dict[str, Any]


PROBE_TABLE: list[ProbeCase] = [
    ProbeCase("system-status", "get_infrastructure_news", {"time": "current"}),
    ProbeCase("software-discovery", "search_software", {"query": "pytorch"}),
    ProbeCase(
        "software-discovery",
        "get_software_details",
        {"software_name": "cuda", "resource": "delta"},
    ),
    ProbeCase(
        "allocations",
        "search_projects",
        {"query": "climate modeling", "resource_name": "Delta"},
    ),
    ProbeCase("nsf-awards", "search_nsf_awards", {"query": "cyberinfrastructure"}),
    ProbeCase("events", "search_events", {"date": "upcoming", "limit": 20}),
    ProbeCase("affinity-groups", "search_affinity_groups", {"query": "GPU", "limit": 20}),
    ProbeCase("announcements", "search_announcements", {"query": "Expanse", "limit": 10}),
    ProbeCase("compute-resources", "search_resources", {"has_gpu": True}),
    # Queue-level hardware filters, read from the resource documentation (D8-2977).
    ProbeCase(
        "compute-resources",
        "search_resources",
        {"gpu_type": "H100", "min_gpu_memory_gb": 80},
    ),
    # "How do I log in to Expanse using SSH?" (eval/README.md). The bare name lists
    # Expanse's systems to choose from, so the probe names the CPU system.
    ProbeCase("compute-resources", "get_resource", {"resource": "Expanse CPU"}),
    # comb-007 in combined_battery.json: "Compare the GPU hardware available on
    # Delta and Bridges-2".
    ProbeCase(
        "compute-resources",
        "compare_resources",
        {"resources": ["Delta GPU", "Bridges-2 GPU"]},
    ),
    ProbeCase(
        "software-discovery",
        "compare_software_availability",
        {"software_names": ["lammps"], "resources": ["stampede3", "anvil"]},
    ),
    ProbeCase("allocations", "get_allocation_statistics", {"pages_to_analyze": 5}),
    # get_chart_data is schema-dependent: a valid call requires realm/dimension/statistic
    # values from a prior describe_realms -> describe_fields flow, so it has no single
    # deterministic known-good arg set. describe_realms (no args) is used instead as the
    # xdmod-server health check.
    ProbeCase("xdmod", "describe_realms", {}),
]
