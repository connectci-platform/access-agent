"""The synthetic per-tool health probe's declarative tool -> server -> args table.

Args are grounded in the eval batteries' real provenance (tool_coverage_battery.yaml,
capability_review_battery.yaml, gapfill_battery.yaml, multiturn_support_battery.yaml),
not invented.
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
    ProbeCase("compute-resources", "get_resource_hardware", {"id": "Anvil"}),
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
