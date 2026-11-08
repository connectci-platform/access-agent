"""Hand-written areas for the full-screen "What can you do?" answer.

The capability registry is per-tool and includes staff tools, which reads as a
long, mixed list. These areas group what a researcher can do in a few plain
lines. Each registry capability is tagged with one area, or None for staff
tools and anything that doesn't belong in the overview. An area shows only if
one of its capabilities is enabled, and an example only if the capability it
exercises is enabled, so DISABLED_CAPABILITIES still removes things here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AreaExample:
    """A suggested question, shown only while ``capability`` is enabled."""

    text: str
    capability: str


@dataclass(frozen=True)
class CapabilityArea:
    id: str
    title: str
    description: str
    examples: tuple[AreaExample, ...] = ()


CAPABILITY_AREAS: tuple[CapabilityArea, ...] = (
    CapabilityArea(
        "find_resource",
        "Find the right resource",
        "Compare systems by hardware, GPUs, storage and software, "
        "and see which ones fit your work.",
        (
            AreaExample("Which ACCESS resources have GPUs?", "browse_resources"),
            AreaExample("Where can I run GROMACS?", "search_software"),
        ),
    ),
    CapabilityArea(
        "get_started",
        "Get started and get unstuck",
        "Answers from ACCESS and resource provider documentation: "
        "logging in, moving data, running jobs.",
        (AreaExample("How do I transfer large files to Bridges-2?", "ask_question"),),
    ),
    CapabilityArea(
        "allocations",
        "Allocations",
        "How to get an allocation, what kinds exist, and, once you sign in, "
        "your own balances and usage.",
        (AreaExample("What allocations are available for new researchers?", "check_allocations"),),
    ),
    CapabilityArea(
        "training",
        "Training, events and community",
        "Upcoming workshops and office hours (I can register you), plus affinity groups to join.",
        (AreaExample("Sign me up for an upcoming GPU training", "browse_events"),),
    ),
    CapabilityArea(
        "system_status",
        "System status",
        "Current outages and scheduled maintenance.",
        (AreaExample("Are there any outages right now?", "check_system_status"),),
    ),
    CapabilityArea(
        "talk_to_person",
        "Talk to a person",
        "If I can't solve it, I can open a support ticket for you.",
    ),
)

# Every registry capability, by id, to its area. None keeps it out of the
# overview: staff tools (event creation, announcements, raw XDMoD extraction,
# security reports) and NSF award search, which no area covers. A test fails
# when a capability is added without a tag here.
AREA_BY_CAPABILITY: dict[str, str | None] = {
    "ask_question": "get_started",
    "ask_about_resource": "get_started",
    "browse_resources": "find_resource",
    "search_software": "find_resource",
    "check_allocations": "allocations",
    "check_usage": "allocations",
    "ask_xdmod_question": "allocations",
    "browse_events": "training",
    "browse_affinity_groups": "training",
    "check_system_status": "system_status",
    "open_ticket": "talk_to_person",
    "report_login_problem": "talk_to_person",
    "manage_events": None,
    "search_announcements": None,
    "manage_announcements": None,
    "extract_xdmod_data": None,
    "report_security": None,
    "search_nsf_awards": None,
}
