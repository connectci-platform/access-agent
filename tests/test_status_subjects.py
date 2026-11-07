"""Status lines name what the assistant is looking up, when it's safe to."""

import pytest

from src.agent.nodes.tool_calling_loop import _friendly_tool_status


@pytest.mark.parametrize(
    ("tool", "inputs", "expected"),
    [
        ("search_events", {"query": "GPU training"}, "Searching events for GPU training..."),
        ("search_resources", {"query": "Delta"}, "Searching resources for Delta..."),
        ("search_affinity_groups", {"query": "ACES"}, "Searching affinity groups for ACES..."),
        ("get_chart_data", {"realm": "SUPREMM"}, "Looking up SUPREMM usage data..."),
        ("get_infrastructure_news", {"resource": "Delta"}, "Checking system status for Delta..."),
    ],
)
def test_a_safe_subject_is_named(tool, inputs, expected):
    assert _friendly_tool_status(tool, inputs) == expected


@pytest.mark.parametrize(
    ("tool", "inputs", "expected"),
    [
        ("search_events", {}, "Searching events..."),
        (
            "search_events",
            {"query": "what GPU training is coming up in the next month"},
            "Searching events...",
        ),
        ("search_resources", {"query": "a@example.com"}, "Searching resources..."),
        ("get_chart_data", {"realm": 5}, "Looking up chart data..."),
    ],
)
def test_an_unsafe_or_missing_subject_stays_generic(tool, inputs, expected):
    assert _friendly_tool_status(tool, inputs) == expected
