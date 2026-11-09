"""Prompt guidance for usage charts and XDMoD data."""

from src.agent.prompts.system_prompt import build_system_prompt


def _prompts():
    return (
        build_system_prompt(acting_user=None),
        build_system_prompt(acting_user="jsmith@access-ci.org"),
    )


def test_chart_answers_point_to_the_chart():
    for prompt in _prompts():
        assert "point to the chart" in prompt


def test_xdmod_is_not_live_queue_status():
    for prompt in _prompts():
        assert "not live queue status" in prompt
        assert "get_infrastructure_news" in prompt


def test_fields_of_science_mapping():
    for prompt in _prompts():
        assert "fields of science" in prompt


def test_xdmod_links_only_as_a_tool_returned_them():
    for prompt in _prompts():
        assert "exactly as a tool returned it" in prompt


def test_usage_by_group_routes_to_chart_data_not_xdmod_data():
    for prompt in _prompts():
        assert "goes through `get_chart_data`" in prompt
        assert "only for what `get_chart_data` can't" in prompt
