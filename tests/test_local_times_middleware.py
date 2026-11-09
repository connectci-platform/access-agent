"""Tool results give event times in the user's own time zone, worked out in code."""

import json
from typing import Any, ClassVar

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from src.agent.middleware.local_times import LocalTimesMiddleware
from src.agent.middleware.preview_first import _is_preview

NY = "America/New_York"


class _Request:
    tool_call: ClassVar[dict[str, Any]] = {
        "name": "search_events",
        "args": {},
        "id": "c1",
        "type": "tool_call",
    }


async def _through(result: Any, time_zone: str | None = NY) -> Any:
    async def handler(request: Any) -> Any:
        return result

    return await LocalTimesMiddleware(time_zone).awrap_tool_call(_Request(), handler)


def _message(data: Any) -> ToolMessage:
    return ToolMessage(content=json.dumps(data), tool_call_id="c1", name="search_events")


async def _rewritten(data: Any, time_zone: str | None = NY) -> Any:
    result = await _through(_message(data), time_zone)
    return json.loads(result.content)


async def test_search_results_get_local_times_beside_the_utc_ones():
    event = {
        "id": 1,
        "title": "Intro to HPC",
        "start_date": "2026-11-20T18:00:00Z",
        "end_date": "2026-11-20T20:00:00Z",
        "location": "Online",
    }

    data = await _rewritten({"total": 1, "items": [event]})

    assert data["items"][0] == {
        "id": 1,
        "title": "Intro to HPC",
        "start_date": "2026-11-20T18:00:00Z",
        "start_local": "Fri, Nov 20, 2026, 1:00 PM EST",
        "end_date": "2026-11-20T20:00:00Z",
        "end_local": "Fri, Nov 20, 2026, 3:00 PM EST",
        "location": "Online",
    }
    assert list(data["items"][0]) == [
        "id",
        "title",
        "start_date",
        "start_local",
        "end_date",
        "end_local",
        "location",
    ]


async def test_a_nested_register_result_gets_local_times_and_still_reads_as_a_preview():
    result = await _through(
        _message(
            {
                "action": "register",
                "status": "preview",
                "executed": False,
                "event": {"title": "Office hours", "start_date": "2026-11-20T18:00:00+00:00"},
            }
        )
    )

    data = json.loads(result.content)
    assert data["event"]["start_local"] == "Fri, Nov 20, 2026, 1:00 PM EST"
    assert data["event"]["start_date"] == "2026-11-20T18:00:00+00:00"
    assert "end_local" not in data["event"]
    assert _is_preview(result)


async def test_a_summer_date_is_given_in_daylight_time():
    data = await _rewritten({"start_date": "2026-07-15T18:00:00Z"})

    assert data["start_local"] == "Wed, Jul 15, 2026, 2:00 PM EDT"


async def test_with_no_time_zone_times_are_labelled_utc():
    data = await _rewritten({"start_date": "2026-11-20T18:00:00Z"}, time_zone=None)

    assert data["start_local"] == "Fri, Nov 20, 2026, 6:00 PM UTC"


async def test_a_naive_time_is_read_as_utc():
    data = await _rewritten({"start_date": "2026-11-20T18:30:00"})

    assert data["start_local"] == "Fri, Nov 20, 2026, 1:30 PM EST"


async def test_bare_dates_and_unparseable_values_are_left_alone():
    original = {
        "items": [
            {"start_date": "2026-11-20", "end_date": "2026-11-21"},
            {"start_date": "soon", "end_date": None},
            {"start_date": "2026-13-45T99:00:00Z"},
        ]
    }

    assert await _rewritten(original) == original


async def test_an_unchanged_result_is_the_same_message():
    message = _message({"items": [], "total": 0})

    assert await _through(message) is message


async def test_non_json_results_pass_through():
    message = ToolMessage(content="Sign in to see this.", tool_call_id="c1")

    assert await _through(message) is message


async def test_command_results_pass_through():
    command = Command(update={"messages": []})

    assert await _through(command) is command


async def test_an_unknown_time_zone_falls_back_to_utc():
    data = await _rewritten({"start_date": "2026-11-20T18:00:00Z"}, time_zone="Mars/Olympus")

    assert data["start_local"] == "Fri, Nov 20, 2026, 6:00 PM UTC"


async def test_content_blocks_pass_through():
    message = ToolMessage(content=[{"type": "text", "text": "hi"}], tool_call_id="c1")

    assert await _through(message) is message


async def test_a_json_value_that_is_not_an_object_or_list_passes_through():
    message = ToolMessage(content="42", tool_call_id="c1")

    assert await _through(message) is message


async def test_an_artifact_survives_the_rewrite():
    message = ToolMessage(
        content=json.dumps({"start_date": "2026-11-20T18:00:00Z"}),
        tool_call_id="c1",
        name="search_events",
        artifact={"structured_content": {"chart": {"a": 1}}},
    )

    result = await _through(message)

    assert "start_local" in result.content
    assert result.artifact == {"structured_content": {"chart": {"a": 1}}}
