"""Tool calls whose object or array arguments arrive as JSON text.

Qwen sometimes sends ``filters`` as the text ``'{"resource": "ACES"}'``. These
run through a real create_agent graph so they cover the validation and error
handling production uses.
"""

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from src.agent.domains.tools import MCPToolWrapper, _build_args_schema
from src.agent.nodes.tool_calling_loop import _build_tool_results
from src.tools.mcp_client import MCPClient, MCPToolResult

PARAMETERS = [
    {"name": "realm", "type": "string", "required": True},
    {"name": "filters", "type": "object"},
    {"name": "fields", "type": "array"},
]


class _ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> "_ScriptedModel":
        return self


def _wrapper(client: MCPClient) -> MCPToolWrapper:
    return MCPToolWrapper(
        name="get_chart_data",
        description="chart data",
        args_schema=_build_args_schema("get_chart_data", PARAMETERS),
        tool_server="xdmod",
        mcp_client=client,
    )


async def _run_call(monkeypatch, args: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Run one scripted tool call; return what reached the server and the messages."""
    client = MCPClient()
    sent: list[dict[str, Any]] = []

    async def fake_call_tool(**kwargs):
        sent.append(kwargs["arguments"])
        return MCPToolResult(success=True, data={"ok": 1}, duration_ms=1)

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    call = {"name": "get_chart_data", "args": args, "id": "c1", "type": "tool_call"}
    model = _ScriptedModel(
        messages=iter([AIMessage(content="", tool_calls=[call]), AIMessage(content="done")])
    )
    agent = create_agent(model=model, tools=[_wrapper(client)])
    result = await agent.ainvoke({"messages": [HumanMessage(content="usage on ACES?")]})
    return sent, result["messages"]


async def test_object_sent_as_text_reaches_the_server_as_an_object(monkeypatch):
    sent, _ = await _run_call(monkeypatch, {"realm": "Jobs", "filters": '{"resource": "ACES"}'})

    assert sent == [{"realm": "Jobs", "filters": {"resource": "ACES"}}]


async def test_array_sent_as_text_reaches_the_server_as_an_array(monkeypatch):
    sent, _ = await _run_call(monkeypatch, {"realm": "Jobs", "fields": '["pi", "fos"]'})

    assert sent == [{"realm": "Jobs", "fields": ["pi", "fos"]}]


async def test_real_objects_are_unchanged(monkeypatch):
    sent, _ = await _run_call(monkeypatch, {"realm": "Jobs", "filters": {"resource": "ACES"}})

    assert sent == [{"realm": "Jobs", "filters": {"resource": "ACES"}}]


async def test_rejected_call_counts_as_a_failed_tool_result(monkeypatch):
    sent, messages = await _run_call(monkeypatch, {"realm": "Jobs", "filters": "not json"})

    results, _ = _build_tool_results(messages, [], [])

    assert sent == []
    assert [r.success for r in results] == [False]
    assert "valid dictionary" in (results[0].error or "")


async def test_successful_call_stays_a_successful_tool_result(monkeypatch):
    _, messages = await _run_call(monkeypatch, {"realm": "Jobs"})

    results, _ = _build_tool_results(messages, [], [])

    assert [r.success for r in results] == [True]
    assert results[0].error is None


def test_error_status_with_json_content_keeps_the_content_as_the_error():
    call = {"name": "t", "args": {}, "id": "c1", "type": "tool_call"}
    messages = [
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content='{"ok": 1}', tool_call_id="c1", status="error"),
    ]

    results, _ = _build_tool_results(messages, [], [])

    assert results[0].success is False
    assert results[0].error == '{"ok": 1}'


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filters", "not json"),
        ("filters", '["a list, not an object"]'),
        ("filters", "null"),
        ("fields", '{"an": "object, not a list"}'),
    ],
)
def test_text_that_is_not_the_right_shape_is_still_rejected(field, value):
    schema = _build_args_schema("get_chart_data", PARAMETERS)

    with pytest.raises(ValidationError):
        schema(realm="Jobs", **{field: value})


def test_required_object_accepts_text():
    schema = _build_args_schema("t", [{"name": "filters", "type": "object", "required": True}])

    assert schema(filters='{"a": 1}').filters == {"a": 1}


def test_string_parameters_keep_json_looking_text():
    schema = _build_args_schema("t", [{"name": "query", "type": "string", "required": True}])

    assert schema(query='{"a": 1}').query == '{"a": 1}'


def test_the_schema_shown_to_the_model_still_asks_for_an_object():
    properties = _build_args_schema("t", PARAMETERS).model_json_schema()["properties"]

    assert {"type": "object", "additionalProperties": True} in properties["filters"]["anyOf"]
