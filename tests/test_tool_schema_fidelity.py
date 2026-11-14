"""The schema the model sees for each MCP tool keeps what the server declared.

Fixture: a snapshot of production ``tools/list`` from 2026-11-12 (48 tools, raw
responses, descriptions cut to 60 characters). It must be refreshed by hand:
initialize a session on https://mcp.access-ci.org/<server>/mcp for each server
(no credential needed), call ``tools/list``, and keep every tool's fields.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from src.agent.domains.capabilities import WRITE_MCP_TOOL_NAMES
from src.agent.domains.tools import _build_args_schema, create_mcp_tools_from_catalog
from src.tools.registry import CatalogAggregator

RAW = json.loads((Path(__file__).parent / "fixtures" / "mcp_tools_prod.json").read_text())
RAW_TOOLS = {t["name"]: t for s in RAW["servers"] for t in s["tools"]}
PLAIN_KEYS = {"type", "description", "default"}


def _catalog() -> dict[str, Any]:
    aggregator = CatalogAggregator()
    return {
        "servers": [
            {
                "server": s["server"],
                "status": "available",
                "tools": [aggregator._process_tool(t) for t in s["tools"]],
            }
            for s in RAW["servers"]
        ]
    }


def _sent_to_model() -> dict[str, dict[str, Any]]:
    tools = create_mcp_tools_from_catalog(_catalog())
    return {t.name: convert_to_openai_tool(t)["function"]["parameters"] for t in tools}


def _declared(schema: dict[str, Any]) -> dict[str, Any]:
    """The non-null branch the model is shown for an optional field."""
    branches = [b for b in schema.get("anyOf", []) if b.get("type") != "null"]
    return branches[0] if branches else schema


def _lossy_tools() -> list[str]:
    return [
        name
        for name, tool in RAW_TOOLS.items()
        if any(set(p) - PLAIN_KEYS for p in tool["inputSchema"].get("properties", {}).values())
    ]


def test_the_fixture_covers_every_live_tool_and_each_lossy_shape():
    assert len(RAW_TOOLS) == 48
    kinds = {
        key
        for name in _lossy_tools()
        for p in RAW_TOOLS[name]["inputSchema"]["properties"].values()
        for key in set(p) - PLAIN_KEYS
    }
    assert {"enum", "items", "properties", "required"} <= kinds


def test_every_write_tool_in_the_catalog_takes_a_boolean_confirmed():
    """The write gate refuses a write tool that has no `confirmed` parameter."""
    present = WRITE_MCP_TOOL_NAMES & set(RAW_TOOLS)
    assert present == WRITE_MCP_TOOL_NAMES

    for name in sorted(present):
        props = RAW_TOOLS[name]["inputSchema"].get("properties", {})
        assert props.get("confirmed", {}).get("type") == "boolean", name


def test_every_declared_constraint_reaches_the_model():
    sent = _sent_to_model()
    lossy = _lossy_tools()
    assert len(lossy) == 28

    for name in lossy:
        sent_props = sent[name]["properties"]
        for param, raw in RAW_TOOLS[name]["inputSchema"]["properties"].items():
            shown = _declared(sent_props[param])
            for key in set(raw) - {"description", "default"}:
                assert shown.get(key) == raw[key], f"{name}.{param}.{key}"


def test_required_parameters_stay_required():
    sent = _sent_to_model()

    for name, tool in RAW_TOOLS.items():
        assert set(sent[name].get("required", [])) == set(
            tool["inputSchema"].get("required", [])
        ), name


def _announcement_args():
    parameters = _catalog_parameters("create_announcement")
    return _build_args_schema("create_announcement", parameters)


def _catalog_parameters(tool_name: str) -> list[dict[str, Any]]:
    for server in _catalog()["servers"]:
        for tool in server["tools"]:
            if tool["name"] == tool_name:
                return tool["parameters"]
    raise AssertionError(tool_name)


def test_the_object_the_model_got_wrong_still_reaches_the_server_to_be_rejected():
    args = _announcement_args()(
        title="t", body="b", summary="s", external_link={"url": "https://x", "text": "x"}
    )

    assert args.external_link == {"url": "https://x", "text": "x"}


def test_values_the_server_accepts_outside_its_listed_choices_still_pass():
    args = _announcement_args()(
        title="t", body="b", summary="s", where_to_share=["on the announcements page"]
    )

    assert args.where_to_share == ["on the announcements page"]


def test_an_object_sent_as_json_text_still_decodes():
    args = _announcement_args()(
        title="t", body="b", summary="s", external_link='{"uri": "https://x"}'
    )

    assert args.external_link == {"uri": "https://x"}


def test_a_value_of_the_wrong_type_is_still_rejected():
    with pytest.raises(ValidationError):
        _announcement_args()(title="t", body="b", summary="s", where_to_share="nope")


def test_the_catalog_keeps_each_parameters_raw_schema():
    param = next(
        p for p in _catalog_parameters("create_announcement") if p["name"] == "external_link"
    )

    assert param["json_schema"]["properties"]["uri"]["type"] == "string"
    assert param["json_schema"]["required"] == ["uri"]


def test_a_scalar_enum_still_accepts_a_value_outside_its_listed_choices():
    parameters = _catalog_parameters("get_chart_data")
    realm = next(p for p in parameters if p["name"] == "realm")
    assert "enum" in realm["json_schema"]

    args = _build_args_schema("get_chart_data", parameters)
    off_list = "not-a-listed-realm"
    assert off_list not in realm["json_schema"]["enum"]

    others = {
        "group_by": "g",
        "statistic": "s",
        "start_date": "2026-01-01",
        "end_date": "2026-02-01",
    }

    assert args(realm=off_list, **others).realm == off_list


def test_no_parameter_schema_uses_ref_defs_or_oneof():
    for name, tool in RAW_TOOLS.items():
        text = json.dumps(tool["inputSchema"])
        for keyword in ("$ref", "$defs", "oneOf"):
            assert keyword not in text, f"{name} uses {keyword}"


def test_the_stored_schema_leaves_out_what_the_catalog_already_keeps_apart():
    for name in _lossy_tools():
        for param in _catalog_parameters(name):
            assert not {"description", "default"} & set(param["json_schema"]), name


def test_trimming_the_stored_schema_leaves_the_model_facing_schema_as_declared():
    sent = _sent_to_model()
    for name, tool in RAW_TOOLS.items():
        for param, raw in tool["inputSchema"].get("properties", {}).items():
            shown = _declared(sent[name]["properties"][param])
            assert shown.get("description", raw.get("description")) == raw.get("description")
            assert shown.get("default", raw.get("default")) == raw.get("default")
