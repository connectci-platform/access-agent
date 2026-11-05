"""A refused tool call never ran, so no record or eval treats it as a call."""

import pytest
from sqlalchemy import create_engine, inspect, text

from src.agent.domains.capabilities import get_capability_registry
from src.agent.state import ToolResult
from src.eval.coverage import parse_tool_result_calls
from src.eval.formatting import format_tool_results
from src.eval.multiturn import _tool_names, _turn_delta_state


def _results() -> list[ToolResult]:
    return [
        ToolResult(
            step_id="c1",
            tool_name="register_for_event",
            server="events",
            success=False,
            refused=True,
            error="refused:write_action",
        ),
        ToolResult(
            step_id="c2",
            tool_name="search_resources",
            server="compute-resources",
            success=True,
            data={"results": []},
        ),
    ]


def test_judge_formatting_and_coverage_leave_out_refused_calls():
    formatted = format_tool_results({"tool_results": _results()})

    assert [name for name, _ in parse_tool_result_calls(formatted)] == ["search_resources"]


def test_judge_formatting_of_a_refused_only_turn_is_empty():
    assert format_tool_results({"tool_results": _results()[:1]}) is None


def test_multiturn_tool_names_leave_out_refused_calls():
    assert _tool_names(_results()) == ["search_resources"]


def test_multiturn_view_carries_the_turns_refusals():
    refusals = [{"tool_name": "register_for_event", "reason": "write_action"}]
    view = _turn_delta_state({"messages": [], "tool_results": [], "tool_refusals": refusals})

    assert view["tool_refusals"] == refusals


@pytest.mark.parametrize("as_dicts", [False, True], ids=["models", "dicts"])
def test_capabilities_leave_out_refused_calls(as_dicts):
    results = [r.model_dump() for r in _results()] if as_dicts else _results()

    assert get_capability_registry().infer_capability_ids(results) == ["browse_resources"]


def test_permission_refusal_is_added_to_an_existing_table(monkeypatch, tmp_path):
    from src.config import settings
    from src.turn_reporter import TurnReportBase, TurnReporter

    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = create_engine(url)
    TurnReportBase.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE turn_reports DROP COLUMN permission_refusal"))
    monkeypatch.setattr(settings, "DATABASE_URL", url, raising=False)

    assert TurnReporter()._ensure_initialized()

    columns = {c["name"] for c in inspect(create_engine(url)).get_columns("turn_reports")}
    assert "permission_refusal" in columns
