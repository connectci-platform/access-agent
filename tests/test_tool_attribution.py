"""Tool attribution counts invocations, not distinct tool names.

``tool_count`` used to be ``len(tools_used)``, so a 16-call fan-out over one
tool reported 1; and on the budget-exhaustion path attribution ran after
``result_messages`` was reset, so a 27-call turn reported 0 — which the
reporting dashboard buckets as ``zero_tool``. Both are pinned below.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from src.agent.turn_capture import record_tool_timing, reset_turn_capture


def _state() -> dict:
    return {
        "messages": [HumanMessage(content="What GPU types do ACCESS resources have?")],
        "tool_catalog": {"tools": [], "quick_lookup": {}},
        "acting_user": None,
        "resource_context": None,
    }


def _fanout_result(n: int = 14) -> dict:
    """A completed thread with n calls to the SAME tool, as a fan-out looks."""
    messages: list = [HumanMessage(content="What GPU types do ACCESS resources have?")]
    for i in range(n):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "get_resource_hardware", "args": {"id": f"r{i}"}, "id": f"c{i}"}
                ],
            )
        )
        messages.append(ToolMessage(content=f"specs for r{i}", tool_call_id=f"c{i}"))
    messages.append(AIMessage(content="Here are the GPU types."))
    return {"messages": messages}


@pytest.mark.asyncio
async def test_fanout_counts_every_invocation_not_distinct_names():
    """14 calls to one tool is tool_call_count=14 and tools_used=[that tool]."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    reset_turn_capture()
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = _fanout_result(14)

    with patch("src.agent.nodes.tool_calling_loop.create_agent", return_value=mock_graph):
        # The real wrappers record one timing per call; simulate that here
        # since create_agent is mocked out.
        for _ in range(14):
            record_tool_timing("get_resource_hardware", 100)
        result = await tool_calling_loop_node(_state())

    assert result["tool_call_count"] == 14
    assert result["tools_used"] == ["get_resource_hardware"]


@pytest.mark.asyncio
async def test_exhaustion_reports_the_calls_it_actually_made():
    """A budget-exhausted turn must not report zero tools — the dashboard
    would classify it as a "zero_tool" turn."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    reset_turn_capture()
    mock_graph = AsyncMock()
    mock_graph.ainvoke.side_effect = GraphRecursionError("budget exhausted")

    with patch("src.agent.nodes.tool_calling_loop.create_agent", return_value=mock_graph):
        for _ in range(27):
            record_tool_timing("get_resource_hardware", 100)
        result = await tool_calling_loop_node(_state())

    assert result["tool_call_count"] == 27, "work done before the budget ran out must be reported"
    assert result["tools_used"] == ["get_resource_hardware"]
    # The safe decline is still what the user sees.
    assert "tool-turn budget" in result["final_answer"]


@pytest.mark.asyncio
async def test_distinct_tools_are_listed_once_each():
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    reset_turn_capture()
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [AIMessage(content="done")]}

    with patch("src.agent.nodes.tool_calling_loop.create_agent", return_value=mock_graph):
        record_tool_timing("search_access_documents", 50)
        record_tool_timing("get_resource_hardware", 60)
        record_tool_timing("get_resource_hardware", 70)
        result = await tool_calling_loop_node(_state())

    assert result["tool_call_count"] == 3
    assert result["tools_used"] == ["search_access_documents", "get_resource_hardware"]


@pytest.mark.asyncio
async def test_message_scan_is_the_fallback_when_no_timings_recorded():
    """Timings come from our own wrappers; if a tool were ever invoked by a
    path that bypasses them, attribution should still populate."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    reset_turn_capture()
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = _fanout_result(3)

    with patch("src.agent.nodes.tool_calling_loop.create_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(_state())  # no timings recorded

    assert result["tools_used"] == ["get_resource_hardware"]
    assert result["tool_call_count"] == 3


class TestConsumersUseInvocationCount:
    """The turn report feeds the reporting dashboard, which buckets
    tool_count == 0 as a "zero_tool" turn — so this number must be real."""

    def _report(self, final_state: dict) -> dict:
        from src.turn_reporter import _assemble_turn_report

        report, _tool_calls = _assemble_turn_report(
            final_state=final_state,
            session_id="s",
            turn_index=0,
            question_id="qid",
            query_text="q",
            duration_ms=1.0,
            acting_user=None,
            success=True,
            capabilities=[],
        )
        return report

    def test_prefers_invocation_count(self):
        report = self._report(
            {
                "tools_used": ["get_resource_hardware"],
                "tool_call_count": 14,
                "tool_results": [],
                "final_answer": "a",
            }
        )
        assert report["tool_count"] == 14

    def test_falls_back_to_distinct_names(self):
        report = self._report({"tools_used": ["a", "b"], "tool_results": [], "final_answer": "a"})
        assert report["tool_count"] == 2
