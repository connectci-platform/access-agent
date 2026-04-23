"""Tests for tool_calling_loop_node (launch Phase 3).

Strategy: mock the LLM + tools at the create_react_agent boundary so the
node's orchestration logic is tested without live API calls. Covers:

1. Basic flow: query in → LLM picks no tools → direct answer out.
2. Tool-calling flow: LLM emits one tool call → tool runs → LLM produces answer.
3. Multi-tool flow: LLM chains two tools using results from the first.
4. Tool failure: LLM sees a failed tool result and recovers gracefully.
5. No tools available: node handles empty tool_catalog without crashing.
6. Message threading: caller's messages are preserved in output state.
7. System prompt: builds correctly from state (rag_matches, acting_user, domain).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


@pytest.fixture
def base_state():
    """Minimal AgentState fields the node consumes."""
    return {
        "messages": [HumanMessage(content="what resources have GPUs?")],
        "query": "what resources have GPUs?",
        "tool_catalog": {
            "servers": [
                {
                    "server": "compute-resources",
                    "tools": [
                        {
                            "name": "search_resources",
                            "description": "Search for compute resources",
                            "inputSchema": {
                                "properties": {"has_gpu": {"type": "boolean"}},
                                "required": [],
                            },
                        }
                    ],
                }
            ]
        },
        "acting_user": None,
        "rag_matches": [],
        "query_classification": None,
    }


@pytest.mark.asyncio
async def test_node_exists_and_is_importable():
    """Sanity: the node function exists at the expected path."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    assert callable(tool_calling_loop_node)


@pytest.mark.asyncio
async def test_direct_answer_no_tools_called(base_state):
    """LLM that returns a final answer immediately produces final_answer in state."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    final_message = AIMessage(content="ACCESS has several GPU resources: Delta, FASTER, ...")

    # Mock create_react_agent to return a compiled-graph-like object whose
    # ainvoke returns the expected messages structure.
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [
            *base_state["messages"],
            final_message,
        ]
    }

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(base_state)

    assert result["final_answer"] == final_message.content
    assert any(isinstance(m, AIMessage) for m in result["messages"])
    assert result.get("tools_used") == []


@pytest.mark.asyncio
async def test_single_tool_call_then_answer(base_state):
    """LLM calls one tool, sees result, produces answer — tools_used tracks the call."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "search_resources", "args": {"has_gpu": True}}],
    )
    tool_result_msg = ToolMessage(
        content='{"resources": [{"name": "Delta"}, {"name": "FASTER"}]}',
        tool_call_id="call_1",
    )
    final_msg = AIMessage(content="ACCESS has GPU resources including Delta and FASTER.")

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [*base_state["messages"], tool_call_msg, tool_result_msg, final_msg]
    }

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(base_state)

    assert result["final_answer"] == final_msg.content
    assert "search_resources" in result["tools_used"]


@pytest.mark.asyncio
async def test_tool_failure_is_recovered_or_reported(base_state):
    """A failed tool result should not crash the node; LLM's response is returned."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    tool_call = AIMessage(
        content="",
        tool_calls=[{"id": "c1", "name": "search_resources", "args": {}}],
    )
    failure = ToolMessage(
        content='{"error": "timeout contacting compute-resources MCP server"}',
        tool_call_id="c1",
    )
    recovery_answer = AIMessage(
        content="I wasn't able to query the live resource list. You can see the "
        "current list at https://access-ci.org/resources."
    )

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [*base_state["messages"], tool_call, failure, recovery_answer]
    }

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(base_state)

    assert result["final_answer"] == recovery_answer.content
    # tools_used still lists the attempted call so traces are honest
    assert "search_resources" in result["tools_used"]


@pytest.mark.asyncio
async def test_empty_tool_catalog_still_produces_answer(base_state):
    """Node must not crash when no tools are available."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    state = {**base_state, "tool_catalog": {"servers": []}}
    answer = AIMessage(content="I don't have live tools available right now, but ...")

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [*state["messages"], answer]}

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(state)

    assert result["final_answer"] == answer.content
    assert result.get("tools_used") == []


@pytest.mark.asyncio
async def test_messages_are_accumulated_not_replaced(base_state):
    """The node must return the full message history (caller-supplied + loop-generated)."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    final = AIMessage(content="ok")
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [*base_state["messages"], final]}

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(base_state)

    # The original HumanMessage must still be present
    assert any(
        isinstance(m, HumanMessage) and m.content == "what resources have GPUs?"
        for m in result["messages"]
    )


@pytest.mark.asyncio
async def test_system_prompt_includes_acting_user_when_authenticated(base_state):
    """When acting_user is set, the system prompt should mention it."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    state = {**base_state, "acting_user": "jsmith@access-ci.org"}
    answer = AIMessage(content="ok")
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [*state["messages"], answer]}

    captured_prompt = {}

    def capture_prompt(**kwargs):  # type: ignore[no-untyped-def]
        captured_prompt["prompt"] = kwargs.get("prompt")
        return mock_graph

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", side_effect=capture_prompt):
        await tool_calling_loop_node(state)

    assert "jsmith@access-ci.org" in captured_prompt["prompt"]


@pytest.mark.asyncio
async def test_system_prompt_includes_rag_context_when_present(base_state):
    """When rag_matches is non-empty, the system prompt should embed them."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node
    from src.agent.state import RAGMatch

    state = {
        **base_state,
        "rag_matches": [
            RAGMatch(
                id="rag_001",
                question="What GPUs exist?",
                answer="Delta has NVIDIA A100s.",
                domain="compute-resources",
                entity_id="delta",
                similarity_score=0.92,
            )
        ],
    }
    answer = AIMessage(content="ok")
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": [*state["messages"], answer]}

    captured_prompt = {}

    def capture_prompt(**kwargs):  # type: ignore[no-untyped-def]
        captured_prompt["prompt"] = kwargs.get("prompt")
        return mock_graph

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", side_effect=capture_prompt):
        await tool_calling_loop_node(state)

    assert "Delta has NVIDIA A100s" in captured_prompt["prompt"]


@pytest.mark.asyncio
async def test_tool_results_backfilled_from_messages(base_state):
    """Loop must populate state.tool_results from ToolMessage content for eval-scorer parity."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "search_resources", "args": {"has_gpu": True}}],
    )
    success_result = ToolMessage(
        content='{"resources": [{"name": "Delta"}]}',
        tool_call_id="call_1",
    )
    tool_call_msg_2 = AIMessage(
        content="",
        tool_calls=[{"id": "call_2", "name": "search_resources", "args": {}}],
    )
    failed_result = ToolMessage(
        content='{"error": "timeout"}',
        tool_call_id="call_2",
    )
    final = AIMessage(content="Found Delta.")

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [
            *base_state["messages"],
            tool_call_msg,
            success_result,
            tool_call_msg_2,
            failed_result,
            final,
        ]
    }

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(base_state)

    tool_results = result["tool_results"]
    assert len(tool_results) == 2

    successes = [r for r in tool_results if r.success]
    failures = [r for r in tool_results if not r.success]
    assert len(successes) == 1
    assert len(failures) == 1

    assert successes[0].tool_name == "search_resources"
    assert successes[0].step_id == "call_1"
    assert successes[0].data == {"resources": [{"name": "Delta"}]}
    assert successes[0].arguments == {"has_gpu": True}

    assert failures[0].tool_name == "search_resources"
    assert failures[0].step_id == "call_2"
    assert failures[0].error == "timeout"


# ---------------------------------------------------------------------------
# Hardening fixes (Phase 3 final review)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_final_answer_is_none_when_no_ai_message_content(base_state):
    """When the loop emits only tool calls + tool results (no AIMessage text),
    final_answer should be None — not "" — so downstream can distinguish
    "loop produced nothing" from "LLM emitted an empty string"."""
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    tool_call = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "search_resources", "args": {}}],
    )
    tool_result = ToolMessage(
        content='{"resources": []}',
        tool_call_id="call_1",
    )

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "messages": [*base_state["messages"], tool_call, tool_result]
    }

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(base_state)

    assert result["final_answer"] is None
    # node_trace should reflect 0 answer_length, not crash on len(None)
    assert result["node_trace"][0]["answer_length"] == 0


@pytest.mark.asyncio
async def test_recursion_limit_error_produces_graceful_response(base_state):
    """GraphRecursionError from create_react_agent is caught and converted to
    a user-facing apology pointing at the support ticket path. The original
    input messages are preserved so the caller sees at minimum their query."""
    from langgraph.errors import GraphRecursionError

    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    mock_graph = AsyncMock()
    mock_graph.ainvoke.side_effect = GraphRecursionError("test recursion limit")

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        # Must NOT re-raise
        result = await tool_calling_loop_node(base_state)

    answer = result["final_answer"]
    assert answer is not None
    assert isinstance(answer, str)
    assert len(answer) > 0
    # Flexible match on the user-facing wording
    lowered = answer.lower()
    assert "support" in lowered or "try rephrasing" in lowered

    # Input messages are preserved
    assert result["messages"] == base_state["messages"]


@pytest.mark.asyncio
async def test_orphan_tool_messages_are_counted(base_state):
    """A ToolMessage with no matching AIMessage tool_call.id is silently dropped
    from tool_results (existing behavior) but tracked in orphan_count so
    telemetry surfaces the rare edge case."""
    from src.agent.nodes.tool_calling_loop import (
        _build_tool_results,
        tool_calling_loop_node,
    )

    # One paired call+result, plus one orphan ToolMessage (no AIMessage anchors it).
    valid_call = AIMessage(
        content="",
        tool_calls=[{"id": "call_1", "name": "search_resources", "args": {}}],
    )
    valid_result = ToolMessage(
        content='{"resources": [{"name": "Delta"}]}',
        tool_call_id="call_1",
    )
    orphan_result = ToolMessage(
        content='{"resources": []}',
        tool_call_id="call_orphan",  # no AIMessage has this id
    )
    final = AIMessage(content="Found Delta.")

    messages = [
        *base_state["messages"],
        valid_call,
        valid_result,
        orphan_result,
        final,
    ]

    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {"messages": messages}

    with patch("src.agent.nodes.tool_calling_loop.create_react_agent", return_value=mock_graph):
        result = await tool_calling_loop_node(base_state)

    # Orphan excluded from tool_results — consistent with existing behavior
    tool_message_count = sum(1 for m in messages if isinstance(m, ToolMessage))
    assert tool_message_count == 2
    assert len(result["tool_results"]) == 1
    assert result["tool_results"][0].step_id == "call_1"

    # Sanity-check the helper directly returns the orphan count
    tool_results, orphan_count = _build_tool_results(messages, [])
    assert orphan_count == 1
    assert len(tool_results) == 1


# ---------------------------------------------------------------------------
# Task 5: graph routing with the USE_TOOL_CALLING_LOOP feature flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_registers_tool_calling_loop_node():
    """The tool_calling_loop node is registered regardless of flag state."""
    from src.agent.graph import create_agent_graph

    graph = create_agent_graph()
    graph_obj = graph.get_graph()
    assert "tool_calling_loop" in graph_obj.nodes


@pytest.mark.asyncio
async def test_route_after_rag_uses_tool_calling_loop_when_flag_on(monkeypatch):
    """With flag on, route_after_rag returns tool_calling_loop where it would have returned plan."""
    from src.agent.graph import route_after_rag

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", True)

    # No classification, no final_answer: hits the "no UKY match, fall back" branch
    state = {"query_classification": None, "final_answer": None}
    assert route_after_rag(state) == "tool_calling_loop"


@pytest.mark.asyncio
async def test_route_after_rag_preserves_plan_when_flag_off(monkeypatch):
    """With flag off (default), route_after_rag returns plan as before."""
    from src.agent.graph import route_after_rag

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", False)

    state = {"query_classification": None, "final_answer": None}
    assert route_after_rag(state) == "plan"


@pytest.mark.asyncio
async def test_route_after_rag_still_ends_on_confident_static_when_flag_on(monkeypatch):
    """Flag doesn't change the 'confident RAG → END' decision."""
    from src.agent.graph import route_after_rag

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", True)

    # final_answer present, non-deflection (no hedge phrases): static/end
    state = {
        "query_classification": None,
        "final_answer": (
            "ACCESS has multiple GPU resources. "
            "See https://access-ci.org/resources for the full list."
        ),
    }
    assert route_after_rag(state) == "end"


@pytest.mark.asyncio
async def test_route_by_classification_forces_rag_answer_when_flag_on(monkeypatch):
    """With flag on, combined/dynamic queries route to rag_answer so the loop can consume RAG context."""
    from src.agent.graph import route_by_classification
    from src.agent.state import QueryClassification

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", True)

    # Combined query that would normally go to rag_and_plan
    state = {
        "query_classification": QueryClassification(query_type="combined"),
    }
    assert route_by_classification(state) == "rag_answer"


@pytest.mark.asyncio
async def test_route_by_classification_preserves_rag_and_plan_when_flag_off(monkeypatch):
    """Default path unchanged: combined/dynamic → rag_and_plan."""
    from src.agent.graph import route_by_classification
    from src.agent.state import QueryClassification

    monkeypatch.setattr("src.config.settings.USE_TOOL_CALLING_LOOP", False)

    state = {
        "query_classification": QueryClassification(query_type="combined"),
    }
    assert route_by_classification(state) == "rag_and_plan"
