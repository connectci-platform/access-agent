"""LangGraph definition for the ACCESS Documentation Agent.

This module defines the state graph that orchestrates query processing:
  START → plan → execute → evaluate → synthesize → END

With recovery and quality loops:
  - execute → recover (on failure) → execute or synthesize
  - evaluate → plan (if unhelpful) for retry
"""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from .edges.routing import (
    should_execute_tools,
    should_recover_or_evaluate,
    should_retry_or_synthesize,
    should_retry_quality,
)
from .nodes import (
    evaluate_node,
    execute_node,
    plan_node,
    recover_node,
    synthesize_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def _build_graph_structure(builder: StateGraph) -> StateGraph:
    """Build the common graph structure with nodes and edges.

    The graph implements a flow with quality and error recovery loops:

    1. Plan: Analyze query and select tools
    2. Execute: Run MCP tools (if needed)
    3. Recover: Handle failures (if any tools failed)
    4. Evaluate: Check if results answer the question
    5. Synthesize: Generate final answer

    With loops:
    - recover → execute (retry after recovery)
    - evaluate → plan (retry with different tools if unhelpful)

    Args:
        builder: A StateGraph builder to configure.

    Returns:
        The configured StateGraph builder.
    """
    # Add nodes
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("recover", recover_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("synthesize", synthesize_node)

    # Add edges
    # Start with planning
    builder.add_edge(START, "plan")

    # After planning, decide if tools are needed
    builder.add_conditional_edges(
        "plan",
        should_execute_tools,
        {
            "execute": "execute",
            "synthesize": "synthesize",
        },
    )

    # After execution, check for failures
    builder.add_conditional_edges(
        "execute",
        should_recover_or_evaluate,
        {
            "recover": "recover",
            "evaluate": "evaluate",
        },
    )

    # After recovery, retry execution or give up
    builder.add_conditional_edges(
        "recover",
        should_retry_or_synthesize,
        {
            "execute": "execute",
            "synthesize": "synthesize",
        },
    )

    # After evaluation, retry planning or synthesize
    builder.add_conditional_edges(
        "evaluate",
        should_retry_quality,
        {
            "plan": "plan",
            "synthesize": "synthesize",
        },
    )

    # End after synthesis
    builder.add_edge("synthesize", END)

    return builder


def create_agent_graph() -> StateGraph:
    """Create the ACCESS Documentation Agent graph.

    Returns:
        A compiled StateGraph ready for execution.
    """
    builder = StateGraph(AgentState)
    _build_graph_structure(builder)
    return builder.compile()


def create_async_checkpointer(db_uri: str) -> Any:
    """Create an async PostgreSQL checkpointer.

    Must be used as an async context manager:
        async with create_async_checkpointer(db_uri) as checkpointer:
            graph = create_checkpointed_graph(checkpointer)

    Args:
        db_uri: PostgreSQL connection string.

    Returns:
        An async context manager that yields an AsyncPostgresSaver.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    return AsyncPostgresSaver.from_conn_string(db_uri)


def create_checkpointed_graph(checkpointer: Any) -> StateGraph:
    """Create graph with PostgreSQL checkpointing for durability.

    Args:
        checkpointer: A PostgresSaver instance.

    Returns:
        A compiled StateGraph with checkpointing enabled.
    """
    builder = StateGraph(AgentState)
    _build_graph_structure(builder)
    return builder.compile(checkpointer=checkpointer)


async def run_agent(
    query: str,
    session_id: str,
    question_id: str,
    tool_catalog: dict[str, Any],
    use_checkpointing: bool = False,
    db_uri: str | None = None,
) -> dict[str, Any]:
    """Run the agent on a query.

    Convenience function that creates the graph and runs it.

    Args:
        query: The user's question.
        session_id: Session identifier.
        question_id: Question identifier.
        tool_catalog: MCP tool catalog.
        use_checkpointing: Whether to use PostgreSQL checkpointing.
        db_uri: Database URI for checkpointing.

    Returns:
        The final agent state.
    """
    from .state import create_initial_state

    # Create initial state
    initial_state = create_initial_state(
        query=query,
        session_id=session_id,
        question_id=question_id,
        tool_catalog=tool_catalog,
    )

    # Run the graph
    logger.info(f"Running agent for query: {query[:50]}...")

    if use_checkpointing and db_uri:
        from langchain_core.messages import HumanMessage

        # Use async checkpointer as context manager
        async with create_async_checkpointer(db_uri) as checkpointer:
            # Setup tables on first use
            await checkpointer.setup()
            graph = create_checkpointed_graph(checkpointer)
            config = {"configurable": {"thread_id": session_id}}

            # Get previous state to preserve conversation history
            previous_state = await graph.aget_state(config)

            if previous_state.values:
                # We have previous conversation - get existing messages
                existing_messages = previous_state.values.get("messages", [])
                # Add the new user message to existing messages
                initial_state["messages"] = [*existing_messages, HumanMessage(content=query)]
                logger.info(
                    f"Resuming conversation with {len(existing_messages)} previous messages"
                )

            final_state = await graph.ainvoke(initial_state, config)
    else:
        graph = create_agent_graph()
        final_state = await graph.ainvoke(initial_state, {})

    logger.info(
        f"Agent complete: tools_used={final_state.get('tools_used', [])}, "
        f"answer_length={len(final_state.get('final_answer', '') or '')}"
    )

    return dict(final_state)
