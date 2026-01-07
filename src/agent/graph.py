"""LangGraph definition for the ACCESS Documentation Agent.

This module defines the state graph that orchestrates query processing:
  START → classify → (static_answer | plan → execute → evaluate → synthesize) → END

Query classification routes:
  - static: Fine-tuned model answers directly (no tools)
  - dynamic: Full agent workflow with MCP tools
  - combined: Fine-tuned model + MCP tool augmentation

With recovery and quality loops:
  - execute → recover (on failure) → execute or synthesize
  - evaluate → plan (if unhelpful) for retry
"""

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .state import ToolCatalog

from langgraph.graph import END, START, StateGraph

from .edges.routing import (
    should_execute_tools,
    should_recover_or_evaluate,
    should_retry_or_synthesize,
    should_retry_quality,
)
from .nodes import (
    classify_node,
    evaluate_node,
    execute_node,
    plan_node,
    recover_node,
    static_answer_node,
    synthesize_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def route_by_classification(state: AgentState) -> Literal["static_answer", "plan"]:
    """Route based on query classification.

    Args:
        state: Current agent state with query_classification.

    Returns:
        Next node: "static_answer" for static queries, "plan" for dynamic/combined.
    """
    classification = state.get("query_classification")

    if classification and classification.query_type == "static":
        logger.info("Routing to static_answer (fine-tuned model)")
        return "static_answer"

    # dynamic and combined both go through the tool-based workflow
    logger.info(
        f"Routing to plan (query_type={classification.query_type if classification else 'unknown'})"
    )
    return "plan"


def route_after_static(state: AgentState) -> Literal["end", "plan"]:
    """Route after static answer attempt.

    If static answer succeeded, go to END.
    If it failed and reclassified as dynamic, go to plan.

    Args:
        state: Current agent state.

    Returns:
        Next node: "end" if answer generated, "plan" if fallback needed.
    """
    if state.get("final_answer"):
        return "end"

    # Static answer failed, fall back to dynamic path
    logger.info("Static answer failed, falling back to plan")
    return "plan"


def _build_graph_structure(
    builder: "StateGraph[AgentState]",
) -> "StateGraph[AgentState]":
    """Build the common graph structure with nodes and edges.

    The graph implements a flow with query classification and routing:

    1. Classify: Determine if query is static, dynamic, or combined
    2a. Static path: Call fine-tuned model directly → END
    2b. Dynamic/Combined path:
        - Plan: Analyze query and select tools
        - Execute: Run MCP tools
        - Recover: Handle failures (if any)
        - Evaluate: Check if results answer the question
        - Synthesize: Generate final answer

    With loops:
    - recover → execute (retry after recovery)
    - evaluate → plan (retry with different tools if unhelpful)
    - static_answer → plan (fallback if static fails)

    Args:
        builder: A StateGraph builder to configure.

    Returns:
        The configured StateGraph builder.
    """
    # Add nodes
    builder.add_node("classify", classify_node)
    builder.add_node("static_answer", static_answer_node)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("recover", recover_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("synthesize", synthesize_node)

    # Add edges
    # Start with classification
    builder.add_edge(START, "classify")

    # After classification, route based on query type
    builder.add_conditional_edges(
        "classify",
        route_by_classification,
        {
            "static_answer": "static_answer",
            "plan": "plan",
        },
    )

    # After static answer, either end or fallback to plan
    builder.add_conditional_edges(
        "static_answer",
        route_after_static,
        {
            "end": END,
            "plan": "plan",
        },
    )

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


def create_agent_graph() -> Any:
    """Create the ACCESS Documentation Agent graph.

    Returns:
        A compiled StateGraph ready for execution.

    Note:
        Return type is Any due to LangGraph's complex generic types
        that vary across versions.
    """
    builder: StateGraph[AgentState] = StateGraph(AgentState)
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

    Note:
        Return type is Any due to LangGraph's complex generic types.
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    return AsyncPostgresSaver.from_conn_string(db_uri)


def create_checkpointed_graph(checkpointer: Any) -> Any:
    """Create graph with PostgreSQL checkpointing for durability.

    Args:
        checkpointer: A PostgresSaver instance.

    Returns:
        A compiled StateGraph with checkpointing enabled.

    Note:
        Types are Any due to LangGraph's complex generic types.
    """
    builder: StateGraph[AgentState] = StateGraph(AgentState)
    _build_graph_structure(builder)
    return builder.compile(checkpointer=checkpointer)


async def run_agent(
    query: str,
    session_id: str,
    question_id: str,
    tool_catalog: "ToolCatalog",
    acting_user: str | None = None,
    use_checkpointing: bool = False,
    db_uri: str | None = None,
) -> AgentState:
    """Run the agent on a query.

    Convenience function that creates the graph and runs it.

    Args:
        query: The user's question.
        session_id: Session identifier.
        question_id: Question identifier.
        tool_catalog: MCP tool catalog.
        acting_user: ACCESS ID of user performing action (e.g., jsmith@access-ci.org).
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
        acting_user=acting_user,
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

    # Cast from Any (LangGraph's dynamic return) to AgentState
    result: AgentState = final_state
    return result
