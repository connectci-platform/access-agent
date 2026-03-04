"""LangGraph definition for the ACCESS Documentation Agent.

This module defines the state graph that orchestrates query processing:
  START → classify → routing based on query type/domain → END

Query classification routes:
  - domain_agent: domain-specific react agent (announcements, jsm) → END
  - static: RAG retrieval → END (or fallback to plan if no match)
  - dynamic: plan → execute → evaluate → synthesize → END
  - combined: RAG retrieval (for context) → plan → execute → evaluate → synthesize → END

Domain agents handle interactive management tasks (create/update/delete) using
a react loop with direct MCP tool access. The general pipeline handles
informational queries (search, lookup, status).

With recovery and quality loops:
  - execute → recover (on failure) → execute or synthesize
  - evaluate → plan (if unhelpful) for retry
"""

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .state import ToolCatalog

from langgraph.graph import END, START, StateGraph

from ..telemetry import get_tracer
from .edges.routing import (
    should_execute_tools,
    should_recover_or_evaluate,
    should_retry_or_synthesize,
    should_retry_quality,
)
from .nodes import (
    classify_node,
    domain_agent_node,
    evaluate_node,
    execute_node,
    plan_node,
    rag_answer_node,
    recover_node,
    synthesize_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def route_by_classification(state: AgentState) -> Literal["domain_agent", "rag_answer", "plan"]:
    """Route based on query classification.

    Domain-first routing:
    - If domain is set: route to domain agent (announcements, jsm, etc.)

    RAG-primary routing (general pipeline):
    - static: Go to RAG for verified answer
    - combined: Go to RAG first to gather context, then continue to tools
    - dynamic: Skip RAG, go directly to tools

    Args:
        state: Current agent state with query_classification.

    Returns:
        Next node name.
    """
    classification = state.get("query_classification")

    # Domain agent takes priority when set
    if classification and classification.domain:
        logger.info(f"Routing to domain_agent ({classification.domain})")
        return "domain_agent"

    if classification and classification.query_type == "dynamic":
        # Dynamic queries skip RAG - they only need real-time data
        logger.info("Routing to plan (dynamic query, skipping RAG)")
        return "plan"

    # Both static and combined queries go through RAG first
    query_type = classification.query_type if classification else "unknown"
    logger.info(f"Routing to rag_answer ({query_type} query)")
    return "rag_answer"


def _rag_answer_is_weak(answer: str) -> bool:
    """Detect when RAG returned a hedged or unhelpful answer.

    The UKY RAG endpoint always returns *something* (it's an LLM), but when its
    retrieval context doesn't cover the topic it produces hedging language.
    These answers should fall through to MCP tools for better results.

    Args:
        answer: The RAG answer text.

    Returns:
        True if the answer appears to be a hedge/deflection.
    """
    lower = answer.lower()
    hedge_phrases = [
        "do not contain",
        "does not contain",
        "do not explicitly",
        "does not explicitly",
        "not provided in",
        "not mentioned in",
        "no specific information",
        "do not have specific information",
        "currently do not have",
        "open a support ticket",
        "open-a-ticket",
        "not available in the provided",
    ]
    return any(phrase in lower for phrase in hedge_phrases)


def route_after_rag(state: AgentState) -> Literal["end", "plan", "synthesize"]:
    """Route after RAG answer attempt.

    For static queries:
    - If RAG found a confident match (final_answer set by UKY) → END
    - If RAG found pgvector matches (no final_answer) → synthesize via LLM
    - If RAG hedged (weak answer) → fallback to plan (tools)
    - If no match → fallback to plan (tools)

    For combined queries:
    - Always continue to plan (tools) to get real-time data
    - RAG results are preserved in state for synthesis

    Args:
        state: Current agent state.

    Returns:
        Next node: "end", "synthesize", or "plan".
    """
    classification = state.get("query_classification")
    query_type = classification.query_type if classification else "static"

    # For combined queries, always continue to tools even if RAG found matches
    if query_type == "combined":
        rag_matches = state.get("rag_matches", [])
        if rag_matches:
            logger.info(
                f"Combined query: RAG found {len(rag_matches)} matches, "
                "continuing to plan for real-time data"
            )
        else:
            logger.info("Combined query: No RAG matches, continuing to plan")
        return "plan"

    # For static queries, end if RAG provided a final answer (e.g., UKY)
    final_answer = state.get("final_answer")
    if final_answer:
        if _rag_answer_is_weak(final_answer):
            logger.info("Static query: RAG answer is weak/hedged, falling back to tools")
            return "plan"
        return "end"

    # pgvector matches without final_answer → synthesize via LLM
    rag_matches = state.get("rag_matches", [])
    if rag_matches and state.get("rag_used"):
        logger.info(
            f"Static query: {len(rag_matches)} pgvector matches, routing to synthesize"
        )
        return "synthesize"

    # No RAG match for static query, fall back to tools
    logger.info("Static query: No RAG match, falling back to plan")
    return "plan"


def _build_graph_structure(
    builder: "StateGraph[AgentState]",
) -> "StateGraph[AgentState]":
    """Build the common graph structure with nodes and edges.

    The graph implements a flow with query classification and routing:

    1. Classify: Determine query type and domain
    2a. Domain path: domain_agent (react loop with MCP tools) → END
    2b. Static path: RAG lookup from Q&A service → END (or fallback to plan)
    2c. Dynamic/Combined path:
        - Plan: Analyze query and select tools
        - Execute: Run MCP tools
        - Recover: Handle failures (if any)
        - Evaluate: Check if results answer the question
        - Synthesize: Generate final answer

    With loops:
    - recover → execute (retry after recovery)
    - evaluate → plan (retry with different tools if unhelpful)
    - rag_answer → plan (fallback if no RAG match)

    Args:
        builder: A StateGraph builder to configure.

    Returns:
        The configured StateGraph builder.
    """
    # Add nodes
    builder.add_node("classify", classify_node)
    builder.add_node("domain_agent", domain_agent_node)
    builder.add_node("rag_answer", rag_answer_node)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("recover", recover_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("synthesize", synthesize_node)

    # Add edges
    # Start with classification
    builder.add_edge(START, "classify")

    # After classification, route based on query type (or domain)
    builder.add_conditional_edges(
        "classify",
        route_by_classification,
        {
            "domain_agent": "domain_agent",
            "rag_answer": "rag_answer",
            "plan": "plan",
        },
    )

    # Domain agent goes directly to END
    builder.add_edge("domain_agent", END)

    # After RAG answer: end, synthesize pgvector matches, or fallback to plan
    builder.add_conditional_edges(
        "rag_answer",
        route_after_rag,
        {
            "end": END,
            "synthesize": "synthesize",
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

    tracer = get_tracer("access-agent")

    # Create root span for entire agent execution
    with tracer.start_as_current_span(
        "agent.run",
        attributes={
            "agent.query": query[:200],  # Truncate long queries
            "agent.session_id": session_id,
            "agent.question_id": question_id,
            "agent.user": acting_user or "anonymous",
        },
    ) as root_span:
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

        # Add result attributes to root span
        tools_used = final_state.get("tools_used", [])
        final_answer = final_state.get("final_answer", "") or ""
        root_span.set_attribute("agent.tools_used", len(tools_used))
        root_span.set_attribute("agent.tool_names", ",".join(tools_used) if tools_used else "")
        root_span.set_attribute("agent.answer_length", len(final_answer))

        classification = final_state.get("query_classification")
        if classification:
            root_span.set_attribute("agent.query_type", classification.query_type)

        logger.info(f"Agent complete: tools_used={tools_used}, answer_length={len(final_answer)}")

        # Cast from Any (LangGraph's dynamic return) to AgentState
        result: AgentState = final_state
        return result
