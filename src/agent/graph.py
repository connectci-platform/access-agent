"""LangGraph definition for the ACCESS Documentation Agent.

This module defines the state graph that orchestrates query processing:
  START → classify → routing based on query type/domain → END

Query classification routes:
  - domain_agent: domain-specific react agent (announcements, jsm) → END
  - static: RAG retrieval → END (or fallback to plan if no match)
  - dynamic/combined: RAG + plan in parallel → execute → evaluate → synthesize → END

For combined/dynamic queries, UKY RAG and tool planning run concurrently.
UKY doesn't block the planner — both results are available for synthesis.

Domain agents handle interactive management tasks (create/update/delete) using
a react loop with direct MCP tool access. The general pipeline handles
informational queries (search, lookup, status).

With recovery and quality loops:
  - execute → recover (on failure) → execute or synthesize
  - evaluate → plan (if unhelpful) for retry
"""

import logging
from collections.abc import AsyncGenerator
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
    rag_and_plan_node,
    rag_answer_node,
    recover_node,
    synthesize_node,
)
from .state import AgentState

logger = logging.getLogger(__name__)


def route_by_classification(
    state: AgentState,
) -> Literal["rag_answer", "rag_and_plan"]:
    """Route after classification based on query type.

    For static and domain queries, UKY RAG runs first (sequential).
    Static queries may END after RAG if the answer is confident.
    Domain queries continue to domain_agent after RAG.

    For combined/dynamic queries, UKY RAG and tool planning run in
    parallel — the planner doesn't need the RAG result, so there's
    no reason to wait.

    Args:
        state: Current agent state with query_classification.

    Returns:
        "rag_answer" for static/domain, "rag_and_plan" for combined/dynamic.
    """
    classification = state.get("query_classification")
    query_type = classification.query_type if classification else "unknown"
    domain = classification.domain if classification else None

    # Static and domain queries go through sequential RAG-first path.
    # Static needs RAG result to decide whether to END or fall back to tools.
    # Domain needs RAG context before routing to domain_agent.
    if query_type == "static" or domain:
        logger.info(
            f"Routing to rag_answer (query_type={query_type}, domain={domain}) "
            "— sequential RAG-first path"
        )
        return "rag_answer"

    # Combined/dynamic queries run RAG and plan concurrently.
    # Both results merge in state for synthesis.
    logger.info(f"Routing to rag_and_plan (query_type={query_type}) — parallel RAG + tool planning")
    return "rag_and_plan"


def _rag_answer_is_deflection(answer: str) -> bool:
    """Detect when RAG returned a true deflection vs a hedge with good content.

    UKY often hedges in the first sentence ("The provided documents do not
    contain...") but then provides useful content — links, contacts, steps.
    Only reject answers that are genuine deflections (short, no real content).
    Keep answers where the hedge is just a preamble before substantive info.

    Args:
        answer: The RAG answer text.

    Returns:
        True only if the answer is a genuine deflection with no useful content.
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
        "not available in the provided",
    ]

    has_hedge = any(phrase in lower for phrase in hedge_phrases)
    if not has_hedge:
        return False

    # Hedge detected — but is there good content after it?
    # Indicators of substantive content despite the hedge:
    has_urls = "http" in lower
    has_email = "@" in answer
    is_long = len(answer) > 500

    # If the answer has links, emails, or substantial length, it's a
    # hedge-with-good-content — keep it (strip preamble at synthesis)
    if has_urls or has_email or is_long:
        logger.info(
            f"Hedge detected but answer has substance "
            f"(len={len(answer)}, urls={has_urls}, email={has_email}) — keeping"
        )
        return False

    # Short answer with hedge and no links/contacts = true deflection
    logger.info(f"Hedge detected, answer is a true deflection (len={len(answer)})")
    return True


def route_after_rag(state: AgentState) -> Literal["end", "plan", "domain_agent"]:
    """Route after RAG answer attempt.

    Every query has now consulted UKY. Decide what to do next based on
    classification and RAG result quality.

    For domain queries (e.g., announcements, jsm):
    - Continue to domain_agent, UKY content available in state as context

    For static queries:
    - If RAG found a confident match → END
    - If RAG hedged (weak answer) → fallback to plan (tools)
    - If no match → fallback to plan (tools)

    For combined/dynamic queries:
    - Always continue to plan (tools) for real-time/supplementary data
    - RAG results are preserved in state for synthesis

    Args:
        state: Current agent state.

    Returns:
        Next node: "end", "plan", or "domain_agent".
    """
    classification = state.get("query_classification")
    query_type = classification.query_type if classification else "static"
    domain = classification.domain if classification else None

    # Domain queries continue to domain_agent (UKY content now in state),
    # but only if that domain has at least one enabled capability.
    if domain:
        from .domains.capabilities import get_capability_registry

        if not get_capability_registry().is_domain_enabled(domain):
            logger.info(
                f"Domain '{domain}' disabled by capability registry, falling through to plan"
            )
            return "plan"
        rag_matches = state.get("rag_matches", [])
        logger.info(
            f"Domain query ({domain}): UKY provided {len(rag_matches)} matches, "
            "continuing to domain_agent"
        )
        return "domain_agent"

    # For combined/dynamic queries, always continue to tools
    # UKY content is preserved in state for synthesis
    if query_type in ("combined", "dynamic"):
        rag_matches = state.get("rag_matches", [])
        if rag_matches:
            logger.info(
                f"{query_type.title()} query: UKY provided {len(rag_matches)} matches, "
                "continuing to plan for supplementary data"
            )
        else:
            logger.info(f"{query_type.title()} query: No UKY matches, continuing to plan")
        return "plan"

    # For static queries, end if RAG provided a confident answer
    final_answer = state.get("final_answer")
    if final_answer:
        if _rag_answer_is_deflection(final_answer):
            logger.info("Static query: RAG answer is a true deflection, falling back to tools")
            return "plan"
        return "end"

    # No RAG match for static query, fall back to tools
    logger.info("Static query: No UKY match, falling back to plan")
    return "plan"


def _build_graph_structure(
    builder: "StateGraph[AgentState]",
) -> "StateGraph[AgentState]":
    """Build the common graph structure with nodes and edges.

    The graph implements two paths after classification:

    Sequential path (static/domain queries):
    1. Classify → RAG answer → route_after_rag
    2a. Static + confident: Serve UKY answer directly → END
    2b. Static + hedged/miss: Fall back to plan → execute → synthesize
    2c. Domain: Continue to domain_agent → END

    Parallel path (combined/dynamic queries):
    1. Classify → rag_and_plan (RAG + plan run concurrently)
    2. Execute tools → Evaluate → Synthesize (with both RAG + tool results)

    With loops:
    - recover → execute (retry after recovery)
    - evaluate → plan (retry with different tools if unhelpful)

    Args:
        builder: A StateGraph builder to configure.

    Returns:
        The configured StateGraph builder.
    """
    # Add nodes
    builder.add_node("classify", classify_node)
    builder.add_node("domain_agent", domain_agent_node)
    builder.add_node("rag_answer", rag_answer_node)
    builder.add_node("rag_and_plan", rag_and_plan_node)
    builder.add_node("plan", plan_node)
    builder.add_node("execute", execute_node)
    builder.add_node("recover", recover_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("synthesize", synthesize_node)

    # Add edges
    # Start with classification
    builder.add_edge(START, "classify")

    # After classification, route by query type:
    # - static/domain → sequential RAG-first path
    # - combined/dynamic → parallel RAG + plan path
    builder.add_conditional_edges(
        "classify",
        route_by_classification,
        {
            "rag_answer": "rag_answer",
            "rag_and_plan": "rag_and_plan",
        },
    )

    # Sequential path: after RAG answer, route based on classification + RAG quality
    builder.add_conditional_edges(
        "rag_answer",
        route_after_rag,
        {
            "end": END,
            "plan": "plan",
            "domain_agent": "domain_agent",
        },
    )

    # Domain agent goes directly to END
    builder.add_edge("domain_agent", END)

    # Parallel path: after rag_and_plan, planning is already done — go to execute
    builder.add_conditional_edges(
        "rag_and_plan",
        should_execute_tools,
        {
            "execute": "execute",
            "synthesize": "synthesize",
        },
    )

    # Sequential path: after planning, decide if tools are needed
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
    resource_context: str | None = None,
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
        resource_context: RP slug for resource-scoped queries (e.g. 'delta').
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
            resource_context=resource_context,
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


async def stream_agent(
    query: str,
    session_id: str,
    question_id: str,
    tool_catalog: "ToolCatalog",
    acting_user: str | None = None,
    resource_context: str | None = None,
    use_checkpointing: bool = False,
    db_uri: str | None = None,
) -> AsyncGenerator[tuple[str, Any], None]:
    """Stream agent execution, yielding events as they occur.

    Yields (stream_type, data) tuples where stream_type is one of:
    - "custom": Status messages from nodes via get_stream_writer()
    - "messages": LLM token chunks (with metadata including langgraph_node)
    - "updates": State updates after each node completes

    Args:
        query: The user's question.
        session_id: Session identifier.
        question_id: Question identifier.
        tool_catalog: MCP tool catalog.
        acting_user: ACCESS ID of user performing action.
        resource_context: RP slug for resource-scoped queries (e.g. 'delta').
        use_checkpointing: Whether to use PostgreSQL checkpointing.
        db_uri: Database URI for checkpointing.

    Yields:
        Tuples of (stream_type, chunk_data) from the LangGraph stream.
    """
    from .state import create_initial_state

    tracer = get_tracer("access-agent")

    with tracer.start_as_current_span(
        "agent.stream",
        attributes={
            "agent.query": query[:200],
            "agent.session_id": session_id,
            "agent.question_id": question_id,
            "agent.user": acting_user or "anonymous",
        },
    ):
        initial_state = create_initial_state(
            query=query,
            session_id=session_id,
            question_id=question_id,
            tool_catalog=tool_catalog,
            acting_user=acting_user,
            resource_context=resource_context,
        )

        stream_mode = ["custom", "messages", "updates"]

        if use_checkpointing and db_uri:
            from langchain_core.messages import HumanMessage

            async with create_async_checkpointer(db_uri) as checkpointer:
                await checkpointer.setup()
                graph = create_checkpointed_graph(checkpointer)
                config = {"configurable": {"thread_id": session_id}}

                previous_state = await graph.aget_state(config)
                if previous_state.values:
                    existing_messages = previous_state.values.get("messages", [])
                    initial_state["messages"] = [*existing_messages, HumanMessage(content=query)]
                    logger.info(
                        f"Resuming conversation with {len(existing_messages)} previous messages"
                    )

                async for stream_type, chunk in graph.astream(
                    initial_state, config, stream_mode=stream_mode
                ):
                    yield stream_type, chunk
        else:
            graph = create_agent_graph()
            async for stream_type, chunk in graph.astream(
                initial_state, {}, stream_mode=stream_mode
            ):
                yield stream_type, chunk
