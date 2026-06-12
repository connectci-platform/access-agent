"""LangGraph definition for the ACCESS Documentation Agent.

Single-node design: every query flows START → tool_calling_loop → END.
The loop's system prompt and tool list (full MCP catalog plus
``search_access_documents``) let the LLM decide when to look things up,
when to call tools, and when to answer — no upstream classifier or
domain router. The legacy classify-then-loop and plan→execute→synthesize
paths have been removed; see ``archive/classify-then-loop`` and
``archive/legacy-chain`` for frozen comparator branches.
"""

import logging
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import ToolCatalog

from langgraph.graph import END, START, StateGraph

from ..telemetry import get_tracer
from .nodes.tool_calling_loop import tool_calling_loop_node
from .state import AgentState
from .turn_capture import record_trace_id

logger = logging.getLogger(__name__)


def _record_current_trace_id(span: Any) -> None:
    """Stash the root span's trace id (32-hex) in turn_capture.

    With telemetry disabled the span is non-recording and its trace_id is 0 —
    record nothing, so turn_reports.trace_id stays NULL rather than holding a
    dead link.
    """
    trace_id = span.get_span_context().trace_id
    if trace_id:
        record_trace_id(format(trace_id, "032x"))


def _build_graph_structure(
    builder: "StateGraph[AgentState]",
) -> "StateGraph[AgentState]":
    """Wire the single-node loop graph: START → tool_calling_loop → END.

    The node signature is ``dict[str, Any] -> dict[str, Any]`` (accepts the
    react-agent messages dict shape); mypy can't narrow that to AgentState
    at the StateGraph generic, so we suppress the type-var mismatch.
    """
    builder.add_node("tool_calling_loop", tool_calling_loop_node)  # type: ignore[type-var]
    builder.add_edge(START, "tool_calling_loop")
    builder.add_edge("tool_calling_loop", END)
    return builder


def create_agent_graph() -> Any:
    """Create the ACCESS Documentation Agent graph."""
    builder: StateGraph[AgentState] = StateGraph(AgentState)
    _build_graph_structure(builder)
    return builder.compile()


def create_async_checkpointer(db_uri: str) -> Any:
    """Create an async PostgreSQL checkpointer.

    Must be used as an async context manager:
        async with create_async_checkpointer(db_uri) as checkpointer:
            graph = create_checkpointed_graph(checkpointer)
    """
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    return AsyncPostgresSaver.from_conn_string(db_uri)


def create_checkpointed_graph(checkpointer: Any) -> Any:
    """Create graph with PostgreSQL checkpointing for durability."""
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
    """Run the agent on a query."""
    from .state import create_initial_state

    tracer = get_tracer("access-agent")

    with tracer.start_as_current_span(
        "agent.run",
        attributes={
            "agent.query": query[:200],
            "agent.session_id": session_id,
            "agent.question_id": question_id,
            "agent.user": acting_user or "anonymous",
        },
    ) as root_span:
        _record_current_trace_id(root_span)
        initial_state = create_initial_state(
            query=query,
            session_id=session_id,
            question_id=question_id,
            tool_catalog=tool_catalog,
            acting_user=acting_user,
            resource_context=resource_context,
        )

        logger.info(f"Running agent for query: {query[:50]}...")

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

                final_state = await graph.ainvoke(initial_state, config)
        else:
            graph = create_agent_graph()
            final_state = await graph.ainvoke(initial_state, {})

        tools_used = final_state.get("tools_used", [])
        final_answer = final_state.get("final_answer", "") or ""
        root_span.set_attribute("agent.tools_used", len(tools_used))
        root_span.set_attribute("agent.tool_names", ",".join(tools_used) if tools_used else "")
        root_span.set_attribute("agent.answer_length", len(final_answer))

        logger.info(f"Agent complete: tools_used={tools_used}, answer_length={len(final_answer)}")

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

    Yields ``(stream_type, data)`` tuples where ``stream_type`` is one of:
      - ``"custom"``: Status messages from nodes via ``get_stream_writer()``
      - ``"messages"``: LLM token chunks (with metadata including ``langgraph_node``)
      - ``"updates"``: State updates after each node completes
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
    ) as root_span:
        _record_current_trace_id(root_span)
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
