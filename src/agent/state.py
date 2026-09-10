"""State schema for the ACCESS Documentation Agent.

Defines the TypedDict that flows through the LangGraph nodes,
containing all state needed for query processing.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

# Type aliases for dynamic MCP data structures
# These are JSON-like structures that vary by tool/server
ToolArguments = dict[str, str | int | float | bool | list[str] | dict[str, str | list[str]] | None]
ToolResultData = dict[str, object] | list[object] | str | None
ToolCatalog = dict[str, dict[str, object]]


class ToolCall(BaseModel):
    """A planned tool call with parameters.

    Represents a single tool that the LLM has decided to call,
    with its arguments and optional dependencies on previous steps.
    """

    step_id: str = Field(description="Unique identifier for this step")
    tool_name: str = Field(description="Name of the MCP tool to call")
    server: str = Field(description="Name of the MCP server hosting this tool")
    arguments: ToolArguments = Field(
        default_factory=dict,
        description="Arguments to pass to the tool",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Step IDs this step depends on (for parameter resolution)",
    )


class ToolResult(BaseModel):
    """Result from executing a tool.

    Captures the outcome of a tool call, including success/failure,
    data or error, and timing information.
    """

    step_id: str = Field(description="Step ID this result corresponds to")
    tool_name: str = Field(description="Name of the tool that was called")
    server: str = Field(description="Server the tool was called on")
    success: bool = Field(description="Whether the call succeeded")
    data: ToolResultData = Field(default=None, description="Result data if successful")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_ms: int = Field(default=0, description="Execution time in milliseconds")
    arguments: ToolArguments = Field(
        default_factory=dict, description="Arguments the tool was called with"
    )
    message_id: str = Field(
        default="",
        description=(
            "LangChain message id of the source ToolMessage. Lets multi-turn "
            "consumers slice the cumulative, rebuilt tool_results down to one turn "
            "by testing membership in the set of ids appearing after the last "
            "HumanMessage of the OUTER merged thread. An id survives the "
            "add_messages merge; a positional index does not, because the list the "
            "loop stamps against is the inner, possibly compaction-rewritten one. "
            "Additive metadata; empty means 'unpositioned'."
        ),
    )


class RAGMatch(BaseModel):
    """A matched Q&A pair from the RAG service.

    Represents a verified answer from the Q&A database.
    """

    id: str = Field(description="Unique ID of the Q&A pair")
    question: str = Field(description="The matched question")
    answer: str = Field(description="The verified answer")
    domain: str = Field(description="Domain (e.g., compute-resources, software-discovery)")
    entity_id: str = Field(description="Entity ID for citation")
    similarity_score: float = Field(description="Semantic similarity score (0-1)")
    metadata: dict[str, object] = Field(default_factory=dict)


class QueryClassification(BaseModel):
    """Classification of query type for routing.

    Determines whether the query can be answered via RAG
    directly (static), requires live MCP data (dynamic), or both (combined).
    """

    query_type: Literal["static", "dynamic", "combined"] = Field(
        description="Type of query: static (model knows), dynamic (needs live data), combined (both)"
    )
    reason: str = Field(
        default="",
        description="Brief explanation of why this classification was chosen",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence in the classification",
    )
    expanded_query: str = Field(
        default="",
        description="Query rewritten as a standalone question with context resolved",
    )
    domain: str | None = Field(
        default=None,
        description="Domain agent to route to (e.g. 'announcements', 'jsm'), or None for general pipeline",
    )
    rag_endpoint: Literal["general", "xdmod"] | None = Field(
        default=None,
        description="UKY RAG endpoint to use: 'general' for ACCESS docs, 'xdmod' for XDMoD Q&A, or None to skip",
    )
    capability_id: str | None = Field(
        default=None,
        description="Primary capability exercised (e.g. 'open_ticket', 'check_allocations'). Set by classifier or inferred post-execution.",
    )


class QueryAnalysis(BaseModel):
    """LLM analysis of the user's query.

    Captures the LLM's understanding of what the user is asking for
    and whether tools are needed to answer.
    """

    user_intent: str = Field(description="Summary of what the user wants")
    entities_mentioned: list[str] = Field(
        default_factory=list,
        description="Key entities mentioned (resources, systems, etc.)",
    )
    requires_tools: bool = Field(
        default=True,
        description="Whether MCP tools are needed to answer",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description="Confidence in the analysis",
    )


class AgentState(TypedDict):
    """Main state schema for the ACCESS Documentation Agent.

    Flows through the LangGraph nodes (today: one node — the tool_calling_loop).
    The loop reads inputs, writes tool_results / tools_used / node_trace as it
    runs tools, and writes final_answer at the end.

    Some fields below are read by downstream consumers (API response metadata,
    telemetry spans, eval judge context) but are no longer written by any node
    on the current path. They survive here for compatibility with those
    consumers; removing them requires updating the consumers too.
    """

    # Conversation memory (accumulates across checkpoints via add_messages reducer)
    messages: Annotated[list[AnyMessage], add_messages]

    # Input fields (set at start)
    query: str
    session_id: str
    question_id: str
    tool_catalog: Annotated[ToolCatalog, "Full MCP tool catalog"]
    acting_user: Annotated[
        str | None, "ACCESS ID of user performing action (e.g., jsmith@access-ci.org)"
    ]
    resource_context: Annotated[str | None, "RP slug for resource-scoped queries (e.g. 'delta')"]

    # Read by api/routes.py for response metadata; not written on the current path.
    query_classification: Annotated[QueryClassification | None, "Query type classification"]

    # Read by eval/runner.py for judge context; not written on the current path.
    rag_matches: Annotated[list[RAGMatch], "Matches from RAG/Q&A service"]

    # Read by api/routes.py for response metadata; not written on the current path.
    query_analysis: Annotated[QueryAnalysis | None, "LLM analysis of user intent"]

    # Read by telemetry/spans.py for span attributes; not written on the current path.
    planned_tools: Annotated[list[ToolCall], "Tools selected for execution"]

    # Loop populates these as it calls tools.
    tool_results: Annotated[list[ToolResult], "Results from tool execution"]
    tools_used: Annotated[
        list[str],
        "DISTINCT names of tools the loop attempted (success or failure); inspect tool_results[].success for outcome. See tool_call_count for how many invocations those names cover",
    ]
    tool_call_count: Annotated[
        int,
        "Total tool INVOCATIONS this turn, counted at the tool wrapper; a 14-call fan-out over one tool is 14 here and one entry in tools_used",
    ]

    # Tracing (accumulated by every node via operator.add reducer)
    node_trace: Annotated[list[dict[str, Any]], operator.add]

    # Read by api/routes.py; the loop does not write it, so reads default to None
    # (treated as "completed") — see is_complete() in routes.py.
    domain_completed: Annotated[
        bool | None,
        "Domain-agent completion flag; legacy — no node writes this on the current path",
    ]

    # Output (set by the loop at end of run).
    final_answer: Annotated[str | None, "Final answer to return to user"]

    # Output (turn-scoped token total summed by the loop's usage callback).
    total_tokens: Annotated[int | None, "Turn-scoped total tokens for this turn's LLM calls"]

    # Output (per-LLM-call timing/tokens recorded by the loop's usage callback).
    model_calls: Annotated[
        list[dict[str, Any]] | None,
        "Per-LLM-call {index, duration_ms, total_tokens} entries for this turn",
    ]


def create_initial_state(
    query: str,
    session_id: str,
    question_id: str,
    tool_catalog: ToolCatalog,
    acting_user: str | None = None,
    resource_context: str | None = None,
) -> AgentState:
    """Create the initial state for a new query.

    Args:
        query: The user's question.
        session_id: Session identifier for conversation tracking.
        question_id: Unique identifier for this question.
        tool_catalog: The MCP tool catalog.
        acting_user: ACCESS ID of user performing action (e.g., jsmith@access-ci.org).
        resource_context: RP slug for resource-scoped queries (e.g. 'delta').

    Returns:
        An initialized AgentState ready for the graph.
    """
    return AgentState(
        # Conversation memory - add the user's query as a message
        # The add_messages reducer will accumulate this with previous messages
        messages=[HumanMessage(content=query)],
        # Inputs
        query=query,
        session_id=session_id,
        question_id=question_id,
        tool_catalog=tool_catalog,
        acting_user=acting_user,
        resource_context=resource_context,
        # Read by consumers but not written on the current path — initialize empty/None.
        query_classification=None,
        rag_matches=[],
        query_analysis=None,
        planned_tools=[],
        # Written by the loop as it runs.
        tool_results=[],
        tools_used=[],
        tool_call_count=0,
        node_trace=[],
        # Legacy domain-agent flag; the loop does not write it.
        domain_completed=None,
        # Output.
        final_answer=None,
        total_tokens=None,
        model_calls=None,
    )
