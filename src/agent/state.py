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


class QualityEvaluation(BaseModel):
    """LLM evaluation of result quality (for future quality loop).

    Used to determine if tool results adequately answer the query
    or if retry with different approach is needed.
    """

    is_helpful: bool = Field(description="Whether results answer the question")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
    reason: str = Field(default="", description="Explanation of evaluation")
    missing_information: str | None = Field(
        default=None,
        description="What information is still needed",
    )


class RetryContext(BaseModel):
    """Context for retry logic (for future error recovery).

    Tracks retry attempts and history for error recovery.
    """

    max_retries_per_tool: int = 2
    max_retries_total: int = 5
    current_total_retries: int = 0
    timeout_budget_ms: int = 120000
    start_time_ms: int = 0
    history: list[dict[str, str | int]] = Field(default_factory=list)


class AgentState(TypedDict):
    """Main state schema for the ACCESS Documentation Agent.

    This TypedDict flows through all LangGraph nodes, with each node
    reading from and writing to specific fields.

    Node responsibilities:
    - classify: Reads query; writes query_classification
    - rag_answer: Reads query; writes rag_matches, final_answer (for static queries)
    - plan: Reads query, tool_catalog, messages, rag_matches; writes query_analysis, planned_tools
    - execute: Reads planned_tools; writes tool_results, tools_used
    - synthesize: Reads query, tool_results, rag_matches, messages; writes final_answer, messages
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
    personalization_context: Annotated[
        str | None, "Pre-formatted user profile text for system prompt injection"
    ]

    # Classification fields (set by classify node)
    query_classification: Annotated[QueryClassification | None, "Query type classification"]

    # RAG fields (set by rag_answer node)
    rag_matches: Annotated[list[RAGMatch], "Matches from RAG/Q&A service"]
    rag_used: Annotated[bool, "Whether RAG provided or augmented the answer"]

    # Planning fields (set by plan node)
    query_analysis: Annotated[QueryAnalysis | None, "LLM analysis of user intent"]
    planned_tools: Annotated[list[ToolCall], "Tools selected for execution"]
    execution_strategy: Literal["sequential", "parallel", "mixed"]

    # Execution fields (set by execute node)
    tool_results: Annotated[list[ToolResult], "Results from tool execution"]
    tools_used: Annotated[list[str], "Names of tools that succeeded"]

    # Quality fields (for quality loop)
    quality_evaluation: Annotated[QualityEvaluation | None, "Result quality assessment"]
    attempt_number: int
    max_attempts: int

    # Retry fields (for future error recovery)
    retry_context: Annotated[RetryContext | None, "Retry tracking context"]

    # Tracing (accumulated by every node via operator.add reducer)
    node_trace: Annotated[list[dict[str, Any]], operator.add]

    # Domain agent fields (set by domain_agent node)
    domain_completed: Annotated[
        bool | None,
        "Whether the domain agent called a tool (True) or is still gathering info (False)",
    ]

    # Output fields (set by synthesize node)
    final_answer: Annotated[str | None, "Final answer to return to user"]


def create_initial_state(
    query: str,
    session_id: str,
    question_id: str,
    tool_catalog: ToolCatalog,
    acting_user: str | None = None,
    resource_context: str | None = None,
    personalization_context: str | None = None,
    max_attempts: int = 3,
) -> AgentState:
    """Create the initial state for a new query.

    Args:
        query: The user's question.
        session_id: Session identifier for conversation tracking.
        question_id: Unique identifier for this question.
        tool_catalog: The MCP tool catalog.
        acting_user: ACCESS ID of user performing action (e.g., jsmith@access-ci.org).
        resource_context: RP slug for resource-scoped queries (e.g. 'delta').
        personalization_context: Pre-formatted user profile text for prompt injection.
        max_attempts: Maximum quality loop attempts.

    Returns:
        An initialized AgentState ready for the graph.
    """
    import time

    return AgentState(
        # Conversation memory - add the user's query as a message
        # The add_messages reducer will accumulate this with previous messages
        messages=[HumanMessage(content=query)],
        # Input
        query=query,
        session_id=session_id,
        question_id=question_id,
        tool_catalog=tool_catalog,
        acting_user=acting_user,
        resource_context=resource_context,
        personalization_context=personalization_context,
        # Classification
        query_classification=None,
        # RAG
        rag_matches=[],
        rag_used=False,
        # Planning
        query_analysis=None,
        planned_tools=[],
        execution_strategy="parallel",
        # Execution
        tool_results=[],
        tools_used=[],
        # Quality
        quality_evaluation=None,
        attempt_number=0,
        max_attempts=max_attempts,
        # Retry
        retry_context=RetryContext(start_time_ms=int(time.time() * 1000)),
        # Domain agent
        domain_completed=None,
        # Tracing
        node_trace=[],
        # Output
        final_answer=None,
    )
