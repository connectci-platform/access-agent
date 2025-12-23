"""State schema for the ACCESS Documentation Agent.

Defines the TypedDict that flows through the LangGraph nodes,
containing all state needed for query processing.
"""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A planned tool call with parameters.

    Represents a single tool that the LLM has decided to call,
    with its arguments and optional dependencies on previous steps.
    """

    step_id: str = Field(description="Unique identifier for this step")
    tool_name: str = Field(description="Name of the MCP tool to call")
    server: str = Field(description="Name of the MCP server hosting this tool")
    arguments: dict[str, Any] = Field(
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
    data: Any = Field(default=None, description="Result data if successful")
    error: str | None = Field(default=None, description="Error message if failed")
    duration_ms: int = Field(default=0, description="Execution time in milliseconds")


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
    history: list[dict[str, Any]] = Field(default_factory=list)


class CompressedResult(TypedDict, total=False):
    """Compressed tool result for synthesis.

    Contains only the essential fields needed for answer generation.
    """

    tool_name: str
    success: bool
    data: dict[str, Any] | list[dict[str, Any]] | None
    error: str | None


class AgentState(TypedDict):
    """Main state schema for the ACCESS Documentation Agent.

    This TypedDict flows through all LangGraph nodes, with each node
    reading from and writing to specific fields.

    Node responsibilities:
    - plan: Reads query, tool_catalog, messages; writes query_analysis, planned_tools
    - execute: Reads planned_tools; writes tool_results, tools_used
    - synthesize: Reads query, tool_results, messages; writes final_answer, messages
    """

    # Conversation memory (accumulates across checkpoints via add_messages reducer)
    messages: Annotated[list[AnyMessage], add_messages]

    # Input fields (set at start)
    query: str
    session_id: str
    question_id: str
    tool_catalog: Annotated[dict[str, Any], "Full MCP tool catalog"]

    # Planning fields (set by plan node)
    query_analysis: Annotated[QueryAnalysis | None, "LLM analysis of user intent"]
    planned_tools: Annotated[list[ToolCall], "Tools selected for execution"]
    execution_strategy: Literal["sequential", "parallel", "mixed"]

    # Execution fields (set by execute node)
    tool_results: Annotated[list[ToolResult], "Results from tool execution"]
    tools_used: Annotated[list[str], "Names of tools that succeeded"]

    # Compression fields (set by compress node)
    compressed_results: Annotated[list[CompressedResult], "Compressed results for synthesis"]

    # Quality fields (for future quality loop)
    quality_evaluation: Annotated[QualityEvaluation | None, "Result quality assessment"]
    attempt_number: int
    max_attempts: int

    # Retry fields (for future error recovery)
    retry_context: Annotated[RetryContext | None, "Retry tracking context"]

    # Output fields (set by synthesize node)
    final_answer: Annotated[str | None, "Final answer to return to user"]


def create_initial_state(
    query: str,
    session_id: str,
    question_id: str,
    tool_catalog: dict[str, Any],
    max_attempts: int = 3,
) -> AgentState:
    """Create the initial state for a new query.

    Args:
        query: The user's question.
        session_id: Session identifier for conversation tracking.
        question_id: Unique identifier for this question.
        tool_catalog: The MCP tool catalog.
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
        # Planning
        query_analysis=None,
        planned_tools=[],
        execution_strategy="parallel",
        # Execution
        tool_results=[],
        tools_used=[],
        # Compression
        compressed_results=[],
        # Quality
        quality_evaluation=None,
        attempt_number=0,
        max_attempts=max_attempts,
        # Retry
        retry_context=RetryContext(start_time_ms=int(time.time() * 1000)),
        # Output
        final_answer=None,
    )
