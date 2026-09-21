"""Tests for agent state schema."""

from src.agent.profile import AllocatedResource, UserProfile
from src.agent.state import (
    QueryAnalysis,
    ToolCall,
    ToolResult,
    create_initial_state,
)


def test_create_initial_state(sample_catalog):
    """Test creating initial agent state."""
    state = create_initial_state(
        query="What GPUs are available?",
        session_id="test_session",
        question_id="test_question",
        tool_catalog=sample_catalog,
    )

    assert state["query"] == "What GPUs are available?"
    assert state["session_id"] == "test_session"
    assert state["question_id"] == "test_question"
    assert state["tool_catalog"] == sample_catalog
    assert state["planned_tools"] == []
    assert state["tool_results"] == []
    assert state["final_answer"] is None


def test_tool_call_model():
    """Test ToolCall model."""
    tool_call = ToolCall(
        step_id="step_1",
        tool_name="search_resources",
        server="compute-resources",
        arguments={"has_gpu": True},
    )

    assert tool_call.step_id == "step_1"
    assert tool_call.tool_name == "search_resources"
    assert tool_call.arguments == {"has_gpu": True}
    assert tool_call.depends_on == []


def test_tool_result_model():
    """Test ToolResult model."""
    result = ToolResult(
        step_id="step_1",
        tool_name="search_resources",
        server="compute-resources",
        success=True,
        data={"resources": [{"name": "Delta"}]},
        duration_ms=150,
    )

    assert result.success is True
    assert result.data["resources"][0]["name"] == "Delta"
    assert result.error is None


def test_create_initial_state_stores_profile_as_dict(sample_catalog):
    """State channels stay JSON-plain; the model is revalidated at the read site."""
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )
    state = create_initial_state(
        query="What GPUs are available?",
        session_id="test_session",
        question_id="test_question",
        tool_catalog=sample_catalog,
        profile=profile,
    )

    assert isinstance(state["profile"], dict)
    assert state["profile"] == profile.model_dump()


def test_create_initial_state_defaults_profile_none(sample_catalog):
    state = create_initial_state(
        query="What GPUs are available?",
        session_id="test_session",
        question_id="test_question",
        tool_catalog=sample_catalog,
    )

    assert state["profile"] is None


def test_query_analysis_model():
    """Test QueryAnalysis model."""
    analysis = QueryAnalysis(
        user_intent="Find GPU resources",
        entities_mentioned=["GPU", "resources"],
        requires_tools=True,
        confidence="high",
    )

    assert analysis.requires_tools is True
    assert "GPU" in analysis.entities_mentioned
