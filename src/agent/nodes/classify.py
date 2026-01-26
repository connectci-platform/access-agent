"""Query classification node.

Classifies queries as static, dynamic, or combined to determine routing:
- static: Fine-tuned model can answer directly (no tools needed)
- dynamic: Requires live MCP data (real-time status, user-specific, events)
- combined: Needs both model knowledge and live data

Uses an LLM for robust natural language understanding.
"""

import logging

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ...config import settings
from ...telemetry import get_tracer
from ..state import AgentState, QueryClassification

logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM_PROMPT = """You are a query classifier for the ACCESS-CI documentation system.

Classify user queries into one of three categories:

**static** - Questions about factual, stable information that a trained model would know:
- Hardware specifications (GPUs, CPUs, memory, nodes)
- Resource descriptions and capabilities
- How-to guides and documentation
- Software availability and versions
- Policies and procedures
- Comparisons between resources
- Follow-up questions asking for more details about previously discussed topics

**dynamic** - Questions requiring live/real-time data from external systems:
- Current system status or outages
- User-specific data (my allocations, my usage, my projects)
- Upcoming events, workshops, or announcements
- Real-time metrics or statistics (XDMoD data)
- Current availability or queue status

**combined** - Questions needing both static knowledge AND live data:
- "Which resources with A100 GPUs are currently available?"
- "What's the status of Delta and what are its specs?"

You will be given conversation history for context. Use it to rewrite the current query as a standalone question by resolving any pronouns or references (e.g., "it", "that", "this one") to their actual referents from the conversation. If the query is already standalone, use it as-is.

Respond with ONLY a JSON object (no markdown):
{"query_type": "static|dynamic|combined", "reason": "brief explanation", "confidence": "high|medium|low", "expanded_query": "the query as a standalone question"}"""


def _get_classifier_llm() -> ChatOpenAI:
    """Get a fast LLM for classification."""
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=SecretStr(settings.OPENAI_API_KEY),
            temperature=0,
            max_completion_tokens=150,
        )
    raise ValueError("OPENAI_API_KEY required for query classification")


def _format_conversation_history(messages: list[AnyMessage]) -> str:
    """Format conversation history for the classifier.

    Args:
        messages: List of previous messages (HumanMessage/AIMessage).

    Returns:
        Formatted conversation history string.
    """
    if not messages:
        return "(No previous conversation)"

    lines = []
    for msg in messages[-6:]:  # Last 6 messages (3 turns) for context
        role = "User" if msg.type == "human" else "Assistant"
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Truncate long messages
        if len(content) > 300:
            content = content[:300] + "..."
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


async def classify_query_with_llm(
    query: str, conversation_history: str = ""
) -> QueryClassification:
    """Classify a query using an LLM for robust understanding.

    Args:
        query: The user's question.
        conversation_history: Formatted previous conversation for context.

    Returns:
        QueryClassification with query_type, reason, confidence, and expanded_query.
    """
    import json

    llm = _get_classifier_llm()

    # Build the user message with conversation context
    if conversation_history and conversation_history != "(No previous conversation)":
        user_content = f"Conversation history:\n{conversation_history}\n\nCurrent query: {query}"
    else:
        user_content = query

    messages = [
        SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = await llm.ainvoke(messages)
        content = str(response.content).strip()

        # Parse JSON response
        # Handle potential markdown code blocks
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        result = json.loads(content)

        return QueryClassification(
            query_type=result.get("query_type", "combined"),
            reason=result.get("reason", ""),
            confidence=result.get("confidence", "medium"),
            expanded_query=result.get("expanded_query", query),
        )

    except Exception as e:
        logger.warning(f"LLM classification failed, defaulting to combined: {e}")
        return QueryClassification(
            query_type="combined",
            reason=f"Classification error: {e}",
            confidence="low",
            expanded_query=query,
        )


async def classify_node(state: AgentState) -> dict[str, QueryClassification]:
    """Classify the query to determine routing.

    Args:
        state: Current agent state with query and messages.

    Returns:
        State update with query_classification.
    """
    tracer = get_tracer("access-agent.nodes")
    query = state["query"]
    messages = state.get("messages", [])

    # Build conversation history from previous messages (exclude current query)
    # The current query is the last message, so we take all but the last
    previous_messages = messages[:-1] if len(messages) > 1 else []
    conversation_history = _format_conversation_history(previous_messages)

    with tracer.start_as_current_span(
        "agent.classify",
        attributes={
            "agent.node": "classify",
            "agent.query_length": len(query),
            "agent.has_history": len(previous_messages) > 0,
        },
    ) as span:
        classification = await classify_query_with_llm(query, conversation_history)

        # Add classification results to span
        span.set_attribute("agent.query_type", classification.query_type)
        span.set_attribute("agent.confidence", classification.confidence)
        span.set_attribute(
            "agent.reason", classification.reason[:100] if classification.reason else ""
        )
        span.set_attribute("agent.query_expanded", classification.expanded_query != query)

        logger.info(
            f"Query classified as {classification.query_type} "
            f"(confidence={classification.confidence}): {classification.reason}"
        )
        if classification.expanded_query != query:
            logger.info(f"Query expanded: '{query}' -> '{classification.expanded_query}'")

        return {"query_classification": classification}
