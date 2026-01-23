"""Query classification node.

Classifies queries as static, dynamic, or combined to determine routing:
- static: Fine-tuned model can answer directly (no tools needed)
- dynamic: Requires live MCP data (real-time status, user-specific, events)
- combined: Needs both model knowledge and live data

Uses an LLM for robust natural language understanding.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage
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

Respond with ONLY a JSON object (no markdown):
{"query_type": "static|dynamic|combined", "reason": "brief explanation", "confidence": "high|medium|low"}"""


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


async def classify_query_with_llm(query: str) -> QueryClassification:
    """Classify a query using an LLM for robust understanding.

    Args:
        query: The user's question.

    Returns:
        QueryClassification with query_type, reason, and confidence.
    """
    import json

    llm = _get_classifier_llm()

    messages = [
        SystemMessage(content=CLASSIFICATION_SYSTEM_PROMPT),
        HumanMessage(content=query),
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
        )

    except Exception as e:
        logger.warning(f"LLM classification failed, defaulting to combined: {e}")
        return QueryClassification(
            query_type="combined",
            reason=f"Classification error: {e}",
            confidence="low",
        )


async def classify_node(state: AgentState) -> dict[str, QueryClassification]:
    """Classify the query to determine routing.

    Args:
        state: Current agent state with query.

    Returns:
        State update with query_classification.
    """
    tracer = get_tracer("access-agent.nodes")
    query = state["query"]

    with tracer.start_as_current_span(
        "agent.classify",
        attributes={
            "agent.node": "classify",
            "agent.query_length": len(query),
        },
    ) as span:
        classification = await classify_query_with_llm(query)

        # Add classification results to span
        span.set_attribute("agent.query_type", classification.query_type)
        span.set_attribute("agent.confidence", classification.confidence)
        span.set_attribute(
            "agent.reason", classification.reason[:100] if classification.reason else ""
        )

        logger.info(
            f"Query classified as {classification.query_type} "
            f"(confidence={classification.confidence}): {classification.reason}"
        )

        return {"query_classification": classification}
