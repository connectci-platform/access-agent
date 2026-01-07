"""Query classification node.

Classifies queries as static, dynamic, or combined to determine routing:
- static: Fine-tuned model can answer directly (no tools needed)
- dynamic: Requires live MCP data (real-time status, user-specific, events)
- combined: Needs both model knowledge and live data
"""

import logging
import re

from ..state import AgentState, QueryClassification

logger = logging.getLogger(__name__)

# Patterns that indicate dynamic queries (need live MCP data)
DYNAMIC_PATTERNS = [
    # Time-sensitive
    r"\b(currently|right now|today|this week|this month)\b",
    r"\b(happening|scheduled|upcoming|next)\b",
    # Status/outages
    r"\b(status|outage|down|maintenance|available now|working)\b",
    r"\b(is .+ (down|up|running|operational))\b",
    # User-specific (various phrasings for allocation/balance queries)
    r"\b(my|mine)\b.*(project|allocation|balance|account|usage)",
    r"\b(how (much|many) .* do I have)\b",
    r"\bhow much .* (left|remaining)\b",
    r"\b(do I have|I have left)\b",
    # Events and announcements
    r"\b(event|workshop|training|webinar|announcement)\b",
    # Metrics and usage
    r"\b(usage|metrics|statistics|utilization)\b",
    r"\bxdmod\b",
]

# Patterns that indicate static queries (model can answer directly)
STATIC_PATTERNS = [
    # Factual questions about resources
    r"\b(what is|what are|describe|explain|tell me about)\b",
    r"\b(what .* does .* have)\b",  # "What GPUs does Delta have?"
    r"\b(does .* have|does .* support)\b",  # "Does Anvil have A100s?"
    r"\b(specifications|specs|hardware|gpu|cpu|memory|nodes)\b",
    r"\b(how (do|does|to)|guide|tutorial)\b",
    # Software availability (general, not real-time)
    r"\b(software|module|package|version|available on)\b",
    # Documentation questions
    r"\b(documentation|docs|policy|policies|how do i)\b",
    # Comparisons
    r"\b(compare|comparison|difference|versus|vs\.?)\b",
    # Resource discovery
    r"\b(which (resource|system|cluster)|what (resources|systems))\b",
]

# Combined patterns - static info + dynamic status
COMBINED_PATTERNS = [
    r"\b(available|working)\b.*\b(gpu|resource|system)\b",
    r"\b(resource|system)\b.*\b(available|working)\b",
]


def classify_query(query: str) -> QueryClassification:
    """Classify a query using rule-based pattern matching.

    Args:
        query: The user's question.

    Returns:
        QueryClassification with query_type, reason, and confidence.
    """
    query_lower = query.lower()

    # Check for combined patterns first (most specific)
    for pattern in COMBINED_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryClassification(
                query_type="combined",
                reason="Query asks about resources with availability/status",
                confidence="medium",
            )

    # Check for dynamic patterns
    for pattern in DYNAMIC_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryClassification(
                query_type="dynamic",
                reason="Query contains time-sensitive or real-time indicator",
                confidence="high",
            )

    # Check for static patterns
    for pattern in STATIC_PATTERNS:
        if re.search(pattern, query_lower):
            return QueryClassification(
                query_type="static",
                reason="Query asks for factual/documentation information",
                confidence="high",
            )

    # Default to combined (safest fallback - will check both sources)
    return QueryClassification(
        query_type="combined",
        reason="Query type unclear, using combined approach",
        confidence="low",
    )


async def classify_node(state: AgentState) -> dict[str, QueryClassification]:
    """Classify the query to determine routing.

    Args:
        state: Current agent state with query.

    Returns:
        State update with query_classification.
    """
    query = state["query"]

    classification = classify_query(query)

    logger.info(
        f"Query classified as {classification.query_type} "
        f"(confidence={classification.confidence}): {classification.reason}"
    )

    return {"query_classification": classification}
