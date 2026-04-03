"""Query classification node.

Classifies queries as static, dynamic, or combined to determine routing:
- static: Fine-tuned model can answer directly (no tools needed)
- dynamic: Requires live MCP data (real-time status, user-specific, events)
- combined: Needs both model knowledge and live data

Uses an LLM for robust natural language understanding.
"""

import logging
from typing import Any, Literal, cast

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from ...config import settings
from ...telemetry import get_tracer
from ..state import AgentState, QueryClassification

logger = logging.getLogger(__name__)

CLASSIFICATION_SYSTEM_PROMPT = """You are a query classifier for the ACCESS-CI documentation system.

Classify user queries into one of three categories:

**static** - Questions about stable information found only in documentation:
- How-to guides, tutorials, and procedures (how to log in, submit jobs, use Globus)
- Policies and rules (allocation policies, password requirements, SU calculations)
- General explanations (what is ACCESS, what is Kerberos, how do SUs work)
- Follow-up questions asking for more details about previously discussed topics

**dynamic** - Questions requiring ONLY live/real-time data:
- Current system status or outages ("is Delta down right now?")
- User-specific data (my allocations, my usage, my projects)
- Current availability or queue status

**combined** - Questions where documentation AND live data together give the best answer. USE THIS LIBERALLY — when in doubt between static and combined, prefer combined:
- Hardware specs (GPUs, CPUs, memory, storage) — docs may be stale, live data is current
- Software availability and versions — changes frequently as modules are added/updated
- Resource descriptions and capabilities — docs provide context, live data provides current specs
- Comparisons between resources — need current data from multiple sources
- Upcoming events, workshops, or announcements — need both descriptions and current schedules
- Any question mentioning specific resource names (Delta, Bridges-2, Expanse, Anvil, etc.) paired with specs, hardware, software, or storage

Also determine which RAG endpoint should answer the question. Set "rag_endpoint" to:
- "general" — ACCESS documentation: allocations, resources, how-tos, policies, hardware specs
- "xdmod" — XDMoD features/capabilities, what metrics are available, how to interpret data, links to XDMoD charts and visualizations
- null — purely dynamic queries where only live MCP tool data can answer (user-specific data, real-time status)

XDMoD routing guidance:
- Any question about aggregate numbers, counts, totals, trends, or utilization across ACCESS resources should route to XDMoD (rag_endpoint: "xdmod", query_type: "combined"). This includes questions about: job counts, CPU hours, GPU utilization, allocations, user accounts, gateways, projects, proposals, storage, and resource capacity. XDMoD has data realms for: Jobs, SUPREMM, Cloud, Gateways, Allocations, Accounts, Requests, ResourceSpecifications, Storage.
- Most XDMoD questions benefit from BOTH the RAG answer AND MCP tool data, so prefer query_type "combined".
- Only use query_type "dynamic" with rag_endpoint null for purely user-specific XDMoD queries like "my usage".

**MCP-backed capabilities** — the following topics have live data available via MCP tools and should NEVER be classified as "static". Use "dynamic" (rag_endpoint: null) when only live data is needed, or "combined" when documentation context also helps:
- NSF awards — searchable via MCP tool. Always dynamic (rag_endpoint: null). No documentation covers specific awards.
- Affinity groups — browsable via MCP tool. Always dynamic (rag_endpoint: null).
- Events, workshops, office hours — fetched live via MCP tool. Always dynamic (rag_endpoint: null) for "what events are coming up", combined if asking about event policies or how to register.
- Announcements (reading/searching) — fetched live via MCP tool. Always dynamic (rag_endpoint: null).
- System status and outages — fetched live via MCP tool. Always dynamic (rag_endpoint: null).
- Software availability — fetched live via MCP tool. Use combined (rag_endpoint: "general") since docs add context.
- Allocation lookups — fetched live via MCP tool. Use dynamic (rag_endpoint: null) for "show my allocations", combined for "how do allocations work on Delta".

Examples:
- "How do I get an allocation?" → rag_endpoint: "general", query_type: "static", capability_id: "ask_question"
- "What are the password requirements?" → rag_endpoint: "general", query_type: "static", capability_id: "ask_question"
- "How do I use Globus to transfer files?" → rag_endpoint: "general", query_type: "static", capability_id: "ask_question"
- "What GPUs does Delta have?" → rag_endpoint: "general", query_type: "combined", capability_id: "ask_question"
- "What software is on Bridges-2?" → rag_endpoint: "general", query_type: "combined", capability_id: "search_software"
- "What storage options are on Anvil?" → rag_endpoint: "general", query_type: "combined", capability_id: "ask_question"
- "Which resources support A100 GPUs?" → rag_endpoint: "general", query_type: "combined", capability_id: "ask_question"
- "Show me CPU hours on Delta last month" → rag_endpoint: "xdmod", query_type: "combined", capability_id: "check_usage"
- "How many active allocations are there?" → rag_endpoint: "xdmod", query_type: "combined", capability_id: "check_usage"
- "What's my usage on Expanse?" → rag_endpoint: null, query_type: "dynamic", capability_id: "check_usage"
- "Is Delta down right now?" → rag_endpoint: null, query_type: "dynamic", capability_id: "check_system_status"
- "Search NSF awards" → rag_endpoint: null, query_type: "dynamic", capability_id: "search_nsf_awards"
- "Browse affinity groups" → rag_endpoint: null, query_type: "dynamic", capability_id: "browse_affinity_groups"
- "What events are coming up?" → rag_endpoint: null, query_type: "dynamic", capability_id: "browse_events"
- "Show recent announcements" → rag_endpoint: null, query_type: "dynamic", capability_id: "search_announcements"
- "What affinity groups are there?" → rag_endpoint: null, query_type: "dynamic", capability_id: "browse_affinity_groups"

Also detect if the query should be handled by a specialized domain agent. Set "domain" to:
- "announcements" — ONLY when the user explicitly asks to CREATE, UPDATE, DELETE, or MANAGE announcements (not just search/read them)
- "jsm" — ONLY when the user explicitly asks to CREATE or FILE a support ticket, using imperative language like "open a ticket", "file a ticket", "create a ticket", "submit a ticket"
- null — for EVERYTHING else, including:
  - Describing problems ("password not working", "can't login", "job failed") — these are troubleshooting questions, NOT ticket requests
  - Asking for help ("how do I fix", "what should I do about") — these want guidance, NOT a ticket
  - Reporting status ("my allocation shows pending", "I got an error") — these want explanations, NOT a ticket
  - General questions, searches, informational queries, reading announcements

IMPORTANT: A user describing a problem is NOT the same as requesting a ticket. Only set domain to "jsm" when the user uses explicit action language requesting ticket creation.

Domain examples:
- "Please open a support ticket about my login issue" → domain: "jsm" (explicit ticket request)
- "I'd like to file a ticket" → domain: "jsm" (explicit ticket request)
- "Password not working to ssh into Bridges-2" → domain: null (troubleshooting question, NOT a ticket request)
- "I can't find my allocation" → domain: null (informational question, NOT a ticket request)
- "My scratch files keep getting purged" → domain: null (troubleshooting question, NOT a ticket request)
- "Can you help me with my software codes?" → domain: null (help request, NOT a ticket request)
- "Create an announcement about the workshop" → domain: "announcements" (explicit create request)

Also identify which capability the query best matches. Set "capability_id" to one of:
- "ask_question" — general Q&A about ACCESS (default for most queries)
- "check_allocations" — allocation lookups and status
- "search_software" — software availability on resources
- "check_system_status" — outages and resource status
- "browse_events" — upcoming trainings, workshops, office hours
- "browse_affinity_groups" — community affinity groups
- "check_usage" — XDMoD usage and performance data
- "search_nsf_awards" — NSF award lookups
- "search_announcements" — reading/searching announcements
- "manage_announcements" — creating, updating, or deleting announcements (domain: "announcements")
- "open_ticket" — creating a support ticket (domain: "jsm")
- "report_login_problem" — reporting login issues (domain: "jsm")
- "report_security" — reporting security concerns (domain: "jsm")

When domain is set, capability_id should match (e.g., domain "jsm" + ticket request → "open_ticket"). When domain is null, pick the best-fit capability from the general list. Default to "ask_question" if unclear.

You will be given conversation history for context. Use it to rewrite the current query as a standalone question by resolving any pronouns or references (e.g., "it", "that", "this one") to their actual referents from the conversation. If the query is already standalone, use it as-is.

Respond with ONLY a JSON object (no markdown):
{"query_type": "static|dynamic|combined", "rag_endpoint": "general|xdmod|null", "reason": "brief explanation", "confidence": "high|medium|low", "expanded_query": "the query as a standalone question", "domain": "announcements|jsm|null", "capability_id": "ask_question|check_allocations|..."}"""


def _get_classifier_llm() -> ChatOpenAI:
    """Get a fast LLM for classification."""
    if settings.OPENAI_API_KEY:
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=SecretStr(settings.OPENAI_API_KEY),
            temperature=0,
            max_completion_tokens=350,
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

        # Parse domain — LLM may return null, "null", or missing
        raw_domain = result.get("domain")
        domain = raw_domain if isinstance(raw_domain, str) and raw_domain != "null" else None

        # Parse rag_endpoint — LLM may return null, "null", or missing
        raw_rag_endpoint = result.get("rag_endpoint")
        rag_endpoint: Literal["general", "xdmod"] | None = (
            cast("Literal['general', 'xdmod']", raw_rag_endpoint)
            if isinstance(raw_rag_endpoint, str) and raw_rag_endpoint in ("general", "xdmod")
            else None
        )

        # Parse capability_id — LLM may return null, "null", or missing
        raw_capability = result.get("capability_id")
        capability_id = (
            raw_capability if isinstance(raw_capability, str) and raw_capability != "null" else None
        )

        return QueryClassification(
            query_type=result.get("query_type", "combined"),
            reason=result.get("reason", ""),
            confidence=result.get("confidence", "medium"),
            expanded_query=result.get("expanded_query", query),
            domain=domain,
            rag_endpoint=rag_endpoint,
            capability_id=capability_id,
        )

    except Exception as e:
        logger.warning(f"LLM classification failed, defaulting to combined: {e}")
        return QueryClassification(
            query_type="combined",
            reason=f"Classification error: {e}",
            confidence="low",
            expanded_query=query,
        )


async def classify_node(state: AgentState) -> dict[str, Any]:
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

        if classification.domain:
            span.set_attribute("agent.domain", classification.domain)
        if classification.rag_endpoint:
            span.set_attribute("agent.rag_endpoint", classification.rag_endpoint)
        if classification.capability_id:
            span.set_attribute("agent.capability_id", classification.capability_id)

        logger.info(
            f"Query classified as {classification.query_type} "
            f"(confidence={classification.confidence}, domain={classification.domain}, "
            f"rag_endpoint={classification.rag_endpoint}, capability={classification.capability_id}): "
            f"{classification.reason}"
        )
        if classification.expanded_query != query:
            logger.info(f"Query expanded: '{query}' -> '{classification.expanded_query}'")

        return {
            "query_classification": classification,
            "node_trace": [
                {
                    "node": "classify",
                    "query_type": classification.query_type,
                    "confidence": classification.confidence,
                    "domain": classification.domain,
                    "rag_endpoint": classification.rag_endpoint,
                    "reason": classification.reason[:200] if classification.reason else "",
                    "expanded": classification.expanded_query != query,
                    "capability_id": classification.capability_id,
                }
            ],
        }
