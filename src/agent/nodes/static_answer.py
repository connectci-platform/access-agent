"""Static answer node.

Calls the fine-tuned model directly for static queries,
bypassing the tool-based agent workflow.
"""

import logging
import re
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage
from pydantic import SecretStr

from ...config import settings
from ..state import AgentState

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def process_citations(text: str) -> str:
    """Convert <<SRC:...>> markers to readable citations.

    The fine-tuned model outputs citations like:
        <<SRC:compute-resources:delta.ncsa.access-ci.org>>

    This converts them to a more readable format.
    Future: Could look up URLs from a registry.

    Args:
        text: Response text with citation markers.

    Returns:
        Text with processed citations.
    """
    # Pattern: <<SRC:domain:entity_id>>
    pattern = r"<<SRC:([^:>]+):([^>]+)>>"

    def replace_citation(match: re.Match[str]) -> str:
        domain = match.group(1)
        entity_id = match.group(2)
        # For now, just make it a readable reference
        # Future: look up actual URLs from a registry
        return f"[Source: {domain}/{entity_id}]"

    return re.sub(pattern, replace_citation, text)


def get_static_llm() -> "ChatOpenAI | None":
    """Get the LLM for static answers.

    Uses STATIC_LLM_PROVIDER if set, otherwise falls back to LLM_PROVIDER.
    Currently only Fireworks is supported for fine-tuned model answers.

    Returns:
        ChatOpenAI instance configured for static answers, or None if not available.
    """
    from langchain_openai import ChatOpenAI

    # Determine which provider to use for static answers
    provider = settings.STATIC_LLM_PROVIDER or settings.LLM_PROVIDER

    if provider == "fireworks":
        if not settings.FIREWORKS_API_KEY:
            return None
        return ChatOpenAI(
            model=settings.FIREWORKS_MODEL,
            api_key=SecretStr(settings.FIREWORKS_API_KEY),
            base_url="https://api.fireworks.ai/inference/v1",
            temperature=0.3,
            max_completion_tokens=1024,
        )

    # Other providers don't have fine-tuned models yet
    logger.warning(f"Static answers not supported for provider: {provider}")
    return None


async def static_answer_node(state: AgentState) -> dict[str, object]:
    """Generate answer directly from fine-tuned model.

    This node is used for static queries where the fine-tuned model
    already knows the answer without needing MCP tool calls.

    Args:
        state: Current agent state with query.

    Returns:
        State update with final_answer and messages.
    """
    query = state["query"]

    logger.info(f"Generating static answer for: {query[:50]}...")

    # Get the fine-tuned model
    llm = get_static_llm()
    classification = state["query_classification"]
    if llm is None or classification is None:
        logger.warning("Static LLM not available, falling back to dynamic path")
        if classification is not None:
            return {
                "query_classification": classification.model_copy(
                    update={
                        "query_type": "dynamic",
                        "reason": "Static LLM not configured, using tools",
                    }
                )
            }
        return {}

    # Build conversation context from messages
    messages = state.get("messages", [])

    try:
        # Call the model
        response = await llm.ainvoke(messages)
        content = response.content

        # Handle content types - should be string but LangChain types allow list
        answer = " ".join(str(c) for c in content) if isinstance(content, list) else str(content)

        # Process citations
        answer = process_citations(answer)

        logger.info(f"Static answer generated: {len(answer)} characters")

        # Add the response to conversation history
        return {
            "final_answer": answer,
            "messages": [AIMessage(content=answer)],
            "tools_used": [],  # No tools used for static answers
        }

    except Exception as e:
        logger.error(f"Error generating static answer: {e}")
        # Fall back to dynamic path on error
        return {
            "query_classification": classification.model_copy(
                update={"query_type": "dynamic", "reason": f"Static answer failed: {e}"}
            )
        }
