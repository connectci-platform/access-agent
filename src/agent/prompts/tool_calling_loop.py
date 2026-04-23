"""System prompt assembly for the tool_calling_loop node (launch Phase 3).

The loop's system prompt weaves together:
  - the agent's general identity and behavioral instructions
  - optional RAG context (when rag_answer produced matches)
  - optional domain hint (when classifier identified a specific domain)
  - optional acting-user identity (when the request is authenticated)

Kept separate from the node logic so prompt iteration is a targeted PR that
doesn't invalidate review of the orchestration code.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..state import RAGMatch

SYSTEM_IDENTITY = """You are the ACCESS-CI assistant. You help US researchers \
understand and use ACCESS-CI — a federally funded program that allocates \
computing resources (supercomputers, cloud, storage) to researchers.

Your job in this conversation: answer the user's question using the tools \
provided, along with any reference context below. Call tools when you need \
live data. Do not invent data you can't verify. If a tool fails or returns \
unexpected output, try a different approach or tell the user what you tried \
and what failed — do not silently drop the failure.

When you produce your final answer:
- Cite specific resources or facts you retrieved. Link to official ACCESS-CI \
pages where relevant.
- Be complete. Include the specific details researchers need to act — commands, links, numeric values, step-by-step instructions where relevant. Don't pad with ceremony, but don't strip substance either.
- If the answer depends on the user's specific situation (allocations, \
account state), say so clearly and explain how they can check.
- If you genuinely cannot answer, say that and point the user to the support \
ticket path at https://support.access-ci.org/open-a-ticket."""


def build_system_prompt(
    rag_context: str | None = None,
    domain_hint: str | None = None,
    acting_user: str | None = None,
) -> str:
    """Assemble the loop's system prompt from optional context fragments.

    Args:
        rag_context: Formatted text block of RAG matches from rag_answer_node.
            If present, inserted as a "Reference context" section.
        domain_hint: Classifier's domain guess (e.g. "jsm", "announcements").
            If present, inserted as a "Likely domain" hint the LLM can override.
        acting_user: ACCESS ID of the authenticated requester. If present,
            inserted so the LLM can personalize allocation/usage queries.

    Returns:
        Complete system prompt string. Always begins with SYSTEM_IDENTITY.
    """
    sections: list[str] = [SYSTEM_IDENTITY]

    if rag_context:
        sections.append(
            "## Reference context (from retrieval)\n\n"
            "The following is background material retrieved for this query. "
            "Use it as reference; you still need to call tools for live data.\n\n"
            f"{rag_context}"
        )

    if domain_hint:
        sections.append(
            f"## Classifier hint\n\n"
            f"The classifier identified this query's domain as `{domain_hint}`. "
            f"You may override if the user's intent suggests otherwise."
        )

    if acting_user:
        sections.append(
            f"## Acting user\n\nCurrent user: `{acting_user}`. "
            f"Use this when calling tools that accept an acting-user identity "
            f"(allocations, ticket-creation, personal usage queries)."
        )
    else:
        sections.append(
            "## Acting user\n\n"
            "The user is anonymous (not logged in). Tools that require identity "
            "will return auth errors — handle those gracefully and suggest login "
            "at https://access-ci.org/sign-in."
        )

    return "\n\n".join(sections)


def format_rag_matches(matches: list[RAGMatch]) -> str:
    """Render a list of RAGMatch objects as a prompt-ready text block.

    Args:
        matches: List of RAGMatch instances from src.agent.state.

    Returns:
        Markdown-formatted text block, ready to pass to build_system_prompt's
        rag_context argument. Empty string when matches is empty.
    """
    if not matches:
        return ""

    rendered: list[str] = []
    for i, match in enumerate(matches, start=1):
        block = f"### Match {i} (score: {match.similarity_score:.2f})"
        block += f"\n\n**Q:** {match.question}\n\n**A:** {match.answer}"
        block += f"\n\n*Source:* {match.domain}/{match.entity_id}"
        rendered.append(block)

    return "\n\n".join(rendered)
