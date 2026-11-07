"""Which streamed message chunks are the user's answer text."""

from typing import Any

from langchain_core.messages import AIMessageChunk

from ..llm import is_empty_answer

# The model node inside create_agent, which tool_calling_loop runs as a nested graph.
ANSWER_MODEL_NODE = "model"


def is_answer_token(msg: Any, metadata: dict[str, Any]) -> bool:
    """Whether a LangGraph message chunk is answer text that should reach the browser.

    * AIMessageChunk only, since the messages stream also carries complete
      AIMessage objects added to state, and streaming those duplicates the answer.
    * ANSWER_MODEL_NODE only, so summarization and other nodes' chatter is excluded.
    * Non-empty, since there is nothing to render otherwise.
    * Not the empty-answer sentinel, which is wire-protocol only. qa-bot-core
      concatenates every token event, so streaming it would paste the marker into
      the visible answer. tool_calling_loop substitutes real prose into final_answer,
      which the done event carries.
    """
    return (
        isinstance(msg, AIMessageChunk)
        and metadata.get("langgraph_node") == ANSWER_MODEL_NODE
        and bool(msg.content)
        and not is_empty_answer(msg.content if isinstance(msg.content, str) else None)
    )
