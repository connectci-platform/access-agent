"""A conservative estimate of how many tokens Qwen counts, and cutting text to one.

Every digit and every character that isn't a letter or digit is counted as a
token, and letters at four to a token. Measured against the deployment's
tokenizer (prompt_tokens from UKY vLLM, 2026-11-09), it is an upper bound on
both kinds of tool result the agent sees:

- list_all_software for Anvil, 74,376 chars of JSON: 19,584 real, 33,832 here.
- get_chart_data, monthly wait times, 5,068 chars of text with dates and
  numbers: 3,911 real, 4,297 here.

``count_tokens_approximately`` (four characters a token) counted that chart
text as 1,267, a third of the real figure, since Qwen gives each digit its own
token. Over-counting JSON costs some room; under-counting overflows the window.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.messages import BaseMessage

_LETTERS = re.compile(r"[^\W\d_]+")
# Role and framing tokens each message carries in the chat template.
_PER_MESSAGE = 8


def estimate_text(text: str) -> int:
    """Tokens ``text`` is at most expected to cost."""
    letters = len(text) - len(_LETTERS.sub("", text))
    return len(text) - letters + -(-letters // 4)


def estimate_messages(messages: Iterable[BaseMessage]) -> int:
    """Tokens these messages are at most expected to cost, calls included."""
    total = 0
    for message in messages:
        total += _PER_MESSAGE + estimate_text(message.text)
        for call in getattr(message, "tool_calls", None) or []:
            total += estimate_text(json.dumps([call["name"], call["args"]], default=str))
    return total


def estimate_tools(tools: Iterable[Any]) -> int:
    """Tokens the schemas of these tools are at most expected to cost."""
    total = 0
    for tool in tools:
        if isinstance(tool, BaseTool | dict):
            total += estimate_text(json.dumps(convert_to_openai_tool(tool)))
        else:
            total += estimate_text(str(tool))
    return total


def prefix_within(text: str, tokens: int) -> str:
    """The longest start of ``text`` whose estimate is at most ``tokens``."""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if estimate_text(text[:middle]) <= tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low]


def truncated_note(kept: int, total: int) -> str:
    return f"\n\n[result truncated: {kept} of {total} characters]"


def cut_to_tokens(text: str, tokens: int) -> str:
    """``text`` cut so it and its truncation note cost at most ``tokens``, or
    ``text`` if it fits. Below the note's own cost, only the note is left."""
    if estimate_text(text) <= tokens:
        return text
    room = tokens - estimate_text(truncated_note(len(text), len(text)))
    kept = prefix_within(text, max(room, 0))
    return f"{kept}{truncated_note(len(kept), len(text))}"
