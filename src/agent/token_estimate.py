"""A close upper bound on how many tokens Qwen counts, and cutting text to one.

The estimate is 1.25 times cl100k_base's count with digits counted one a token,
as Qwen counts them. Measured against the deployment's tokenizer
(prompt_tokens from UKY vLLM, max_tokens=1, 2026-11-09), it is an upper bound
on every sample, 4-25% over:

    sample                          real    estimate / real
    list_all_software, Anvil        19,584  1.23
    get_infrastructure_news          1,047  1.14
    search_resources, GPU            8,385  1.22
    tool schemas, 5 servers   1,086-2,158  1.25
    get_chart_data, wait time        3,911  1.04
    system prompt                    3,767  1.24

Per-character rules did worse. Four characters a token
(``count_tokens_approximately``) counted the chart text at a third of its real
cost, since Qwen gives each digit its own token, and digits plus punctuation
plus letters/4, an upper bound, over-counted everything else 1.7-2.1 times.
That rule is kept as the fallback for when the cl100k encoding can't load,
since tiktoken fetches it on first use (the image caches it at build time).
"""

from __future__ import annotations

import functools
import json
import logging
import math
import re
from typing import TYPE_CHECKING, Any

import tiktoken
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

_DIGITS = re.compile(r"\d+")
_LETTERS = re.compile(r"[^\W\d_]+")
_MARGIN = 1.25
_CHUNK = 1000
# Role and framing tokens each message carries in the chat template.
_PER_MESSAGE = 8


@functools.cache
def _encoding() -> tiktoken.Encoding | None:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.warning(
            "cl100k_base couldn't be loaded, so token counts use the per-character "
            "fallback, which over-counts most text about twice",
            exc_info=True,
        )
        return None


def _by_character(text: str) -> int:
    """Digits and punctuation a token each, letters four to a token."""
    letters = len(text) - len(_LETTERS.sub("", text))
    return len(text) - letters + -(-letters // 4)


@functools.lru_cache(maxsize=256)
def estimate_text(text: str) -> int:
    """Tokens ``text`` is at most expected to cost.

    Cached, since every model call and every compaction check counts the same
    tool results again.
    """
    encoding = _encoding()
    if encoding is None:
        return _by_character(text)
    without_digits = _DIGITS.sub("", text)
    digits = len(text) - len(without_digits)
    # In chunks, since BPE on one long run with no spaces is quadratic. A cut can
    # only split a token, so the count can only rise.
    counted = sum(
        len(encoding.encode_ordinary(without_digits[start : start + _CHUNK]))
        for start in range(0, len(without_digits), _CHUNK)
    )
    return math.ceil(_MARGIN * (counted + digits))


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
    """A start of ``text`` whose estimate is at most ``tokens``, close to the longest.

    Sized in proportion to the whole text's estimate, then shortened by a tenth
    until it fits, which usually takes one or two counts.
    """
    length = len(text) * max(tokens, 0) // max(estimate_text(text), 1)
    while length > 0 and estimate_text(text[:length]) > tokens:
        length = length * 9 // 10
    return text[:length]


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
