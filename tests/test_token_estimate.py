"""The token estimate is a close upper bound on Qwen's real counts (D8-3055).

Real counts are vLLM's prompt_tokens for each sample, measured against the
deployment on 2026-11-09 (max_tokens=1, less a one-character baseline). The two
schema samples are the smallest and largest of the five servers measured.
"""

import json
import logging
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.agent import token_estimate
from src.agent.token_estimate import (
    cut_to_tokens,
    estimate_messages,
    estimate_text,
    estimate_tools,
)

FIXTURES = Path(__file__).parent / "fixtures" / "tool_results"

REAL_TOKENS = {
    "list_all_software_anvil.json": 19_584,
    "get_infrastructure_news.json": 1_047,
    "search_resources_gpu.json": 8_385,
    "get_chart_data_wait_time.json": 3_911,
    "system_prompt.json": 3_767,
    "tool_schemas_3004.json": 1_086,
    "tool_schemas_3005.json": 2_158,
}


def sample(name: str) -> str:
    data = json.loads((FIXTURES / name).read_text())
    return data["text"] if "text" in data else data["content"][0]["text"]


@pytest.mark.parametrize("name", sorted(REAL_TOKENS))
def test_the_estimate_is_a_close_upper_bound(name):
    real = REAL_TOKENS[name]

    estimate = estimate_text(sample(name))

    assert real <= estimate <= 1.3 * real


def test_every_digit_counts_as_a_token():
    assert estimate_text("2026") >= 4


def test_special_token_text_is_counted_as_text():
    assert estimate_text("<|endoftext|>") > 1


def test_without_the_encoding_the_per_character_rule_still_bounds_the_count(monkeypatch, caplog):
    token_estimate._encoding.cache_clear()
    token_estimate.estimate_text.cache_clear()

    def offline(name: str):
        raise OSError("no network")

    monkeypatch.setattr(token_estimate.tiktoken, "get_encoding", offline)
    try:
        with caplog.at_level(logging.WARNING):
            estimates = {name: estimate_text(sample(name)) for name in REAL_TOKENS}
            estimate_text("again")
    finally:
        token_estimate._encoding.cache_clear()
        token_estimate.estimate_text.cache_clear()

    assert all(estimates[name] >= real for name, real in REAL_TOKENS.items())
    assert caplog.text.count("cl100k_base couldn't be loaded") == 1


def test_messages_count_their_text_calls_and_framing():
    call = {"name": "search", "args": {"query": "gpu"}, "id": "c1"}
    messages = [HumanMessage("hello"), AIMessage("", tool_calls=[call])]

    assert estimate_messages(messages) > estimate_text("hello") + estimate_text("gpu")


def test_tool_schemas_are_counted_as_sent():
    schema = {"type": "function", "function": {"name": "search", "parameters": {}}}

    assert estimate_tools([schema]) == estimate_text(json.dumps(schema))
    assert estimate_tools(["a_tool"]) == estimate_text("a_tool")


def test_a_cut_text_and_its_note_fit_the_budget():
    text = sample("list_all_software_anvil.json")

    cut = cut_to_tokens(text, 5_000)

    assert estimate_text(cut) <= 5_000
    assert cut.endswith(f"of {len(text)} characters]")
    assert cut_to_tokens("short", 5_000) == "short"


def test_a_cut_text_is_close_to_the_longest_that_fits():
    text = sample("search_resources_gpu.json")

    cut = cut_to_tokens(text, 5_000)

    assert estimate_text(cut) > 4_000
