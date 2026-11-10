"""The system prompt must carry the D8-3057 answering rules.

The assistant said XDMoD didn't track something it serves, misstated a sum in
prose next to a correct table, wrote its answer before a tool call and again
after, answered from excerpts about other systems, and looped over whether
documentation search had to come before a resource tool. These pin stable
phrases for each rule in both prompt branches.
"""

import pytest

from src.agent.prompts.system_prompt import build_system_prompt


@pytest.fixture(params=[None, "jsmith@access-ci.org"])
def prompt(request):
    return build_system_prompt(acting_user=request.param)


def test_doc_search_comes_first_when_another_tool_also_fits(prompt):
    assert "without calling `search_access_documents` first" in prompt
    assert "That settles the order when another tool also fits" in prompt
    assert "search first, then call the other." in prompt


def test_a_tool_is_tried_before_saying_it_cannot_answer(prompt):
    assert "Try a tool before saying it can't answer" in prompt
    assert "until you've called it for that" in prompt
    assert "covers how" in prompt


def test_search_again_when_excerpts_cover_other_systems(prompt):
    assert "If every excerpt is about a system other than the one the" in prompt
    assert "search again with a query that names it" in prompt


def test_text_before_a_tool_call_is_not_a_draft_answer(prompt):
    assert "The user sees any text you write before a tool call" in prompt
    assert "don't draft the answer there, or it appears twice" in prompt


def test_prose_arithmetic_is_checked_against_the_numbers(prompt):
    assert "Check any sum, difference, ratio or percentage you state in prose" in prompt
