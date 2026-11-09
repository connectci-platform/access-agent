"""The system prompt must carry the D8-3055 grounding rules.

QA caught the assistant pricing a Jetstream2 flavor that doesn't exist, giving
password-reset advice for Bridges-2 during its outage, searching one resource's
docs for an ACCESS account question, and saying XDMoD timed out when it hadn't.
These pin stable phrases for each rule in both prompt branches.
"""

import pytest

from src.agent.prompts.system_prompt import build_system_prompt


@pytest.fixture(params=[None, "jsmith@access-ci.org"])
def prompt(request):
    return build_system_prompt(acting_user=request.param)


def test_undocumented_configurations_are_not_described(prompt):
    assert "Never price, size or describe a configuration" in prompt
    assert "Derive nothing by arithmetic" in prompt
    assert "If the exact thing isn't documented, say so and link the page" in prompt


def test_status_is_checked_before_login_advice(prompt):
    assert "can't connect, ssh or submit questions" in prompt
    assert "first call `get_infrastructure_news` with `resource`" in prompt
    assert "lead with the outage" in prompt


def test_account_questions_are_searched_access_wide(prompt):
    assert "Account and allocation questions are ACCESS-wide" in prompt
    assert "without `rp_name`" in prompt


def test_gateway_allocations_use_the_gateway_portal(prompt):
    assert "NSG (Neuroscience Gateway) or CIPRES" in prompt
    assert "ask which gateway they mean" in prompt


def test_failures_are_checked_against_results_and_lists_are_not_padded(prompt):
    assert "Before saying a lookup" in prompt
    assert "failed or timed out, check the tool results you have" in prompt
    assert "Never pad a list with placeholder entries" in prompt
