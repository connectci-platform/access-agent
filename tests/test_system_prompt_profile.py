"""Tests for the `## User profile` section in build_system_prompt.

Pattern of tests/test_system_prompt_urls.py: direct build_system_prompt
calls against the composed output.
"""

from src.agent.profile import AllocatedResource, UserProfile
from src.agent.prompts.system_prompt import build_system_prompt


def test_no_profile_section_when_profile_is_none():
    prompt = build_system_prompt(profile=None)
    assert "## User profile" not in prompt


def test_no_profile_section_when_profile_is_empty():
    prompt = build_system_prompt(profile=UserProfile())
    assert "## User profile" not in prompt


def test_profile_section_renders_composed_output():
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )
    prompt = build_system_prompt(profile=profile)
    assert "About this user:" in prompt
    assert "rp_name='delta'" in prompt


def test_profile_section_follows_resource_context_section():
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )
    prompt = build_system_prompt(resource_context="delta", profile=profile)
    resource_idx = prompt.index("## Resource context")
    profile_idx = prompt.index("## User profile")
    assert resource_idx < profile_idx


def test_resource_context_wins_precedence_wording():
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Bridges-2 RM", rp_slug="bridges2")]
    )
    prompt = build_system_prompt(resource_context="delta", profile=profile)
    assert "that resource is what" in prompt
    # The profile's own single-slug claim must not survive alongside the
    # resource_context override — otherwise the model sees two competing
    # answers for "this system" instead of the one the override provides.
    assert "they mean `bridges2`" not in prompt


def test_resource_context_overrides_profile_slug_claim():
    """resource_context="anvil" + a profile naming Delta GPU (slug delta): the
    profile section must not claim "this system" means delta — resource_context
    wins — but the fact line still lists Delta GPU as supplied."""
    profile = UserProfile(
        allocated_resources=[AllocatedResource(name="Delta GPU", rp_slug="delta")]
    )
    prompt = build_system_prompt(resource_context="anvil", profile=profile)

    assert "they mean `delta`" not in prompt
    assert 'that resource is what "this system" means' in prompt
    assert "Delta GPU" in prompt


def test_resource_context_with_empty_resources_has_no_dangling_override():
    """An explicitly empty allocated_resources list plus resource_context:
    the none-supplied fact still renders, but the override sentence — which
    would claim "the allocations listed here" with nothing listed — and the
    ordinary "ask which resource" instruction (moot; resource_context already
    answers it) must both be absent."""
    profile = UserProfile(allocated_resources=[])
    prompt = build_system_prompt(resource_context="anvil", profile=profile)

    assert "Allocated resources: none" in prompt
    assert "allocations listed here" not in prompt
    assert "ask which resource" not in prompt


# ── Not opted in (D8-3124) ───────────────────────────────────────────────

import pytest  # noqa: E402

from src.agent.prompts.system_prompt import (  # noqa: E402
    CONSENT_UNKNOWN_SECTION,
    NOT_PERSONALIZED_SECTION,
)

USER = "alice@access-ci.org"


@pytest.mark.parametrize("consent", ["deny", "unset"])
def test_a_user_who_hasnt_opted_in_gets_the_ask_first_rule(consent):
    prompt = build_system_prompt(acting_user=USER, can_write=True, consent=consent)

    assert NOT_PERSONALIZED_SECTION in prompt
    assert "only when the user asks about their own allocations" in prompt
    assert "turn on personalized answers in the assistant's Settings" in prompt


def test_an_allowed_user_whose_lookup_failed_isnt_told_to_turn_it_on():
    prompt = build_system_prompt(acting_user=USER, can_write=True, consent="allow")

    assert "## Personalized answers" not in prompt


def test_an_unreadable_choice_holds_back_without_saying_turn_it_on():
    prompt = build_system_prompt(acting_user=USER, can_write=True, consent=None)

    assert CONSENT_UNKNOWN_SECTION in prompt
    assert "turn on personalized answers" not in prompt


@pytest.mark.parametrize("consent", ["allow", "deny", "unset", None])
def test_no_rule_with_a_profile_or_when_anonymous(consent):
    profile = UserProfile(allocated_resources=[AllocatedResource(name="Delta GPU")])
    assert "## Personalized answers" not in build_system_prompt(
        acting_user=USER, can_write=True, profile=profile, consent=consent
    )
    assert "## Personalized answers" not in build_system_prompt(acting_user=None, consent=consent)


def test_the_loop_passes_consent_from_state_to_the_prompt():
    from src.agent.nodes.tool_calling_loop import _build_prompt_and_tools

    prompt, _, used = _build_prompt_and_tools(
        tool_catalog={},
        acting_user=USER,
        can_write=True,
        resource_context=None,
        profile=None,
        consent="deny",
    )
    assert NOT_PERSONALIZED_SECTION in prompt
    assert used is False


def test_initial_state_carries_consent():
    from src.agent.state import create_initial_state

    state = create_initial_state("q", "s", "q1", {}, consent="unset")
    assert state["consent"] == "unset"
