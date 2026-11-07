"""The system prompt tells the model today's date and how to report lost scope."""

from datetime import date

from src.agent.prompts import system_prompt
from src.agent.prompts.system_prompt import SYSTEM_IDENTITY, build_system_prompt


def test_prompt_states_todays_date(monkeypatch):
    monkeypatch.setattr(system_prompt, "_today", lambda: date(2026, 11, 7))

    prompt = build_system_prompt()

    assert "Today is Saturday, November 7, 2026 (2026-11-07)." in prompt


def test_date_follows_the_fixed_identity_text(monkeypatch):
    monkeypatch.setattr(system_prompt, "_today", lambda: date(2026, 11, 7))

    prompt = build_system_prompt()

    assert prompt.startswith(SYSTEM_IDENTITY)
    assert prompt.index("## Current date") < prompt.index("## Acting user")


def test_today_is_a_real_date():
    assert isinstance(system_prompt._today(), date)


def test_prompt_requires_saying_when_the_scope_changed():
    assert "broader or different scope" in build_system_prompt()


def test_signed_in_prompt_says_earlier_sign_in_requests_no_longer_apply():
    assert "no longer applies" in build_system_prompt(acting_user="u@access-ci.org", can_write=True)
    assert "no longer applies" in build_system_prompt(acting_user="u@access-ci.org")


def test_anonymous_prompt_does_not_say_sign_in_is_done():
    assert "no longer applies" not in build_system_prompt()


def test_prompt_sets_out_the_registration_steps():
    prompt = build_system_prompt()

    assert "## Event registration" in prompt
    assert "without `confirmed`" in prompt
    assert "confirmation email" in prompt


def test_prompt_forbids_guessing_the_weekday():
    assert "weekday" in build_system_prompt()
