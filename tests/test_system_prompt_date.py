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


def test_a_caller_who_can_write_is_told_earlier_sign_in_requests_no_longer_apply():
    assert "no longer applies" in build_system_prompt(acting_user="u@access-ci.org", can_write=True)
    assert "no longer applies" not in build_system_prompt(acting_user="u@access-ci.org")


def test_anonymous_prompt_does_not_say_sign_in_is_done():
    assert "no longer applies" not in build_system_prompt()


def test_prompt_sets_out_the_registration_steps():
    prompt = build_system_prompt()

    assert "## Event registration" in prompt
    assert "call `register_for_event`" in prompt
    assert "waiting for the organizer's approval" in prompt
    assert "Never say they are registered" in prompt
    # The model once looked an event up with get_event and wrote its own
    # preview, so the client had no preview result to show a confirm button for.
    assert "your next call is that preview" in prompt


def test_prompt_states_one_confirm_rule_for_every_write():
    for kwargs in ({}, {"acting_user": "u@access-ci.org", "can_write": True}):
        prompt = build_system_prompt(**kwargs)

        assert "## Making changes" in prompt
        assert "Never pass `confirmed`" in prompt
        assert 'When their message is "Confirm", the change has already been made' in prompt
        assert 'When it is "Cancel", say nothing was changed' in prompt
        assert 'A typed "yes" or "go ahead" is not a confirmation' in prompt
        assert "call the tool again with the change to get a new preview" in prompt
    # Tickets are writes, so a caller who can't otherwise write still gets it.
    assert "## Making changes" in build_system_prompt(acting_user="u@access-ci.org")


def test_prompt_takes_the_event_series_id_from_get_event():
    prompt = build_system_prompt()

    assert "comes from the `series_id` returned by `get_event`" in prompt
    assert "Never take it from an event page URL or an eventinstance id" in prompt


def test_prompt_never_asks_for_a_typed_confirmation_or_confirmed_true():
    prompt = build_system_prompt(acting_user="u@access-ci.org", can_write=True)

    assert "confirmed: true" not in prompt
    assert "after they confirm" not in prompt
    assert "only after the user confirms" not in prompt
    assert "Confirm the specific announcement with the user" not in prompt
    assert "Ask which one only if it's ambiguous, then call `delete_announcement`" in prompt


def test_prompt_forbids_guessing_the_weekday():
    assert "weekday" in build_system_prompt()


def test_a_caller_who_cannot_write_is_not_told_how_to_register():
    assert "## Event registration" not in build_system_prompt(acting_user="u@access-ci.org")


def test_prompt_says_to_quote_the_local_event_times():
    prompt = build_system_prompt()

    assert "`start_local`" in prompt
    assert "never convert a time yourself" in prompt
    assert "time zone isn't known" in prompt


def test_prompt_names_the_users_time_zone():
    prompt = build_system_prompt(time_zone="America/New_York")

    assert "The user's time zone is America/New_York" in prompt
