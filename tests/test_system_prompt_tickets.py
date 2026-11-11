"""A signed-in user's ticket is filed under their account (D8-3069).

The ticket server takes a signed-in user's name and email from the verified
token, so the workflow must not ask them for either, and should say which
email the ticket went to.
"""

from src.agent.prompts.system_prompt import build_system_prompt


def test_a_signed_in_user_is_not_asked_for_contact_details():
    prompt = build_system_prompt(acting_user="jsmith@access-ci.org", can_write=True)

    assert "the ticket uses their ACCESS account's name and email" in prompt
    assert "Their name and email (required for all tickets)" not in prompt
    assert "gather contact info" not in prompt


def test_a_signed_out_user_can_still_file_and_is_asked_for_an_email():
    # The identity gate lets anyone call the ticket tools, and a login ticket
    # is what someone who can't sign in needs most.
    prompt = build_system_prompt(acting_user=None)

    assert "anyone may still file a ticket" in prompt
    assert "ask for their email, and their name if they'll give it" in prompt
    assert "note these weren't verified" in prompt


def test_the_preview_never_shows_the_access_id_as_an_email():
    prompt = build_system_prompt(acting_user="jsmith@access-ci.org", can_write=True)

    assert "never show their ACCESS ID as an email" in prompt


def test_ticket_workflow_reports_the_email_it_was_filed_under():
    prompt = build_system_prompt(acting_user="jsmith@access-ci.org", can_write=True)

    assert "`filed_as`" in prompt
