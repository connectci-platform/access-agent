"""A signed-in user's ticket is filed under their account (D8-3069).

The ticket server takes a signed-in user's name and email from the verified
token, so the workflow must not ask them for either, and should say which
email the ticket went to.
"""

from src.agent.prompts.system_prompt import build_system_prompt


def test_ticket_workflow_never_asks_for_name_and_email():
    prompt = build_system_prompt(acting_user="jsmith@access-ci.org", can_write=True)

    assert "filed under their ACCESS account's name and email" in prompt
    assert "don't ask; the ticket uses their account's" in prompt
    assert "Their name and email (required for all tickets)" not in prompt


def test_ticket_workflow_sends_signed_out_users_to_sign_in_or_the_form():
    # The agent refuses ticket tools for anonymous callers, so collecting their
    # details would go nowhere.
    prompt = build_system_prompt(acting_user=None)

    assert "You can't file a ticket for someone who" in prompt
    assert "don't collect their name" in prompt


def test_ticket_workflow_reports_the_email_it_was_filed_under():
    prompt = build_system_prompt(acting_user="jsmith@access-ci.org", can_write=True)

    assert "`filed_as`" in prompt
