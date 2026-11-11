"""A signed-in user's ticket is filed under their account (D8-3069).

The ticket server takes a signed-in user's name and email from the verified
token, so the workflow must not ask them for either, and should say which
email the ticket went to.
"""

from src.agent.prompts.system_prompt import build_system_prompt


def test_ticket_workflow_asks_for_name_and_email_only_when_signed_out():
    prompt = build_system_prompt(acting_user="jsmith@access-ci.org", can_write=True)

    assert "only if they aren't signed in" in prompt
    assert "filed under their ACCESS account's name and email" in prompt
    assert "Their name and email (required for all tickets)" not in prompt


def test_ticket_workflow_reports_the_email_it_was_filed_under():
    prompt = build_system_prompt(acting_user="jsmith@access-ci.org", can_write=True)

    assert "`filed_as`" in prompt
