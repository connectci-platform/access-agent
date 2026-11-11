"""The signed-in user's ACCESS ID must not be read as their email (D8-3077).

ACCESS IDs end in @access-ci.org, so a bare "Current user: apasquale@access-ci.org"
led the assistant to tell a signed-in user it could see "your email address".
Both signed-in branches must name it as an ACCESS ID and say it isn't an email.
"""

import pytest

from src.agent.prompts.system_prompt import build_system_prompt


@pytest.fixture(params=[True, False], ids=["can_write", "read_only"])
def prompt(request):
    return build_system_prompt(acting_user="jsmith@access-ci.org", can_write=request.param)


def test_the_acting_user_is_named_as_an_access_id(prompt):
    assert "ACCESS ID: `jsmith@access-ci.org`" in prompt
    assert "Current user: `jsmith@access-ci.org`" not in prompt


def test_the_access_id_is_not_called_an_email(prompt):
    assert "not their email address" in prompt
    assert "ask for one" in prompt


def test_an_anonymous_prompt_names_no_access_id():
    prompt = build_system_prompt(acting_user=None)

    assert "ACCESS ID: `" not in prompt


def test_the_access_id_is_not_offered_for_tickets(prompt):
    # Ticket tools take the requester from the verified sign-in (D8-3069), so
    # nothing should invite the model to put the ACCESS ID on a ticket.
    assert "ticket-creation" not in prompt


def test_a_read_only_chat_may_still_file_a_ticket():
    prompt = build_system_prompt(acting_user="jsmith@access-ci.org", can_write=False)

    assert "can't make changes on their behalf, except filing a support ticket" in prompt
