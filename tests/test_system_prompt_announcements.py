"""Tests for the announcements create workflow in build_system_prompt.

Pattern of tests/test_system_prompt_urls.py: direct build_system_prompt
calls against the composed output.
"""

from src.agent.prompts.system_prompt import (
    ANNOUNCEMENTS_WORKFLOWS_SECTION,
    build_system_prompt,
)


def _normalize(text: str) -> str:
    return " ".join(text.replace("\\\n", " ").split())


def test_workflow_always_shows_affiliation_and_sharing_in_preview():
    section = _normalize(build_system_prompt())
    assert "always including tags, summary, affiliation and where it will be shared" in section
    assert "Community" in section
    assert "Announcements page and Bi-Weekly Digest" in section


def test_workflow_does_not_ask_coordinators_about_affiliation_or_sharing():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    assert "also ask about affinity group and where to share" not in section
    assert "affiliation if unclear" not in section
    assert "Don't ask separate questions about affiliation, sharing or affinity group" in section


def test_update_workflow_shows_full_replacement_set():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    assert "replace the whole set" in section


def test_missing_fields_are_asked_before_the_preview():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    ask = section.index("Ask only about a missing or unclear title, body or summary")
    preview = section.index("Present a preview")
    assert ask < preview


def test_change_request_triggers_re_preview_not_creation():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    assert "If the user asks for a change without saying to go ahead" in section
    assert "show the updated preview and ask again" in section
    assert "A bare change request is not confirmation" in section
    assert "requests a change and explicitly says to go ahead" in section
    assert "an explicit instruction to create in the same reply" in section
    assert '"go ahead", "create it" or "post it"' in section
    assert 'Approval words like "looks good" or "sounds good"' in section
    assert "get an updated preview" in section
    assert "only after the user confirms the current preview" in section
    assert "call `create_announcement` with that change applied" in section
    assert "tell the user what was saved" in section


def test_affinity_group_defaults_to_none_with_one_allowed_question():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    assert "affinity group defaults to None unless the user named one" in section
    assert "'Affinity Group page' or 'Email to Affinity Group' without a group" in section
    assert "ask which group" in section


def test_create_passes_fields_exactly_as_confirmed_preview():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    assert "affiliation, where_to_share, tags, and affinity_group" in section
    assert "exactly as the confirmed preview showed them" in section


def test_style_exception_for_preview_lines():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    assert "except the preview's affiliation and sharing lines" in section


def test_create_leaves_published_date_out_unless_asked():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    assert "Leave `published_date` out unless the user asks for a specific publish date" in section
    assert "never the date of an event the announcement describes" in section


def test_preview_shows_non_today_publish_date_and_digest_effect():
    section = _normalize(ANNOUNCEMENTS_WORKFLOWS_SECTION)
    assert "publish date other than today is set" in section
    assert "future date keeps the announcement out of the Bi-Weekly Digest until then" in section
