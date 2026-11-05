"""URLs baked into the system prompt must point at live pages.

The agent hands these links to users verbatim. A dead one lands the user on
a 404 at exactly the moment they are trying to fix something — which is what
happened with https://access-ci.org/sign-in (404 in production, 2026-09-04),
surfaced to anonymous users asking for personal data.

These are offline assertions (no network in CI): they pin the URLs whose
liveness was checked by hand, so a regression to a known-dead address fails
here. Re-verify by hand when adding a URL to the prompt.
"""

from src.agent.prompts.system_prompt import build_system_prompt

# Checked 2026-09-04: 404.
DEAD_URLS = ("https://access-ci.org/sign-in",)


def test_anonymous_prompt_points_at_the_configured_login_url():
    """The anonymous branch sends users to the same sign-in the refusal text
    and the requires_auth event name. Checked 2026-11-04: it redirects to the
    support site's CILogon sign-in."""
    from src.config import settings

    prompt = build_system_prompt(acting_user=None)
    assert settings.LOGIN_URL in prompt


def test_no_known_dead_urls_in_either_prompt_branch():
    for acting_user in (None, "jsmith@access-ci.org"):
        prompt = build_system_prompt(acting_user=acting_user)
        for dead in DEAD_URLS:
            assert dead not in prompt, f"prompt contains known-dead URL {dead}"
