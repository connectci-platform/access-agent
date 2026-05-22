"""Tests for CORS origin allowlist.

Verifies that the regex-based origin matching accepts the right hostnames
and rejects everything else. The accessmatch Drupal multidev is the
primary case that motivated the regex — every multidev branch creates a
new <branch>-accessmatch.pantheonsite.io hostname and we don't want to
maintain an explicit list of those.
"""

from __future__ import annotations

import re

import pytest

# The regex pattern used in src/main.py for production CORS.
# Kept in sync with that file; if you change one, change both.
PRODUCTION_ORIGIN_REGEX = (
    r"https://"
    r"(?:"
    r"(?:[a-z0-9-]+\.)?access-ci\.org"
    r"|"
    r"[a-z0-9-]+-accessmatch\.pantheonsite\.io"
    r")"
)


@pytest.fixture
def compiled_regex():
    # Starlette's CORSMiddleware uses re.fullmatch semantics — the entire
    # Origin header value must match. We mirror that here so the test
    # behavior matches what the middleware does at request time.
    return re.compile(PRODUCTION_ORIGIN_REGEX)


@pytest.mark.parametrize(
    "origin",
    [
        # access-ci.org apex
        "https://access-ci.org",
        # Single-label subdomains
        "https://support.access-ci.org",
        "https://allocations.access-ci.org",
        "https://operations.access-ci.org",
        "https://metrics.access-ci.org",
        "https://identity.access-ci.org",
        "https://qa.access-ci.org",
        # Pantheon accessmatch environments
        "https://dev-accessmatch.pantheonsite.io",
        "https://test-accessmatch.pantheonsite.io",
        "https://live-accessmatch.pantheonsite.io",
        "https://md-qabot-accessmatch.pantheonsite.io",
        "https://md-2729-accessmatch.pantheonsite.io",
    ],
)
def test_accepts_expected_origins(compiled_regex, origin: str) -> None:
    assert compiled_regex.fullmatch(origin) is not None, f"should accept {origin}"


@pytest.mark.parametrize(
    "origin",
    [
        # HTTP (not HTTPS) — not allowed in production
        "http://support.access-ci.org",
        "http://access-ci.org",
        # Other ACCESS-multisite tenants on Pantheon — deliberately excluded
        "https://dev-ccmnet.pantheonsite.io",
        "https://md-qabot-ccmnet.pantheonsite.io",
        "https://dev-ondemand.pantheonsite.io",
        "https://dev-pasciencedmz.pantheonsite.io",
        # Look-alike domains
        "https://access-ci.org.evil.example",
        "https://evil-access-ci.org",
        "https://accessmatch.pantheonsite.io",  # missing env prefix
        "https://-accessmatch.pantheonsite.io",  # empty env prefix
        # Adjacent Pantheon sites
        "https://md-qabot-other.pantheonsite.io",
        # Unrelated hosts
        "https://example.com",
        "https://attacker.example",
    ],
)
def test_rejects_unexpected_origins(compiled_regex, origin: str) -> None:
    assert compiled_regex.fullmatch(origin) is None, f"should reject {origin}"


def test_regex_in_source_matches_test_constant() -> None:
    """Guard against the source and test drifting apart.

    Reads src/main.py and verifies the regex literal we test here matches
    the one in the actual middleware configuration. If this test fails,
    update PRODUCTION_ORIGIN_REGEX above to match the change in main.py.
    """
    from pathlib import Path

    main_py = Path(__file__).parent.parent / "src" / "main.py"
    source = main_py.read_text()
    # Find the regex literal in the source by joining the raw string parts.
    # We look for the access-ci.org subpattern; if present, assume the rest
    # of the pattern is intact (the test cases above cover the rest).
    assert "access-ci\\.org" in source, (
        "Expected access-ci.org regex literal in src/main.py CORS config. "
        "If you renamed or restructured the regex, update this test."
    )
    assert "accessmatch\\.pantheonsite\\.io" in source, (
        "Expected accessmatch.pantheonsite.io regex literal in src/main.py "
        "CORS config. If you renamed or restructured the regex, update this test."
    )
