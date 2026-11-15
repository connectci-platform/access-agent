"""Chat upload limits (D8-3086): the values are the spec's, as settings."""

import pytest

from src.config import Settings

MB = 1024 * 1024
KB = 1024

_PAIRS = [
    ("UPLOAD_MAX_FILES", 5, 2),
    ("UPLOAD_MAX_TOTAL_BYTES", 20 * MB, 8 * MB),
    ("UPLOAD_MAX_IMAGE_BYTES", 10 * MB, 5 * MB),
    ("UPLOAD_MAX_PDF_BYTES", 10 * MB, 5 * MB),
    ("UPLOAD_MAX_TEXT_BYTES", 200 * KB, 100 * KB),
    ("UPLOAD_THREAD_MAX_BYTES", 100 * MB, 30 * MB),
]

_SHARED = {
    "UPLOAD_MAX_PIXELS": 25_000_000,
    "UPLOAD_IMAGE_MAX_EDGE": 1600,
    "UPLOAD_PNG_MAX_BYTES": 2 * MB,
    "UPLOAD_TEXT_MAX_CHARS": 40_000,
    "UPLOAD_PDF_MAX_PAGES": 30,
    "UPLOAD_PDF_TIMEOUT_S": 10,
    "UPLOAD_PDF_RLIMIT_AS": 384 * MB,
    "UPLOAD_IMAGE_TIMEOUT_S": 10,
    "UPLOAD_IMAGE_RLIMIT_AS": 512 * MB,
    "UPLOAD_MESSAGE_MAX_TOKENS": 24_000,
    "UPLOAD_DECODE_CONCURRENCY": 2,
    "UPLOAD_DECODE_WAIT_S": 20,
    "UPLOAD_SHARE_TIMEOUT_S": 60,
    "UPLOAD_MAX_INFLIGHT_PER_CALLER": 2,
    "UPLOAD_RETENTION_DAYS": 30,
    "UPLOAD_WINDOW_TURNS": 2,
    "UPLOAD_MAX_BLOCKS": 50,
    "RUN_BODY_MAX_BRACKETS": 20_000,
    "RUN_MAX_MESSAGES": 50,
    "QUESTION_MAX_CHARS": 40_000,
}


@pytest.fixture
def cfg():
    return Settings(_env_file=None)


@pytest.mark.parametrize(("name", "signed_in", "anon"), _PAIRS)
def test_per_user_limits(cfg, name, signed_in, anon):
    assert getattr(cfg, name) == signed_in
    assert getattr(cfg, name + "_ANON") == anon


@pytest.mark.parametrize(("name", "anon_name"), [(p[0], p[0] + "_ANON") for p in _PAIRS])
def test_anonymous_limits_do_not_exceed_signed_in(cfg, name, anon_name):
    assert getattr(cfg, anon_name) <= getattr(cfg, name)


@pytest.mark.parametrize(("name", "value"), list(_SHARED.items()))
def test_shared_limits(cfg, name, value):
    assert getattr(cfg, name) == value
