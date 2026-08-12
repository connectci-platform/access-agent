from pathlib import Path

import pytest

from src.redteam.suite import (
    SuiteKeyMismatch,
    SuiteVersionMismatch,
    assert_versions_match,
    join_suite,
    load_baseline,
    load_prompts,
)

FIXTURES = Path(__file__).parent / "fixtures"
BASELINE = Path(__file__).parent / "suite-v1" / "baseline.json"


def test_baseline_derives_defended_for_unlisted_ids():
    b = load_baseline(BASELINE)
    assert b.verdicts["wrapped__aligned__stop-sign"] == "known-jailbreak"
    assert b.verdicts["wrapped__wikipedia_with_title__credentials"] == "soft"
    # an id not in known_jailbreak/soft resolves to defended via join
    p = load_prompts(FIXTURES / "prompts.sample.json")
    items = {i.id: i for i in join_suite(b, p)}
    assert items["floor__illegal__benign-placeholder"].expected == "defended"


def test_version_match_passes_when_equal():
    b = load_baseline(BASELINE)
    p = load_prompts(FIXTURES / "prompts.sample.json")
    assert_versions_match(b, p)  # no raise


def test_version_mismatch_aborts(tmp_path):
    b = load_baseline(BASELINE)
    p = load_prompts(FIXTURES / "prompts.sample.json")
    object.__setattr__(p, "suite_version", "v2-2026-99-99")
    with pytest.raises(SuiteVersionMismatch):
        assert_versions_match(b, p)


def test_join_flags_orphan_prompt_id(tmp_path):
    b = load_baseline(BASELINE)
    p = load_prompts(FIXTURES / "prompts.sample.json")
    # a known_jailbreak id with no corresponding prompt entry is an orphan
    object.__setattr__(b, "verdicts", {**b.verdicts, "wrapped__ghost__nowhere": "known-jailbreak"})
    with pytest.raises(SuiteKeyMismatch):
        join_suite(b, p)
