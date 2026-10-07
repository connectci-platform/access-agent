"""Tests for battery-key normalization and per-battery folding in the HTML report.

Both regressions these cover produced a *rendered* "undefined" rather than an
error, so they were invisible to the generator's own success path.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from src.eval.db import EvalDB


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("eval/questions/capability_review_battery.yaml", "capability_review_battery"),
        ("no_suffix_battery", "no_suffix_battery"),
    ],
)
def test_normalize_question_set_strips_yaml(raw: str, expected: str) -> None:
    """YAML batteries must normalize like JSON ones, or every preset lookup misses."""
    from src.eval.html_report.builder import _normalize_question_set

    assert _normalize_question_set(raw) == expected


def _pair(battery: str, qid: str, delta: float) -> dict[str, object]:
    """One all_pairs entry, carrying only what _build_per_battery_views reads."""
    a = 0.4
    return {
        "qid": qid,
        "battery": battery,
        "question": qid,
        "raw_rag": {"composite": a, "duration_ms": 1000},
        "agent_full": {"composite": a + delta, "duration_ms": 2000},
        "delta": delta,
    }


def test_per_battery_covers_batteries_outside_the_preset_order() -> None:
    """A battery absent from BATTERY_ORDER still gets views, via the fallback.

    Regression: the fold iterated BATTERY_ORDER, so an unlisted battery was
    never visited, per_battery came back empty and the header rendered
    "undefined".
    """
    from src.eval.html_report.builder import _build_per_battery_views

    pairs = [
        _pair("capability_review", "tc-1", 0.5),
        _pair("capability_review", "tc-2", 0.0),
        _pair("gapfill", "gf-1", 0.4),
    ]
    per_battery, info, labels, order = _build_per_battery_views(pairs, ["friendly_battery"], {})

    assert set(per_battery) == {"capability_review", "gapfill"}
    assert per_battery["capability_review"]["n"] == 2
    assert per_battery["capability_review"]["wins"] == 1
    assert per_battery["capability_review"]["ties"] == 1
    assert per_battery["capability_review"]["losses"] == 0
    assert labels["capability_review"] == "Capability Review"
    assert info["gapfill"]["name"] == "Gapfill"
    assert order == ["capability_review", "gapfill"]


def test_per_battery_puts_preset_batteries_first() -> None:
    """Preset order leads; batteries outside it follow, sorted."""
    from src.eval.html_report.builder import _build_per_battery_views

    pairs = [_pair("zeta", "z-1", 0.1), _pair("gapfill", "gf-1", 0.1), _pair("alpha", "a-1", 0.1)]
    _, _, _, order = _build_per_battery_views(pairs, ["gapfill_battery"], {})

    assert order == ["gapfill", "alpha", "zeta"]


def test_per_battery_prefers_preset_info_over_fallback() -> None:
    """A battery the preset describes keeps the preset's prose."""
    from src.eval.html_report.builder import _build_per_battery_views

    supplied = {
        "gapfill_battery": {"name": "Gap Fill", "count": 11, "what": "w", "why": "y"},
    }
    _, info, labels, _ = _build_per_battery_views(
        [_pair("gapfill", "gf-1", 0.1)], ["gapfill_battery"], supplied
    )

    assert labels["gapfill"] == "Gap Fill"
    assert info["gapfill"]["what"] == "w"


def test_pick_run_ids_on_date_filters_to_that_day(tmp_path) -> None:
    """on_date restricts pick_run_ids to runs created on that UTC day.

    Seeds two agent_full/friendly_battery runs on different days and asserts
    the date filter keeps only the matching one, while an unfiltered call
    considers both (picking the newest, per pick_run_ids' "newest per
    (system, question_set)" contract).
    """
    from src.eval.html_report.builder import pick_run_ids

    url = f"sqlite:///{tmp_path}/e.db"
    db = EvalDB(url)

    run_a = db.create_run(
        run_type="pre_production",
        metadata_={"system": "agent_full"},
        question_set="friendly_battery",
        created_at=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
    )
    run_b = db.create_run(
        run_type="pre_production",
        metadata_={"system": "agent_full"},
        question_set="friendly_battery",
        created_at=datetime(2026, 1, 16, 12, 0, 0, tzinfo=UTC),
    )

    filtered = pick_run_ids(url, on_date=date(2026, 1, 15))
    assert [ref.id for ref in filtered] == [str(run_a.id)]
    assert filtered[0].system == "agent_full"
    assert filtered[0].question_set_key == "friendly_battery"

    unfiltered = pick_run_ids(url)
    assert [ref.id for ref in unfiltered] == [str(run_b.id)]
