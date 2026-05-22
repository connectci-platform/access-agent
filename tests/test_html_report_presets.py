"""Tests for the preset registry in ``src.eval.html_report.notes``."""

from __future__ import annotations

import pytest


def test_grand_prix_preset_matches_legacy_module_constants() -> None:
    """grand-prix preset values are byte-identical to the legacy module-level constants."""
    from src.eval.html_report import notes

    preset = notes.get_preset("grand-prix")
    assert preset.report_subtitle == notes.REPORT_SUBTITLE
    assert preset.battery_info == notes.BATTERY_INFO
    assert preset.observations == notes.OBSERVATIONS


def test_unknown_preset_raises() -> None:
    """Unknown preset names should raise ValueError with the valid choices."""
    from src.eval.html_report import notes

    with pytest.raises(ValueError, match="unknown preset"):
        notes.get_preset("nope")
