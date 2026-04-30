"""Tests for the --preset flag on html_report.

Covers the preset registry in ``src.eval.html_report.notes``:

1. The legacy module-level constants (REPORT_SUBTITLE, BATTERY_INFO,
   OBSERVATIONS) remain byte-identical to the ``grand-prix`` preset — this is
   the backward-compatibility contract for any caller that imported those
   names before the preset registry existed.
2. The ``phase3-parity`` preset's prose is actually neutral — it doesn't
   frame the comparison as "raw RAG vs Agent" the way grand-prix does, and
   it references the loop/chain distinction the parity report is about.
3. Unknown preset names raise ValueError (not KeyError) with a helpful
   message listing valid choices.
"""

from __future__ import annotations

import pytest


def test_grand_prix_preset_matches_legacy_module_constants() -> None:
    """grand-prix preset values are byte-identical to the legacy module-level constants."""
    from src.eval.html_report import notes

    preset = notes.get_preset("grand-prix")
    assert preset.report_subtitle == notes.REPORT_SUBTITLE
    assert preset.battery_info == notes.BATTERY_INFO
    assert preset.observations == notes.OBSERVATIONS


def test_phase3_parity_preset_has_no_raw_rag_language() -> None:
    """Parity preset prose must not mention 'raw RAG' (that's grand-prix-specific).

    Also asserts the subtitle mentions the loop/chain distinction that is the
    whole point of a Phase 3 parity comparison — if neither word appears, the
    prose has drifted off-topic.
    """
    from src.eval.html_report import notes

    preset = notes.get_preset("phase3-parity")

    # Subtitle must frame the comparison in terms of loop vs chain.
    subtitle_lower = preset.report_subtitle.lower()
    assert "loop" in subtitle_lower or "chain" in subtitle_lower, (
        f"phase3-parity subtitle should reference loop or chain, got: {preset.report_subtitle!r}"
    )

    # Subtitle must not carry the grand-prix framing.
    assert "raw rag" not in subtitle_lower, (
        f"phase3-parity subtitle should not mention 'Raw RAG', got: {preset.report_subtitle!r}"
    )

    # Neither the battery descriptions nor observations should
    # hard-code the grand-prix "raw RAG pulls ahead / structural disadvantage"
    # framing. Join all editorial text, lowercase, and spot-check.
    editorial_text = " ".join(
        [
            preset.report_subtitle,
            *(o.get("text", "") for o in preset.observations),
            *(f"{b.get('what', '')} {b.get('why', '')}" for b in preset.battery_info.values()),
        ]
    ).lower()
    # "pull ahead of raw RAG" and "structural disadvantage" are the two
    # grand-prix-specific phrases most likely to mislead on a parity report.
    assert "pull ahead of raw rag" not in editorial_text
    assert "structural disadvantage" not in editorial_text


def test_unknown_preset_raises() -> None:
    """Unknown preset names should raise ValueError with the valid choices."""
    from src.eval.html_report import notes

    with pytest.raises(ValueError, match="unknown preset"):
        notes.get_preset("nope")
