"""Tests for --system CLI semantics (agent_full = loop, agent_full_legacy = old chain)."""

from __future__ import annotations

import contextlib
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_flag_after_test():
    """Ensure the module-level settings singleton is reset between tests."""
    from src.config import settings

    original = settings.USE_TOOL_CALLING_LOOP
    try:
        yield
    finally:
        settings.USE_TOOL_CALLING_LOOP = original


@pytest.mark.asyncio
async def test_agent_full_sets_use_tool_calling_loop_true():
    """--system agent_full → settings.USE_TOOL_CALLING_LOOP = True."""
    from src.config import settings
    from src.eval.scorer import run_eval

    # Ensure flag starts False to verify the override actually fires
    settings.USE_TOOL_CALLING_LOOP = False

    # Mock everything downstream of the flag override
    with (
        patch("src.eval.scorer.load_questions", return_value=[]),
        patch("src.eval.scorer.get_catalog_aggregator") as mock_agg,
        patch("src.eval.scorer.ToolRegistry"),
        patch("src.eval.scorer.EvalDB"),
        patch("src.eval.scorer.Judge"),
        patch("src.eval.scorer.get_git_info", return_value={}),
    ):
        mock_agg.return_value.fetch_catalog = AsyncMock(return_value={})
        # We don't care whether the outer run succeeds — only the flag override
        with contextlib.suppress(Exception):
            await run_eval(question_set_path="dummy.json", system="agent_full")

    assert settings.USE_TOOL_CALLING_LOOP is True


@pytest.mark.asyncio
async def test_agent_full_legacy_sets_use_tool_calling_loop_false():
    """--system agent_full_legacy → settings.USE_TOOL_CALLING_LOOP = False."""
    from src.config import settings
    from src.eval.scorer import run_eval

    # Start flag True to verify the override pulls it down
    settings.USE_TOOL_CALLING_LOOP = True

    with (
        patch("src.eval.scorer.load_questions", return_value=[]),
        patch("src.eval.scorer.get_catalog_aggregator") as mock_agg,
        patch("src.eval.scorer.ToolRegistry"),
        patch("src.eval.scorer.EvalDB"),
        patch("src.eval.scorer.Judge"),
        patch("src.eval.scorer.get_git_info", return_value={}),
    ):
        mock_agg.return_value.fetch_catalog = AsyncMock(return_value={})
        with contextlib.suppress(Exception):
            await run_eval(question_set_path="dummy.json", system="agent_full_legacy")

    assert settings.USE_TOOL_CALLING_LOOP is False


@pytest.mark.asyncio
async def test_non_agent_systems_also_force_flag_false():
    """--system raw_rag or agent_rag_only → flag forced False.

    The flag is a no-op for these systems, but we set it for log clarity.
    """
    from src.config import settings
    from src.eval.scorer import run_eval

    settings.USE_TOOL_CALLING_LOOP = True

    with (
        patch("src.eval.scorer.load_questions", return_value=[]),
        patch("src.eval.scorer.get_catalog_aggregator") as mock_agg,
        patch("src.eval.scorer.ToolRegistry"),
        patch("src.eval.scorer.EvalDB"),
        patch("src.eval.scorer.Judge"),
        patch("src.eval.scorer.get_git_info", return_value={}),
    ):
        mock_agg.return_value.fetch_catalog = AsyncMock(return_value={})
        with contextlib.suppress(Exception):
            await run_eval(question_set_path="dummy.json", system="raw_rag")

    assert settings.USE_TOOL_CALLING_LOOP is False


def test_gen_semantic_run_id_format():
    """Semantic run IDs follow {shortcode}-{YYYYMMDD}-{HHMMSS}-{hash6} and fit in 36 chars."""
    import re

    from src.eval.runner import gen_semantic_run_id

    for system in ("agent_full", "agent_full_legacy", "agent_rag_only", "raw_rag"):
        run_id = gen_semantic_run_id(system)  # type: ignore[arg-type]
        # Must fit in the existing String(36) column
        assert len(run_id) <= 36, f"ID too long for String(36): {run_id!r}"
        # Must match the format
        assert re.match(r"^[a-z_]+-\d{8}-\d{6}-[0-9a-f]{6}$", run_id), (
            f"ID does not match expected format: {run_id!r}"
        )
        # Shortcode must be one of the four expected values
        shortcode = run_id.split("-")[0]
        assert shortcode in {"loop", "chain", "rag_only", "raw_rag"}


def test_gen_semantic_run_id_uses_correct_shortcode_per_system():
    """Each system maps to its expected shortcode."""
    from src.eval.runner import gen_semantic_run_id

    assert gen_semantic_run_id("agent_full").startswith("loop-")
    assert gen_semantic_run_id("agent_full_legacy").startswith("chain-")
    assert gen_semantic_run_id("agent_rag_only").startswith("rag_only-")
    assert gen_semantic_run_id("raw_rag").startswith("raw_rag-")


def test_gen_semantic_run_id_is_unique_across_rapid_calls():
    """Two calls in the same second must produce different IDs (hash6 suffix)."""
    from src.eval.runner import gen_semantic_run_id

    ids = {gen_semantic_run_id("agent_full") for _ in range(20)}
    assert len(ids) == 20, f"Expected 20 unique IDs, got {len(ids)}: {ids}"
