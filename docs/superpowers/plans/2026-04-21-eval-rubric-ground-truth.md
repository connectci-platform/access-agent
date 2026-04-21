# Eval Rubric Ground Truth — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the LLM judge authored ground-truth "required facts" per question (sourced from human annotation in Argilla) so factually-wrong-but-plausible answers stop scoring well.

**Architecture:** Extend the Argilla dataset schema with `required_facts` + `ground_truth_stability` + `ground_truth_valid_until`. Add a `GroundTruthCache` that pulls the latest authored facts per question at eval-run start. Extend the judge prompt and response schema to consume facts and emit per-fact coverage. No agent-side changes; this is eval-layer only.

**Tech Stack:** Python 3.11+, Argilla SDK 2.x (already a dev dep), OpenAI-compatible judge LLM (existing), `pytest` / `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-04-21-eval-rubric-ground-truth-design.md`.

**Parallel to launch, not a launch blocker.** If any phase below is incomplete when launch ships, it is deferred post-launch. No launch delay.

---

## File Structure

**Create:**
- `src/eval/argilla_ground_truth.py` — `RequiredFacts` dataclass, `GroundTruthCache` class, markdown-bullet parser.
- `tests/test_eval_ground_truth.py` — unit tests for `RequiredFacts` parsing, `GroundTruthCache` behavior (mocked Argilla client), stability/expiry filtering.

**Modify:**
- `src/eval/argilla_push.py` — add three new Argilla schema fields in `create_eval_dataset()`; update guidelines text.
- `src/eval/rubric.py` — extend `build_judge_prompt()` signature and prompt body to consume required facts.
- `src/eval/judge.py` — extend `JudgeResult` with `required_facts_coverage`; `parse_judge_response()` tolerates + extracts the new field; `Judge.score()` accepts and forwards `required_facts`.
- `src/eval/scorer.py` — prime the `GroundTruthCache` at run start; fetch per-question facts before judging.
- `tests/test_eval_judge.py` (may exist already) — add cases for prompt-with-facts and parse-with-coverage.
- `tests/test_eval_argilla.py` — add a schema test for the three new fields.

**Not touched:**
- `src/agent/*` — this plan is eval-layer only.
- `src/eval/runner.py` — agent execution unchanged.
- `src/eval/questions.py` — per-question JSON schema unchanged; required facts live in Argilla, not JSON.
- `src/eval/db.py` — DB schema for scores; per-fact coverage can land in the existing `justifications` dict via `correctness`.
- `eval/questions/*.json` — no edits.

---

## Phase 1 — Argilla Schema Extension

### Task 1.1: Create branch

**Files:** none.

- [ ] **Step 1: Create feature branch**

Run:
```bash
git checkout main
git pull
git checkout -b feature/eval-ground-truth
```
Expected: new branch from current `main`.

- [ ] **Step 2: Verify clean state**

Run:
```bash
uv run pytest -q
```
Expected: all tests pass on current `main`.

### Task 1.2: Write the failing test for the new schema fields

**Files:**
- Modify: `tests/test_eval_argilla.py`

- [ ] **Step 1: Add schema-coverage tests**

Append to `tests/test_eval_argilla.py`:

```python
def test_dataset_settings_include_ground_truth_fields():
    """The dataset schema must include required_facts, ground_truth_stability,
    and ground_truth_valid_until. These are required by the ground-truth
    pull module to function."""
    from src.eval.argilla_push import build_dataset_settings

    settings = build_dataset_settings()

    # required_facts is a TextQuestion reviewers author into.
    question_names = [q.name for q in settings.questions]
    assert "required_facts" in question_names
    assert "ground_truth_stability" in question_names

    # Stability label values are the four expected strings.
    stability_q = next(q for q in settings.questions if q.name == "ground_truth_stability")
    assert set(stability_q.labels) == {"stable", "time_bound", "not_applicable", "not_reviewed"}

    # valid_until is metadata, not a question.
    metadata_names = [m.name for m in settings.metadata]
    assert "ground_truth_valid_until" in metadata_names
```

- [ ] **Step 2: Run to confirm it fails**

Run:
```bash
uv run pytest tests/test_eval_argilla.py::test_dataset_settings_include_ground_truth_fields -v
```
Expected: `AttributeError: module 'src.eval.argilla_push' has no attribute 'build_dataset_settings'` (the function doesn't exist yet — we'll extract it from `create_eval_dataset()`).

### Task 1.3: Extract the dataset settings into a testable helper

**Files:**
- Modify: `src/eval/argilla_push.py`

The current `create_eval_dataset()` builds `rg.Settings(...)` inline, then instantiates the Argilla client. To test the schema without Argilla, extract the settings builder.

- [ ] **Step 1: Add `build_dataset_settings()` above `create_eval_dataset()`**

Insert before the existing `create_eval_dataset()`:

```python
def build_dataset_settings() -> Any:
    """Build the Argilla dataset Settings object.

    Extracted from create_eval_dataset() so tests can verify schema shape
    without needing a live Argilla connection.
    """
    try:
        import argilla as rg
    except ImportError:
        logger.error("argilla package not installed. Install with: pip install argilla>=2.0.0")
        raise

    return rg.Settings(
        guidelines=_dataset_guidelines(),
        fields=_dataset_fields(rg),
        questions=_dataset_questions(rg),
        metadata=_dataset_metadata(rg),
    )


def _dataset_guidelines() -> str:
    return (
        "# How to Review\n\n"
        "Evaluate the agent's answer using the scoring rubric below. "
        "The LLM judge's scores appear as suggestions — accept or override based on your judgment.\n\n"
        "Use the **RAG Documents Retrieved** and **MCP Tool Results** panels to verify "
        "the agent's claims against its actual sources. The **Agent Trace** panel shows "
        "how the agent classified the query and which tools it chose.\n\n"
        "## Scoring Rubric (1-5 each)\n\n"
        "- **Correctness**: Does the answer accurately represent its sources? "
        "Score 5 if faithful to sources, even if the sources themselves are outdated. "
        "Score 1-2 if the agent hallucinated or contradicted its sources.\n"
        "- **Completeness**: Does the answer address all parts of the question? "
        "Score 5 if thorough, 1 if it misses the main point.\n"
        "- **Relevance**: Does the answer stay on topic? "
        "Score 5 if focused, 1 if mostly irrelevant padding.\n"
        "- **Citation Quality**: Are URLs present, valid, and from the source docs? "
        "Score 5 if all relevant URLs preserved, 1 if missing or hallucinated.\n"
        "- **Hedging**: Is the agent's confidence calibrated to its source quality? "
        "Score 5 if well-calibrated, 1 if confidently wrong or hedges everything.\n\n"
        "## Decision\n\n"
        "- **Approved**: Answer is good enough to serve to users. Minor quibbles are OK.\n"
        "- **Needs revision**: Answer has real problems (missing key info, wrong facts, bad URLs) "
        "but the agent was on the right track. Indicates the agent or its sources need improvement.\n"
        "- **Rejected**: Answer is actively wrong, misleading, or unhelpful. "
        "Would cause confusion or harm if served to a user.\n\n"
        "## Feedback\n\n"
        "Use the optional feedback field to explain *why* you scored differently from the judge, "
        "or to note issues the rubric doesn't capture (e.g., tone, formatting, missing context).\n\n"
        "## Ground Truthing (optional, highly valuable)\n\n"
        "If the answer has factual errors you can articulate, author required facts in the "
        "**Required Facts** field. List each fact on its own bullet — atomic claims like "
        "'Delta has A100 GPUs' rather than full paragraphs. Set **Ground Truth Stability**:\n"
        "- `stable` — docs/policies that rarely change (how-tos, hardware specs, URLs).\n"
        "- `time_bound` — facts that will go stale; also populate the `ground_truth_valid_until` "
        "metadata with an ISO date.\n"
        "- `not_applicable` — the question doesn't have enumerable required facts.\n"
        "- `not_reviewed` — you haven't ground-truthed this question (default)."
    )


def _dataset_fields(rg: Any) -> list[Any]:
    return [
        rg.TextField(name="question_id", title="Question ID", required=False),
        rg.TextField(name="run_id", title="Run ID", required=False),
        rg.TextField(name="user_query", title="User Query", required=True),
        rg.TextField(name="agent_answer", title="Agent Response", use_markdown=True, required=True),
        rg.TextField(
            name="judge_justifications",
            title="Judge Justifications",
            use_markdown=True,
            required=False,
        ),
        rg.TextField(
            name="rag_context", title="RAG Documents Retrieved", use_markdown=True, required=False
        ),
        rg.TextField(
            name="tool_results", title="MCP Tool Results", use_markdown=True, required=False
        ),
        rg.TextField(
            name="agent_reasoning",
            title="Agent Trace & Reasoning",
            use_markdown=True,
            required=False,
        ),
    ]


def _dataset_questions(rg: Any) -> list[Any]:
    return [
        rg.RatingQuestion(
            name="correctness",
            title="Correctness (1=contradicts sources, 5=faithful)",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
        rg.RatingQuestion(
            name="completeness",
            title="Completeness (1=misses main point, 5=thorough)",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
        rg.RatingQuestion(
            name="relevance",
            title="Relevance (1=off-topic, 5=focused)",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
        rg.RatingQuestion(
            name="citation_quality",
            title="Citation Quality (1=no/bad URLs, 5=all preserved)",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
        rg.RatingQuestion(
            name="hedging",
            title="Hedging (1=miscalibrated, 5=well-calibrated)",
            values=[1, 2, 3, 4, 5],
            required=True,
        ),
        rg.LabelQuestion(
            name="decision",
            title="Decision",
            labels=["approved", "needs_revision", "rejected"],
            required=True,
        ),
        rg.TextQuestion(name="feedback", title="Feedback (optional)", required=False),
        rg.TextQuestion(
            name="required_facts",
            title="Required Facts (optional, bulleted list)",
            required=False,
            use_markdown=True,
        ),
        rg.LabelQuestion(
            name="ground_truth_stability",
            title="Ground Truth Stability",
            labels=["stable", "time_bound", "not_applicable", "not_reviewed"],
            required=False,
        ),
    ]


def _dataset_metadata(rg: Any) -> list[Any]:
    return [
        rg.TermsMetadataProperty(name="battery", title="Question Battery"),
        rg.TermsMetadataProperty(name="capabilities", title="Capabilities"),
        rg.TermsMetadataProperty(name="classification", title="Query Classification"),
        rg.TermsMetadataProperty(name="tools_used", title="Tools Used"),
        rg.TermsMetadataProperty(name="capability_area", title="Capability Area"),
        rg.TermsMetadataProperty(name="composite_score", title="Composite Score"),
        rg.TermsMetadataProperty(name="review_priority", title="Review Priority"),
        rg.TermsMetadataProperty(name="filter_run_id", title="Eval Run ID"),
        rg.TermsMetadataProperty(name="agent_branch", title="Agent Branch"),
        rg.TermsMetadataProperty(name="judge_model", title="Judge Model"),
        rg.TermsMetadataProperty(
            name="ground_truth_valid_until",
            title="Ground Truth Valid Until (ISO date)",
        ),
    ]
```

- [ ] **Step 2: Simplify `create_eval_dataset()` to use the helper**

Replace the body of `create_eval_dataset()` up through the `settings = rg.Settings(...)` block with:

```python
def create_eval_dataset(argilla_url: str, argilla_api_key: str, dataset_name: str) -> Any:
    """Create an Argilla dataset with the eval rubric schema. Lazily imports argilla."""
    try:
        import argilla as rg
    except ImportError:
        logger.error("argilla package not installed. Install with: pip install argilla>=2.0.0")
        raise

    client = rg.Argilla(api_url=argilla_url, api_key=argilla_api_key)

    try:
        existing = client.datasets(name=dataset_name)
        if existing:
            logger.info(f"Dataset '{dataset_name}' already exists, reusing")
            return existing
    except Exception:
        pass

    settings = build_dataset_settings()
    dataset = rg.Dataset(name=dataset_name, settings=settings)
    dataset.create()
    logger.info(f"Created Argilla dataset '{dataset_name}'")
    return dataset
```

- [ ] **Step 3: Run the schema test to verify it passes**

Run:
```bash
uv run pytest tests/test_eval_argilla.py::test_dataset_settings_include_ground_truth_fields -v
```
Expected: PASS.

- [ ] **Step 4: Run the full eval test suite**

Run:
```bash
uv run pytest tests/test_eval_argilla.py -v
```
Expected: all tests pass (existing tests still green after the refactor).

- [ ] **Step 5: Commit**

Run:
```bash
git add src/eval/argilla_push.py tests/test_eval_argilla.py
git commit -m "feat(eval): extend argilla schema with required_facts + stability fields"
```

### Task 1.4: Recreate the eval-production dataset

**Files:** none. This is an Argilla operational step, not a code change.

- [ ] **Step 1: Confirm the current `eval-production` dataset is empty**

Run (requires `ARGILLA_URL` and `ARGILLA_API_KEY` set in your local `.env`):

```bash
uv run python -c "
import os, argilla as rg
c = rg.Argilla(api_url=os.environ['ARGILLA_URL'], api_key=os.environ['ARGILLA_API_KEY'])
ds = c.datasets(name='eval-production')
print('records:', sum(1 for _ in ds.records) if ds else 'no dataset')
"
```
Expected: `records: 0` or `no dataset`. If non-zero, stop and surface to Drew — the spec assumes no existing annotations.

- [ ] **Step 2: Delete the existing dataset**

Run:
```bash
uv run python -c "
import os, argilla as rg
c = rg.Argilla(api_url=os.environ['ARGILLA_URL'], api_key=os.environ['ARGILLA_API_KEY'])
ds = c.datasets(name='eval-production')
if ds:
    ds.delete()
    print('deleted')
else:
    print('nothing to delete')
"
```
Expected: either `deleted` or `nothing to delete`.

- [ ] **Step 3: Create the new dataset via the updated code**

Run:
```bash
uv run python -c "
from src.config import settings
from src.eval.argilla_push import create_eval_dataset
create_eval_dataset(settings.ARGILLA_URL, settings.ARGILLA_API_KEY, 'eval-production')
"
```
Expected: `Created Argilla dataset 'eval-production'`.

- [ ] **Step 4: Verify the schema in the Argilla UI**

Open Argilla in a browser, navigate to the `eval-production` dataset settings, and confirm that `required_facts`, `ground_truth_stability`, and `ground_truth_valid_until` appear alongside the existing fields.

---

## Phase 2 — Ground Truth Pull Module

### Task 2.1: Write the failing tests for `RequiredFacts` and `GroundTruthCache`

**Files:**
- Create: `tests/test_eval_ground_truth.py`

- [ ] **Step 1: Write test file**

```python
"""Unit tests for the Argilla ground-truth cache and required-facts parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.eval.argilla_ground_truth import (
    GroundTruthCache,
    RequiredFacts,
    parse_facts_markdown,
)


class TestParseFactsMarkdown:
    def test_parses_dash_bullets(self):
        text = "- Delta has A100 GPUs\n- Answer should cite docs.access-ci.org"
        facts = parse_facts_markdown(text)
        assert facts == ["Delta has A100 GPUs", "Answer should cite docs.access-ci.org"]

    def test_parses_asterisk_bullets(self):
        text = "* fact one\n* fact two"
        assert parse_facts_markdown(text) == ["fact one", "fact two"]

    def test_empty_string(self):
        assert parse_facts_markdown("") == []

    def test_none(self):
        assert parse_facts_markdown(None) == []

    def test_ignores_non_bullet_lines(self):
        text = "Some preamble\n- fact one\n\n- fact two\nSome trailing text"
        assert parse_facts_markdown(text) == ["fact one", "fact two"]

    def test_strips_whitespace(self):
        text = "-    padded   \n-another"
        # Second line has no space after dash — still treat as bullet if the marker is at col 0.
        assert parse_facts_markdown(text) == ["padded", "another"]


@dataclass
class _FakeRecord:
    """Shape-compatible with argilla.Record for what the cache reads."""

    question_id: str
    required_facts_response: str | None
    stability: str
    valid_until: str | None
    annotated_at: datetime
    annotator: str | None = None

    @property
    def responses(self) -> dict[str, Any]:
        return {
            "required_facts": (
                [MagicMock(value=self.required_facts_response)]
                if self.required_facts_response is not None
                else []
            ),
            "ground_truth_stability": [MagicMock(value=self.stability)],
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return {"ground_truth_valid_until": self.valid_until} if self.valid_until else {}

    @property
    def updated_at(self) -> datetime:
        return self.annotated_at


def _mk_record(
    qid: str,
    facts: str | None,
    stability: str = "stable",
    valid_until: str | None = None,
    days_ago: int = 0,
) -> _FakeRecord:
    from datetime import timedelta

    return _FakeRecord(
        question_id=qid,
        required_facts_response=facts,
        stability=stability,
        valid_until=valid_until,
        annotated_at=datetime.now(tz=timezone.utc) - timedelta(days=days_ago),
    )


class TestGroundTruthCachePrime:
    @pytest.mark.asyncio
    async def test_prime_picks_most_recent_populated_record(self):
        """Two annotations for the same question — prefer the newer one."""
        older = _mk_record("q-001", "- old fact", days_ago=10)
        newer = _mk_record("q-001", "- new fact", days_ago=1)

        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[older, newer])  # type: ignore[method-assign]

        await cache.prime(["q-001"])
        facts = cache.get("q-001")
        assert facts is not None
        assert facts.facts == ["new fact"]

    @pytest.mark.asyncio
    async def test_prime_skips_empty_required_facts(self):
        """A record with no required_facts response is treated as no ground truth."""
        unannotated = _mk_record("q-001", None)
        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[unannotated])  # type: ignore[method-assign]

        await cache.prime(["q-001"])
        assert cache.get("q-001") is None

    @pytest.mark.asyncio
    async def test_prime_honors_valid_until(self):
        """time_bound facts past valid_until are excluded."""
        from datetime import timedelta

        yesterday = (date.today() - timedelta(days=1)).isoformat()
        expired = _mk_record("q-001", "- stale fact", stability="time_bound", valid_until=yesterday)

        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[expired])  # type: ignore[method-assign]

        await cache.prime(["q-001"])
        assert cache.get("q-001") is None

    @pytest.mark.asyncio
    async def test_prime_keeps_unexpired_time_bound(self):
        from datetime import timedelta

        future = (date.today() + timedelta(days=30)).isoformat()
        r = _mk_record("q-001", "- fresh fact", stability="time_bound", valid_until=future)

        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[r])  # type: ignore[method-assign]

        await cache.prime(["q-001"])
        facts = cache.get("q-001")
        assert facts is not None
        assert facts.facts == ["fresh fact"]

    @pytest.mark.asyncio
    async def test_prime_skips_not_applicable_and_not_reviewed(self):
        na = _mk_record("q-001", "- irrelevant", stability="not_applicable")
        nr = _mk_record("q-002", "- irrelevant", stability="not_reviewed")

        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[na, nr])  # type: ignore[method-assign]

        await cache.prime(["q-001", "q-002"])
        assert cache.get("q-001") is None
        assert cache.get("q-002") is None

    @pytest.mark.asyncio
    async def test_prime_drops_time_bound_with_missing_valid_until(self):
        """Fail-closed: time_bound without valid_until is invalid ground truth."""
        r = _mk_record("q-001", "- fact", stability="time_bound", valid_until=None)
        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[r])  # type: ignore[method-assign]

        await cache.prime(["q-001"])
        assert cache.get("q-001") is None

    @pytest.mark.asyncio
    async def test_prime_drops_time_bound_with_malformed_valid_until(self):
        """Fail-closed: malformed valid_until is treated as invalid, not ignored."""
        r = _mk_record("q-001", "- fact", stability="time_bound", valid_until="not-a-date")
        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[r])  # type: ignore[method-assign]

        await cache.prime(["q-001"])
        assert cache.get("q-001") is None

    @pytest.mark.asyncio
    async def test_recency_cascade_falls_back_to_inserted_at(self):
        """When updated_at is absent, inserted_at should rank records."""
        from datetime import timedelta

        @dataclass
        class _NoUpdatedAt:
            """Fake record with only inserted_at (no updated_at attribute)."""

            question_id: str
            required_facts_response: str
            stability: str = "stable"
            inserted_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

            @property
            def responses(self) -> dict[str, Any]:
                return {
                    "required_facts": [MagicMock(value=self.required_facts_response)],
                    "ground_truth_stability": [MagicMock(value=self.stability)],
                }

            @property
            def metadata(self) -> dict[str, Any]:
                return {}

        older = _NoUpdatedAt(
            question_id="q-001",
            required_facts_response="- old fact",
            inserted_at=datetime.now(tz=timezone.utc) - timedelta(days=10),
        )
        newer = _NoUpdatedAt(
            question_id="q-001",
            required_facts_response="- new fact",
            inserted_at=datetime.now(tz=timezone.utc) - timedelta(days=1),
        )

        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[older, newer])  # type: ignore[method-assign]
        await cache.prime(["q-001"])
        facts = cache.get("q-001")
        assert facts is not None
        assert facts.facts == ["new fact"]

    @pytest.mark.asyncio
    async def test_recency_cascade_falls_back_to_id_ordering(self):
        """When updated_at and inserted_at are absent, id ordering is used."""

        @dataclass
        class _OnlyId:
            question_id: str
            required_facts_response: str
            id: str
            stability: str = "stable"

            @property
            def responses(self) -> dict[str, Any]:
                return {
                    "required_facts": [MagicMock(value=self.required_facts_response)],
                    "ground_truth_stability": [MagicMock(value=self.stability)],
                }

            @property
            def metadata(self) -> dict[str, Any]:
                return {}

        # ULID-like sortable ids — higher lexicographic value is newer.
        older = _OnlyId(question_id="q-001", required_facts_response="- old", id="01AAAA")
        newer = _OnlyId(question_id="q-001", required_facts_response="- new", id="01ZZZZ")

        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")
        cache._fetch_records = MagicMock(return_value=[older, newer])  # type: ignore[method-assign]
        await cache.prime(["q-001"])
        facts = cache.get("q-001")
        assert facts is not None
        assert facts.facts == ["new"]

    @pytest.mark.asyncio
    async def test_prime_tolerates_argilla_connection_failure(self):
        """Connection error → empty cache, no exception."""
        cache = GroundTruthCache(argilla_url="x", argilla_api_key="y", dataset_name="z")

        def explode(_ids):
            raise ConnectionError("argilla unreachable")

        cache._fetch_records = explode  # type: ignore[method-assign]

        # Must not raise
        await cache.prime(["q-001"])
        assert cache.get("q-001") is None
```

- [ ] **Step 2: Run to confirm they fail**

Run:
```bash
uv run pytest tests/test_eval_ground_truth.py -v
```
Expected: `ModuleNotFoundError: src.eval.argilla_ground_truth`.

### Task 2.2: Implement the ground-truth cache module

**Files:**
- Create: `src/eval/argilla_ground_truth.py`

- [ ] **Step 1: Write the implementation**

```python
"""Fetch annotator-authored ground truth ('required facts') from Argilla.

Used by the eval scorer to feed reference facts to the judge. Keeps the eval
pipeline from depending on Argilla being reachable — connection failures
degrade silently to 'no ground truth available'.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

logger = logging.getLogger(__name__)

_BULLET_RE = re.compile(r"^\s*[-*]\s*(.+?)\s*$")


def parse_facts_markdown(text: str | None) -> list[str]:
    """Parse a markdown bulleted list into a flat list of atomic facts.

    One fact per bullet. Non-bullet lines are ignored. Empty input returns [].
    """
    if not text:
        return []
    facts: list[str] = []
    for line in text.splitlines():
        m = _BULLET_RE.match(line)
        if m:
            fact = m.group(1).strip()
            if fact:
                facts.append(fact)
    return facts


@dataclass(frozen=True)
class RequiredFacts:
    question_id: str
    facts: list[str]
    stability: str
    valid_until: date | None
    annotator: str | None
    # Recency key from the cascade: (tier, value). Higher tuple > more recent.
    recency_key: tuple[int, Any]


class GroundTruthCache:
    """Fetches the most recent authored required_facts per question.

    Usage:
        cache = GroundTruthCache(url, key, "eval-production")
        await cache.prime(all_question_ids)
        facts = cache.get("q-001")  # RequiredFacts | None
    """

    def __init__(
        self,
        argilla_url: str,
        argilla_api_key: str,
        dataset_name: str,
    ) -> None:
        self._argilla_url = argilla_url
        self._argilla_api_key = argilla_api_key
        self._dataset_name = dataset_name
        self._cache: dict[str, RequiredFacts] = {}
        self._primed = False

    async def prime(self, question_ids: list[str]) -> None:
        """Fetch all relevant records once and build the per-question cache.

        Silently degrades to an empty cache on any fetch error.
        """
        if self._primed:
            return
        self._primed = True

        try:
            records = self._fetch_records(question_ids)
        except Exception as exc:  # noqa: BLE001 — defensive, never fail the run
            logger.warning(
                "ground truth fetch failed (dataset=%s): %s", self._dataset_name, exc
            )
            return

        today = date.today()
        best: dict[str, RequiredFacts] = {}
        for record in records:
            qid = getattr(record, "question_id", None)
            if not qid or qid not in question_ids:
                continue

            stability = _response_value(record, "ground_truth_stability") or "not_reviewed"
            if stability not in ("stable", "time_bound"):
                continue

            facts_text = _response_value(record, "required_facts")
            facts = parse_facts_markdown(facts_text)
            if not facts:
                continue

            valid_until_raw = _metadata_value(record, "ground_truth_valid_until")
            valid_until = _parse_iso_date(valid_until_raw)
            # Fail-closed rule: time_bound records MUST have a valid valid_until.
            # Missing or malformed → drop, don't assume indefinite validity.
            if stability == "time_bound":
                if valid_until is None:
                    logger.warning(
                        "dropping time_bound record for %s: missing/malformed "
                        "ground_truth_valid_until=%r",
                        qid,
                        valid_until_raw,
                    )
                    continue
                if valid_until < today:
                    continue

            recency_key = _pick_recency_key(record)
            current = RequiredFacts(
                question_id=qid,
                facts=facts,
                stability=stability,
                valid_until=valid_until,
                annotator=getattr(record, "annotator", None),
                recency_key=recency_key,
            )

            existing = best.get(qid)
            if existing is None or current.recency_key > existing.recency_key:
                best[qid] = current

        self._cache = best
        logger.info(
            "ground truth primed: %d/%d questions have required_facts",
            len(self._cache),
            len(question_ids),
        )

    def get(self, question_id: str) -> RequiredFacts | None:
        return self._cache.get(question_id)

    # Overridden in tests.
    def _fetch_records(self, question_ids: list[str]) -> list[Any]:
        try:
            import argilla as rg
        except ImportError:
            logger.error("argilla package not installed")
            return []

        client = rg.Argilla(api_url=self._argilla_url, api_key=self._argilla_api_key)
        dataset = client.datasets(name=self._dataset_name)
        if dataset is None:
            logger.warning("argilla dataset %r not found", self._dataset_name)
            return []

        # NOTE: Argilla 2.x query shape verification is part of Phase 2 (see spec
        # §Open Questions). If the SDK doesn't support IN-queries on question_id,
        # fall through to iterating all records and filtering client-side. Batch
        # size is small (<200 records), so this is acceptable.
        return list(dataset.records)


def _response_value(record: Any, name: str) -> str | None:
    """Extract the most-recent response value for a named question.

    Argilla's record.responses is a dict of question_name -> list of Response;
    we pick the last one. Returns None if absent.
    """
    responses = getattr(record, "responses", None) or {}
    entries = responses.get(name) or []
    if not entries:
        return None
    value = getattr(entries[-1], "value", None)
    return str(value) if value is not None else None


def _metadata_value(record: Any, name: str) -> str | None:
    metadata = getattr(record, "metadata", None) or {}
    raw = metadata.get(name)
    return str(raw) if raw else None


def _parse_iso_date(raw: str | None) -> date | None:
    if not raw:
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        logger.warning("could not parse ground_truth_valid_until=%r", raw)
        return None


def _pick_recency_key(record: Any) -> tuple[int, datetime | str]:
    """Pick a sortable recency key using the cascade from the spec.

    Returns a (tier, value) tuple so Python tuple-ordering first compares tiers
    (higher tier beats lower — see inversion below) then values within a tier.
    The cascade:

      Tier 3 (highest): ``updated_at`` datetime
      Tier 2: ``inserted_at`` datetime
      Tier 1: sortable ``id`` string (lexicographic; Argilla commonly uses ULIDs)
      Tier 0 (lowest): a fixed sentinel datetime, with a logged warning

    Tier numbers are returned as positive so *higher* is more recent. The
    caller uses ``max(records, key=_pick_recency_key)`` for selection.
    Phase 2 verifies which tiers the Argilla SDK exposes; any tier that is
    absent or non-sensible falls through to the next.
    """
    updated_at = getattr(record, "updated_at", None)
    if isinstance(updated_at, datetime):
        return (3, updated_at)

    inserted_at = getattr(record, "inserted_at", None)
    if isinstance(inserted_at, datetime):
        return (2, inserted_at)

    rec_id = getattr(record, "id", None)
    if isinstance(rec_id, str) and rec_id:
        logger.debug(
            "no updated_at/inserted_at on record %r; using id-based ordering",
            rec_id,
        )
        return (1, rec_id)

    logger.warning(
        "record has no updated_at, inserted_at, or usable id; cascade fell "
        "through to sentinel — recency comparison will not be meaningful"
    )
    return (0, datetime.fromtimestamp(0))
```

- [ ] **Step 2: Run the tests**

Run:
```bash
uv run pytest tests/test_eval_ground_truth.py -v
```
Expected: all tests PASS.

- [ ] **Step 3: Commit**

Run:
```bash
git add src/eval/argilla_ground_truth.py tests/test_eval_ground_truth.py
git commit -m "feat(eval): add GroundTruthCache to fetch required_facts from argilla"
```

---

## Phase 3 — Judge Prompt and Scorer Integration

### Task 3.1: Write failing tests for `build_judge_prompt` with required facts

**Files:**
- Modify: `tests/test_eval_judge.py` (create if it doesn't exist, otherwise append)

- [ ] **Step 1: Check whether the file exists**

Run:
```bash
ls tests/test_eval_judge.py 2>/dev/null || echo "does not exist"
```

If the file does not exist, create it with a minimal header:

```python
"""Tests for the eval judge prompt construction and response parsing."""

from src.eval.rubric import build_judge_prompt
from src.eval.judge import parse_judge_response
```

If it exists, append to it.

- [ ] **Step 2: Add prompt-with-facts test cases**

Append (or include in the new file):

```python
def test_prompt_includes_required_facts_section_when_given():
    prompt = build_judge_prompt(
        query="What GPUs are on Delta?",
        answer="Delta has A100 GPUs.",
        required_facts=["Delta has A100 GPUs", "Answer should cite docs"],
    )
    assert "## Required Facts" in prompt
    assert "- Delta has A100 GPUs" in prompt
    assert "- Answer should cite docs" in prompt
    # Updated correctness guidance visible to the judge.
    assert "Score 5 only if every required fact is accurately represented" in prompt
    assert "required_facts_coverage" in prompt


def test_prompt_omits_required_facts_section_when_none():
    prompt = build_judge_prompt(
        query="What GPUs are on Delta?",
        answer="Delta has A100 GPUs.",
        required_facts=None,
    )
    assert "## Required Facts" not in prompt
    assert "required_facts_coverage" not in prompt


def test_prompt_omits_required_facts_section_when_empty_list():
    prompt = build_judge_prompt(
        query="Q",
        answer="A",
        required_facts=[],
    )
    assert "## Required Facts" not in prompt
```

- [ ] **Step 3: Add response-parsing test cases**

Append:

```python
def test_parse_judge_response_with_required_facts_coverage():
    raw = '''{
      "correctness": {
        "score": 4,
        "justification": "Covers A100 fact; missing citation.",
        "required_facts_coverage": [
          {"fact": "Delta has A100 GPUs", "covered": true, "note": "mentioned A100"},
          {"fact": "Answer should cite docs", "covered": false, "note": "no URL"}
        ]
      },
      "completeness": {"score": 4, "justification": ""},
      "relevance": {"score": 5, "justification": ""},
      "citation_quality": {"score": 2, "justification": ""},
      "hedging": {"score": 4, "justification": ""}
    }'''
    result = parse_judge_response(raw)
    assert result is not None
    assert result.scores["correctness"] == 4
    assert result.required_facts_coverage is not None
    assert len(result.required_facts_coverage) == 2
    assert result.required_facts_coverage[0]["covered"] is True
    assert result.required_facts_coverage[1]["covered"] is False


def test_parse_judge_response_without_required_facts_coverage_is_backward_compat():
    raw = '''{
      "correctness": {"score": 4, "justification": "ok"},
      "completeness": {"score": 4, "justification": ""},
      "relevance": {"score": 5, "justification": ""},
      "citation_quality": {"score": 4, "justification": ""},
      "hedging": {"score": 4, "justification": ""}
    }'''
    result = parse_judge_response(raw)
    assert result is not None
    assert result.scores["correctness"] == 4
    assert result.required_facts_coverage is None
```

- [ ] **Step 4: Run to confirm they fail**

Run:
```bash
uv run pytest tests/test_eval_judge.py -v
```
Expected: the three new `prompt_*` tests fail with `TypeError: build_judge_prompt() got an unexpected keyword argument 'required_facts'`; the two parse tests fail with `AttributeError: 'JudgeResult' object has no attribute 'required_facts_coverage'`.

### Task 3.2: Extend `build_judge_prompt`

**Files:**
- Modify: `src/eval/rubric.py`

- [ ] **Step 1: Update signature and prompt body**

Replace the existing `build_judge_prompt` with:

```python
def build_judge_prompt(
    query: str,
    answer: str,
    rag_context: str | None = None,
    tool_results: str | None = None,
    node_trace: str | None = None,
    required_facts: list[str] | None = None,
) -> str:
    """Build the LLM judge prompt with the rubric and context.

    When ``required_facts`` is non-empty, prepend a "Required Facts" section
    with tightened correctness scoring guidance; the judge must emit a
    per-fact coverage list in its correctness response.
    """
    rubric_text = "\n".join(
        f"- **{d.name}** (1-5): {d.description}\n  1 = {d.low}\n  5 = {d.high}" for d in DIMENSIONS
    )

    context_sections = []
    if rag_context:
        context_sections.append(f"## RAG Documents Retrieved\n{rag_context}")
    if tool_results:
        context_sections.append(f"## Tool Results\n{tool_results}")
    if node_trace:
        context_sections.append(f"## Agent Decision Trace\n{node_trace}")
    context_text = "\n\n".join(context_sections) if context_sections else "No context available."

    # Required facts section — only emitted when facts are present.
    facts_section = ""
    coverage_json_hint = ""
    if required_facts:
        facts_list = "\n".join(f"- {f}" for f in required_facts)
        facts_section = (
            "## Required Facts\n\n"
            "A correct answer to this question must accurately represent all of the following:\n\n"
            f"{facts_list}\n\n"
            "When scoring correctness:\n"
            "- Score 5 only if every required fact is accurately represented.\n"
            "- Score 1-2 if any required fact is missing or contradicted.\n"
            "- In your correctness justification, note each required fact and "
            "whether the answer covered it.\n\n"
            "Include a `required_facts_coverage` array on the correctness dimension "
            "with one entry per fact.\n\n"
        )
        coverage_json_hint = (
            ',\n    "required_facts_coverage": [\n'
            '      {"fact": "<fact 1>", "covered": true|false, "note": "<brief>"},\n'
            '      ...\n    ]'
        )

    return f"""You are evaluating the quality of an AI agent's answer to a user question.

## Scoring Rubric

Score each dimension from 1 (worst) to 5 (best):

{rubric_text}

## Important

- Judge whether the agent accurately represented the information it HAD ACCESS TO.
- If the source documents contain outdated information and the agent faithfully reported it, that is CORRECT (score 5 on correctness). Data quality is not the agent's fault.
- If the agent added information not in the sources, that is a hallucination (score 1-2 on correctness).
- CRITICAL: Tool Results are LIVE DATA from real-time APIs and are MORE CURRENT than RAG Documents. When tool results and RAG documents conflict (e.g., RAG says "there are upcoming webinars" but tool results show total: 0), the agent is CORRECT to trust the tool results. Score the agent based on whether it accurately represented the tool results, not the stale RAG data.
- If tool results show 0 items/no results for something the user asked about, and the agent correctly reports that nothing was found, that is CORRECT — even if RAG documents suggest otherwise.

{facts_section}## User Question

{query}

## Agent Answer

{answer}

## Context the Agent Had Access To

{context_text}

## Your Response

Return a JSON object with this exact structure (no other text):
```json
{{
  "correctness": {{"score": <1-5>, "justification": "<brief explanation>"{coverage_json_hint}}},
  "completeness": {{"score": <1-5>, "justification": "<brief explanation>"}},
  "relevance": {{"score": <1-5>, "justification": "<brief explanation>"}},
  "citation_quality": {{"score": <1-5>, "justification": "<brief explanation>"}},
  "hedging": {{"score": <1-5>, "justification": "<brief explanation>"}}
}}
```"""
```

- [ ] **Step 2: Run the prompt tests**

Run:
```bash
uv run pytest tests/test_eval_judge.py -k "prompt_" -v
```
Expected: the three prompt tests PASS. Response-parsing tests still fail (fixed next task).

- [ ] **Step 3: Commit**

Run:
```bash
git add src/eval/rubric.py tests/test_eval_judge.py
git commit -m "feat(eval): judge prompt accepts required_facts and emits coverage guidance"
```

### Task 3.3: Extend `JudgeResult` and `parse_judge_response`

**Files:**
- Modify: `src/eval/judge.py`

- [ ] **Step 1: Update `JudgeResult` and `parse_judge_response`**

Replace the `JudgeResult` dataclass and `parse_judge_response` function with:

```python
@dataclass
class JudgeResult:
    scores: dict[str, int]
    justifications: dict[str, str]
    composite: float
    required_facts_coverage: list[dict[str, Any]] | None = None


def parse_judge_response(raw: str) -> JudgeResult | None:
    """Parse judge LLM response into JudgeResult. Returns None if malformed."""
    cleaned = raw.strip()
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Judge returned invalid JSON")
        return None

    scores: dict[str, int] = {}
    justifications: dict[str, str] = {}
    required_facts_coverage: list[dict[str, Any]] | None = None

    for name in DIMENSION_NAMES:
        if name not in data:
            logger.warning(f"Judge response missing dimension: {name}")
            return None
        entry = data[name]
        if not isinstance(entry, dict) or "score" not in entry:
            logger.warning(f"Judge response malformed for dimension: {name}")
            return None
        score = entry["score"]
        if not isinstance(score, int) or score < 1 or score > 5:
            logger.warning(f"Judge score out of range for {name}: {score}")
            return None
        scores[name] = score
        justifications[name] = entry.get("justification", "")

        # Capture per-fact coverage when the correctness entry provides it.
        if name == "correctness":
            coverage = entry.get("required_facts_coverage")
            if isinstance(coverage, list):
                required_facts_coverage = coverage

    return JudgeResult(
        scores=scores,
        justifications=justifications,
        composite=compute_composite(scores),
        required_facts_coverage=required_facts_coverage,
    )
```

Add `Any` to the existing `typing` import at the top if not already present:

```python
from typing import Any
```

- [ ] **Step 2: Run parse tests**

Run:
```bash
uv run pytest tests/test_eval_judge.py -v
```
Expected: all judge tests PASS.

- [ ] **Step 3: Commit**

Run:
```bash
git add src/eval/judge.py
git commit -m "feat(eval): JudgeResult carries optional required_facts_coverage"
```

### Task 3.4: Thread `required_facts` through `Judge.score`

**Files:**
- Modify: `src/eval/judge.py`

- [ ] **Step 1: Update `Judge.score` signature and prompt build**

Replace the existing `Judge.score` method with:

```python
    async def score(
        self,
        query: str,
        answer: str,
        rag_context: str | None = None,
        tool_results: str | None = None,
        node_trace: str | None = None,
        required_facts: list[str] | None = None,
    ) -> JudgeResult | None:
        prompt = build_judge_prompt(
            query=query,
            answer=answer,
            rag_context=rag_context,
            tool_results=tool_results,
            node_trace=node_trace,
            required_facts=required_facts,
        )

        for attempt in range(2):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=700,
                )
                raw = response.choices[0].message.content or ""
                result = parse_judge_response(raw)
                if result is not None:
                    return result
                if attempt == 0:
                    logger.warning("Judge parse failed, retrying with stricter prompt")
                    prompt += "\n\nIMPORTANT: Return ONLY the JSON object. No other text."
            except Exception as e:
                logger.error(f"Judge LLM call failed (attempt {attempt + 1}): {e}")

        logger.error("Judge failed after 2 attempts")
        return None
```

Note: `max_tokens` bumped from 500 to 700 to accommodate the coverage array.

- [ ] **Step 2: Run the full judge test suite**

Run:
```bash
uv run pytest tests/test_eval_judge.py -v
```
Expected: PASS.

- [ ] **Step 3: Commit**

Run:
```bash
git add src/eval/judge.py
git commit -m "feat(eval): Judge.score forwards required_facts to prompt"
```

### Task 3.5: Wire `GroundTruthCache` into the eval scorer

**Files:**
- Modify: `src/eval/scorer.py`

- [ ] **Step 1: Import the cache and prime it at run start**

Near the top of `src/eval/scorer.py`, add the import:

```python
from .argilla_ground_truth import GroundTruthCache
```

After the `judge = Judge(...)` line (currently around line 47), insert:

```python
    # Prime ground-truth cache before scoring. Failures degrade silently —
    # missing ground truth just means the judge behaves as before.
    gt_cache = GroundTruthCache(
        argilla_url=settings.ARGILLA_URL,
        argilla_api_key=settings.ARGILLA_API_KEY,
        dataset_name=settings.ARGILLA_EVAL_DATASET,
    )
    await gt_cache.prime([q.id for q in questions])
```

- [ ] **Step 2: Pass facts into each judge call**

Find the `judge_result = await judge.score(...)` block in the question loop (currently around lines 78–84). Replace it with:

```python
        gt = gt_cache.get(q.id)
        judge_result = await judge.score(
            query=q.question,
            answer=result.answer,
            rag_context=result.rag_context,
            tool_results=result.tool_results,
            node_trace=result.node_trace,
            required_facts=gt.facts if gt else None,
        )
```

- [ ] **Step 3: Thread per-fact coverage into the stored justifications**

This preserves per-fact detail without changing the DB schema. Find the `db.add_score(...)` call that records judge results (currently around lines 101–120). Replace the `justifications=judge_result.justifications` line with:

```python
            justifications=_augment_justifications(
                judge_result.justifications,
                judge_result.required_facts_coverage,
            ),
```

Then add this helper above `run_eval` at module scope:

```python
def _augment_justifications(
    justifications: dict[str, str],
    coverage: list[dict[str, Any]] | None,
) -> dict[str, str]:
    """Attach per-fact coverage to the correctness justification for auditability.

    Stored as a ``required_facts_coverage`` key in the justifications dict.
    The DB ``justifications`` column is an open map, so this is a schema-free
    extension.
    """
    if not coverage:
        return justifications
    augmented = dict(justifications)
    import json
    augmented["required_facts_coverage"] = json.dumps(coverage, default=str)
    return augmented
```

- [ ] **Step 4: Run full eval test suite**

Run:
```bash
uv run pytest tests/ -k "eval" -v
```
Expected: all green. `test_eval_integration.py` may need updating if it mocks the judge — check the output.

- [ ] **Step 5: Commit**

Run:
```bash
git add src/eval/scorer.py
git commit -m "feat(eval): prime ground-truth cache and feed required_facts to judge"
```

### Task 3.6: End-to-end smoke test

**Files:** none; this is a manual verification step.

- [ ] **Step 1: Run the combined battery against current prod**

With `ARGILLA_URL`, `ARGILLA_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL` set in `.env`:

```bash
uv run python -m src.eval run \
    --questions eval/questions/combined_battery.json
```
Expected: eval completes; logs include `ground truth primed: 0/N questions have required_facts` (no annotations yet — expected).

- [ ] **Step 2: Manually author required_facts for one question in Argilla**

In the Argilla UI, find the record for `comb-006` ("Which ACCESS resources have A100 GPUs?"). In the annotation panel:

- **required_facts**:
  ```
  - Delta has NVIDIA A100 GPUs
  - Answer should cite docs.access-ci.org or similar
  ```
- **ground_truth_stability**: `stable`
- **decision**: whatever fits the answer you're reviewing
- Submit the response.

- [ ] **Step 3: Re-run the same battery, confirm the cache picks up the fact**

Run:
```bash
uv run python -m src.eval run \
    --questions eval/questions/combined_battery.json
```
Expected: logs include `ground truth primed: 1/N questions have required_facts`. Confirm by querying the eval DB or by observing the log line.

- [ ] **Step 4: Verify the judge prompt actually included the facts**

Query the eval DB (or check captured prompts if they're logged) for the score record of `comb-006` in the newest run. The `justifications["correctness"]` field should reference the A100 fact by name, and `justifications["required_facts_coverage"]` should contain a JSON array with at least the A100 entry.

- [ ] **Step 5: Commit verification note**

Run:
```bash
git commit --allow-empty -m "chore(eval): smoke-test ground-truth-aware judge on comb-006"
```

---

## Phase 4 — Bootstrap Annotation (human work)

Not coded. Execution bullets are:

- Run the current production battery to populate Argilla with records to annotate:
  ```bash
  uv run python -m src.eval run --questions eval/questions/combined_battery.json --push-argilla
  uv run python -m src.eval run --questions eval/questions/friendly_battery.json --push-argilla
  uv run python -m src.eval run --questions eval/questions/mcp_coverage_battery.json --push-argilla
  uv run python -m src.eval run --questions eval/questions/real_user_battery.json --push-argilla
  ```
- Open Argilla. Work through records in priority order:
  1. Static questions (`expected_type == "static"`) — most leverage.
  2. Combined questions with stable doc components.
  3. Dynamic questions only if trivially groundable; most get `not_applicable`.
- Budget: one 2-3 hour focused session. Expected yield: ~60-80% of questions annotated; remainder `not_applicable` / `not_reviewed`.
- No commit at the end of Phase 4 — annotations live in Argilla, not git.

---

## Phase 5 — Validation (replay, not re-run)

The agent is NOT re-invoked during validation. Re-running the battery would regenerate answers, mixing model/output variance with the rubric change we're trying to measure. Instead: pick a baseline run, re-score its stored answers with the new ground-truth-aware judge, compare judge-to-judge on identical inputs.

### Task 5.1: Add replay mode to the scorer

**Files:**
- Modify: `src/eval/scorer.py`
- Modify: `src/eval/__main__.py` (add a `replay` subcommand)
- Create: `tests/test_eval_replay.py`

- [ ] **Step 1: Write the failing test for replay**

Create `tests/test_eval_replay.py`:

```python
"""Test the replay scoring path — re-judge stored answers without re-invoking the agent."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.eval.scorer import replay_run_with_judge


@pytest.mark.asyncio
async def test_replay_uses_stored_answers_and_does_not_invoke_agent():
    """Replay reads answers from the baseline run and scores them with a fresh judge."""
    baseline_run_id = "test-baseline-run"

    with patch("src.eval.scorer.EvalDB") as mock_db_cls, \
            patch("src.eval.scorer.Judge") as mock_judge_cls, \
            patch("src.eval.scorer.run_question") as mock_run_question:
        mock_db = mock_db_cls.return_value
        mock_db.get_scores_for_run.return_value = [
            _fake_score_row(qid="q-001", answer="Delta has A100 GPUs"),
            _fake_score_row(qid="q-002", answer="Bridges-2 has V100 GPUs"),
        ]
        mock_db.create_run.return_value.id = "test-replay-run"

        mock_judge = mock_judge_cls.return_value
        mock_judge.score = AsyncMock(return_value=_fake_judge_result())

        result = await replay_run_with_judge(
            baseline_run_id=baseline_run_id,
            database_url="postgresql://test",
        )

        # Agent must NOT be invoked during replay.
        mock_run_question.assert_not_called()

        # Judge is called once per baseline answer.
        assert mock_judge.score.await_count == 2

        assert result["replay_run_id"] == "test-replay-run"
        assert result["baseline_run_id"] == baseline_run_id


def _fake_score_row(qid: str, answer: str):
    from types import SimpleNamespace

    return SimpleNamespace(
        question_id=qid,
        question_text=f"question for {qid}",
        answer_text=answer,
        context={
            "rag_context": None,
            "tool_results": None,
            "node_trace": None,
        },
        source="judge",
    )


def _fake_judge_result():
    from src.eval.judge import JudgeResult

    return JudgeResult(
        scores={
            "correctness": 5,
            "completeness": 4,
            "relevance": 5,
            "citation_quality": 4,
            "hedging": 4,
        },
        justifications={"correctness": "ok", "completeness": "ok", "relevance": "ok",
                        "citation_quality": "ok", "hedging": "ok"},
        composite=4.5,
        required_facts_coverage=None,
    )
```

- [ ] **Step 2: Run to confirm it fails**

Run:
```bash
uv run pytest tests/test_eval_replay.py -v
```
Expected: `ImportError: cannot import name 'replay_run_with_judge' from 'src.eval.scorer'`.

- [ ] **Step 3: Add `replay_run_with_judge` to the scorer**

Append to `src/eval/scorer.py`:

```python
async def replay_run_with_judge(
    baseline_run_id: str,
    database_url: str | None = None,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Replay a baseline run's stored answers through a fresh judge.

    Does NOT invoke the agent. Reads ``answer_text`` and context from the
    eval DB for ``baseline_run_id`` and scores each answer via the current
    (ground-truth-aware) judge. Writes results as a new replay run.

    Used by Phase 5 validation to isolate rubric deltas from agent-side
    drift.
    """
    db_url = database_url or settings.DATABASE_URL
    j_base = judge_base_url or settings.EVAL_JUDGE_BASE_URL or None
    j_key = judge_api_key or settings.EVAL_JUDGE_API_KEY or settings.OPENAI_API_KEY
    j_model = judge_model or settings.EVAL_JUDGE_MODEL

    db = EvalDB(db_url)
    baseline_scores = [
        s for s in db.get_scores_for_run(baseline_run_id) if s.source == "judge"
    ]
    if not baseline_scores:
        logger.warning("baseline run %s has no judge-sourced scores", baseline_run_id)
        return {
            "baseline_run_id": baseline_run_id,
            "replay_run_id": None,
            "replayed": 0,
        }

    gt_cache = GroundTruthCache(
        argilla_url=settings.ARGILLA_URL,
        argilla_api_key=settings.ARGILLA_API_KEY,
        dataset_name=settings.ARGILLA_EVAL_DATASET,
    )
    await gt_cache.prime([str(s.question_id) for s in baseline_scores])

    judge = Judge(base_url=j_base, api_key=j_key, model=j_model)

    replay_run = db.create_run(
        run_type="replay",
        agent_commit=None,
        agent_branch=None,
        tool_catalog={"replay_of": baseline_run_id},
        llm_model=None,
        judge_model=j_model,
        question_set=f"replay:{baseline_run_id}",
        question_count=len(baseline_scores),
    )

    all_scores: list[dict[str, int]] = []
    for score in baseline_scores:
        qid = str(score.question_id)
        ctx = score.context or {}
        gt = gt_cache.get(qid)
        judge_result = await judge.score(
            query=str(score.question_text or ""),
            answer=str(score.answer_text or ""),
            rag_context=ctx.get("rag_context"),
            tool_results=ctx.get("tool_results"),
            node_trace=ctx.get("node_trace"),
            required_facts=gt.facts if gt else None,
        )

        if judge_result is None:
            db.add_score(
                run_id=replay_run.id,
                question_id=qid,
                source="judge_error",
                question_text=str(score.question_text or ""),
                answer_text=str(score.answer_text or ""),
                context=ctx,
                justifications={"error": "replay judge failed"},
            )
            continue

        db.add_score(
            run_id=replay_run.id,
            question_id=qid,
            source="judge",
            question_text=str(score.question_text or ""),
            answer_text=str(score.answer_text or ""),
            context=ctx,
            context_completeness="replay",
            correctness=judge_result.scores["correctness"],
            completeness=judge_result.scores["completeness"],
            relevance=judge_result.scores["relevance"],
            citation_quality=judge_result.scores["citation_quality"],
            hedging=judge_result.scores["hedging"],
            composite_score=judge_result.composite,
            justifications=_augment_justifications(
                judge_result.justifications,
                judge_result.required_facts_coverage,
            ),
        )
        all_scores.append(judge_result.scores)

    avg_composite = (
        sum(compute_composite(s) for s in all_scores) / len(all_scores)
        if all_scores
        else 0.0
    )
    db.update_run_summary(
        str(replay_run.id),
        {n: sum(s[n] for s in all_scores) / len(all_scores) for n in DIMENSION_NAMES}
        if all_scores else {},
        avg_composite,
    )

    logger.info(
        "replay complete: baseline=%s replay=%s scored=%d composite=%.2f",
        baseline_run_id, replay_run.id, len(all_scores), avg_composite,
    )

    return {
        "baseline_run_id": baseline_run_id,
        "replay_run_id": str(replay_run.id),
        "replayed": len(all_scores),
        "composite_score": round(avg_composite, 2),
    }
```

- [ ] **Step 4: Run the replay test**

Run:
```bash
uv run pytest tests/test_eval_replay.py -v
```
Expected: PASS.

- [ ] **Step 5: Add CLI subcommand**

Modify `src/eval/__main__.py` to add a `replay` subcommand. Find the `compare_parser = subparsers.add_parser("compare", ...)` block and add after it:

```python
    replay_parser = subparsers.add_parser(
        "replay",
        help="Re-judge an existing run's stored answers with the current judge",
    )
    replay_parser.add_argument(
        "--run-id",
        required=True,
        help="Baseline run ID whose answers should be replayed",
    )
```

Add a handler function above `main()`:

```python
def _handle_replay(args: argparse.Namespace) -> None:
    import asyncio
    from .scorer import replay_run_with_judge

    result = asyncio.run(replay_run_with_judge(baseline_run_id=args.run_id))
    import json
    print(json.dumps(result, indent=2))
```

Register it in the `handlers` dict near the bottom of `main()`:

```python
        "replay": _handle_replay,
```

- [ ] **Step 6: Commit**

Run:
```bash
git add src/eval/scorer.py src/eval/__main__.py tests/test_eval_replay.py
git commit -m "feat(eval): add replay_run_with_judge for Phase 5 validation"
```

### Task 5.2: Execute the replay against a baseline run

**Files:** none; execution step.

- [ ] **Step 1: Identify baseline run**

Pick the most recent production eval run that has `required_facts` annotated on at least some of its questions (from Phase 4). Call its run ID `$RUN_BASELINE`.

```bash
# Inspect the DB for recent runs — exact query depends on EvalDB schema;
# adjust as needed.
uv run python -c "
from src.eval.db import EvalDB
from src.config import settings
db = EvalDB(settings.DATABASE_URL)
for run in db.list_recent_runs(limit=10):
    print(run.id, run.started_at, run.question_set)
"
```
If `list_recent_runs` doesn't exist in `EvalDB`, query directly via psql or add a helper method as part of this step.

- [ ] **Step 2: Run the replay**

```bash
uv run python -m src.eval replay --run-id $RUN_BASELINE 2>&1 | tee /tmp/replay.log
```
Expected: output JSON with `baseline_run_id`, `replay_run_id`, `replayed`, `composite_score`. Record `$RUN_REPLAY = <replay_run_id>`.

- [ ] **Step 3: Compare baseline and replay**

```bash
uv run python -m src.eval compare --run-a $RUN_BASELINE --run-b $RUN_REPLAY
```
Expected: per-question score deltas on identical answers. Any delta reflects rubric change, not agent drift.

### Task 5.3: Human spot-check and measurable-gate verdict

**Files:**
- Create: `docs/superpowers/plans/2026-04-21-eval-rubric-validation.md`

- [ ] **Step 1: Select ≥ 10 spot-check questions**

From the replay output, pick at least 10 questions that have `required_facts` annotated. Prioritize questions where the baseline-vs-replay delta on `correctness` is large (absolute score difference ≥ 1), and include at least 3 questions where scores did not change (so we detect cases where the new judge passed a broken answer just as the old one did).

- [ ] **Step 2: Reviewer reads each and records a verdict**

For each question, the reviewer opens the baseline and replay records in Argilla (or queries the DB directly), reads:
- the question
- the agent's stored answer
- baseline judge's correctness + justification
- replay judge's correctness + justification + `required_facts_coverage`

Records one of four verdicts per question:
- `better` — replay judge's score is more aligned with the reviewer's reading
- `worse` — replay judge's score is less aligned with the reviewer's reading
- `tied` — both judges read the answer the same way and agreed with the reviewer
- `mixed` — ambiguous; the replay judge caught something and missed something else

- [ ] **Step 3: Evaluate the measurable gate**

Count the verdicts. The spec's success criterion #4 pass threshold:
- `better` count ≥ 60% of spot-checked questions
- `worse` count ≤ 10% of spot-checked questions

If both hold → pass. If either fails → investigate specific failures before shipping ground-truth-aware judge behavior into production evaluations.

- [ ] **Step 4: Write `docs/superpowers/plans/2026-04-21-eval-rubric-validation.md`**

```markdown
# Eval Rubric Ground Truth — Validation Report

**Date:** <date>
**Baseline run:** $RUN_BASELINE
**Replay run:** $RUN_REPLAY
**Batteries covered:** <list>

## Coverage

- Total questions in baseline run: <N>
- Questions with required_facts available at replay time: <M>
- Questions rated `stable`: <S>, `time_bound` (unexpired): <T>

## Score deltas (identical answers)

Top N questions by |delta| on correctness:

| question_id | baseline correctness | replay correctness | delta | required_facts covered |
| ----------- | -------------------- | ------------------ | ----- | ---------------------- |
| ...         | ...                  | ...                | ...   | ...                    |

## Spot-check verdicts

| question_id | reviewer verdict | notes |
| ----------- | ---------------- | ----- |
| ...         | better/worse/tied/mixed | ... |

- better: <count> / <total> = <pct>%
- worse: <count> / <total> = <pct>%
- tied: <count>
- mixed: <count>

**Measurable-gate result:** PASS / FAIL

## Observations and follow-ups

(any surprising regressions, judge prompt tweaks recommended, specific facts that appeared to confuse the judge, etc.)
```

- [ ] **Step 5: Commit the report**

```bash
git add docs/superpowers/plans/2026-04-21-eval-rubric-validation.md
git commit -m "test(eval): Phase 5 validation report — ground-truth judge replay"
```

### Task 5.2: Open a pull request

**Files:** none.

- [ ] **Step 1: Push the branch**

Run:
```bash
git push -u origin feature/eval-ground-truth
```

- [ ] **Step 2: Open a PR**

Run:
```bash
gh pr create --title "feat(eval): add ground-truth-aware judging" --body "$(cat <<'EOF'
## Summary
- Extended Argilla dataset schema with `required_facts`, `ground_truth_stability`, `ground_truth_valid_until`
- Added `GroundTruthCache` that primes per-run and feeds authored facts to the judge
- Extended judge prompt + response schema with per-fact coverage
- Bootstrap-annotated the battery via Argilla
- Validation report: `docs/superpowers/plans/2026-04-21-eval-rubric-validation.md`

Spec: `docs/superpowers/specs/2026-04-21-eval-rubric-ground-truth-design.md`.

## Test plan
- [ ] Full eval test suite green (`uv run pytest tests/ -k eval -v`)
- [ ] Smoke test against combined battery with at least one annotated question
- [ ] Validation report reviewed

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Rollout notes

- Phase 1 is safe to merge on its own — schema change only, no behavior shift.
- Phases 2-3 together become active only when there are annotations in Argilla. Merging them before Phase 4 is safe; nothing changes until someone authors facts.
- Phase 4 is human work outside git.
- Phase 5 is the "does this actually help" check.
- If any phase is incomplete at launch, defer post-launch. No launch delay.
