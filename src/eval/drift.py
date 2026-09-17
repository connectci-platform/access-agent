"""Attribute a score change between two runs to drift, regression, or noise.

A composite drop is ambiguous on its own: the system may have regressed, the
world the facts describe may have moved, or the judge may simply have scored
differently. Guessing wrong is expensive in both directions — reverting a good
change, or shipping a real regression because it looked like drift. This module
does the attribution that was previously done by reading fact text by hand.

Three stored signals make it possible, each recorded per run or per fact rather
than reconstructed:

* ``eval_runs.agent_commit`` / ``llm_model`` / ``question_set`` — whether the
  system was actually held fixed. Without this the comparison means nothing,
  so a mismatch is reported rather than silently tolerated.
* ``eval_scores.context.fact_verdicts`` keyed by stable ``fact_id`` — which
  specific claims changed verdict, not just that the composite moved.
* ``eval_scores.context.tool_results`` — what the tools returned. If the same
  question retrieved different content, the world the answer describes moved.

``fact_kind`` (``behavioral`` | ``world`` | ``snapshot``) sharpens the verdict
but is deliberately NOT the sole input. It is a human judgment about whether a
claim *can* drift, made once when the fact was authored; the retrieval delta is
evidence about whether the world *did* move, measured per comparison. Where the
two disagree — a ``behavioral`` fact flipping alongside changed retrieval — the
finding is reported as a suspected mistype rather than forced into a bucket.
That case is not hypothetical: six facts asserting dated resource status were
typed ``behavioral`` until this module's first run surfaced them.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .db import EvalDB


class Verdict(StrEnum):
    """Why a fact's verdict changed between two runs of the same battery."""

    DRIFT = "drift"
    """The world moved: a world/snapshot fact now fails and retrieval changed."""

    REGRESSION = "regression"
    """The system moved: a behavioral fact now fails with retrieval unchanged."""

    NOISE = "noise"
    """Neither answer nor retrieval changed materially — judge variance."""

    SUSPECTED_MISTYPE = "suspected_mistype"
    """A behavioral fact failing alongside changed retrieval. Either the world
    moved and the fact is mistyped, or the system regressed at the same moment
    the corpus shifted. Needs a human read; guessing either way misleads."""

    UNTYPED = "untyped"
    """The fact carries no fact_kind, so drift and regression are
    indistinguishable. Reported so the gap is visible rather than assumed."""


@dataclass(frozen=True)
class FactChange:
    """One fact whose verdict changed, with the evidence behind its attribution."""

    fact_id: str
    question_id: str
    fact_kind: str | None
    before: str
    after: str
    retrieval_changed: bool
    verdict: Verdict
    fact_text: str = ""


@dataclass
class DriftReport:
    """The outcome of comparing two runs of the same battery."""

    baseline_run_id: str
    candidate_run_id: str
    system_fixed: bool
    system_differences: list[str] = field(default_factory=list)
    changes: list[FactChange] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {v.value: 0 for v in Verdict}
        for c in self.changes:
            out[c.verdict.value] += 1
        return out

    @property
    def regressions(self) -> list[FactChange]:
        """The changes that warrant blocking a release."""
        return [c for c in self.changes if c.verdict is Verdict.REGRESSION]


# A retrieval payload rarely round-trips byte-identically — timings and ordering
# move. Compare a normalized hash of the tool *content* so incidental jitter does
# not read as the world changing.
def _retrieval_fingerprint(context: dict[str, Any] | None) -> str:
    if not context:
        return ""
    raw = str(context.get("tool_results") or "")
    # Drop the per-call duration lines, which differ on every run by design.
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("- duration_ms:")]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def _verdicts_by_fact(context: dict[str, Any] | None) -> dict[str, str]:
    """Map fact_id -> verdict from a stored context, tolerating absence."""
    if not context:
        return {}
    raw = context.get("fact_verdicts")
    if not isinstance(raw, list):
        return {}
    out: dict[str, str] = {}
    for entry in raw:
        if isinstance(entry, dict) and entry.get("id") is not None:
            out[str(entry["id"])] = str(entry.get("verdict", ""))
    return out


def _fact_kinds(context: dict[str, Any] | None) -> dict[str, str | None]:
    """Map fact_id -> fact_kind from a stored context's required_facts."""
    if not context:
        return {}
    raw = context.get("required_facts")
    if not isinstance(raw, list):
        return {}
    out: dict[str, str | None] = {}
    for entry in raw:
        if isinstance(entry, dict) and entry.get("fact_id") is not None:
            out[str(entry["fact_id"])] = entry.get("fact_kind")
    return out


def _fact_texts(context: dict[str, Any] | None) -> dict[str, str]:
    if not context:
        return {}
    raw = context.get("required_facts")
    if not isinstance(raw, list):
        return {}
    return {
        str(e["fact_id"]): str(e.get("fact_text", ""))
        for e in raw
        if isinstance(e, dict) and e.get("fact_id") is not None
    }


def classify_change(
    fact_kind: str | None,
    *,
    retrieval_changed: bool,
) -> Verdict:
    """Attribute one pass->fail flip.

    ``fact_kind`` is the human judgment about whether the claim can drift;
    ``retrieval_changed`` is the measured evidence about whether the world did.
    The interesting cell is a behavioral fact with changed retrieval, which is
    reported rather than resolved.
    """
    if fact_kind is None:
        return Verdict.UNTYPED
    if fact_kind in ("world", "snapshot"):
        # A claim about the world failing while the retrieved world is unchanged
        # is not drift — the answer changed, which means the system did.
        return Verdict.DRIFT if retrieval_changed else Verdict.REGRESSION
    if fact_kind == "behavioral":
        return Verdict.SUSPECTED_MISTYPE if retrieval_changed else Verdict.REGRESSION
    return Verdict.UNTYPED


def compare_runs(db: EvalDB, baseline_run_id: str, candidate_run_id: str) -> DriftReport:
    """Compare two runs of the same battery and attribute every verdict flip."""
    baseline = db.get_run(baseline_run_id)
    candidate = db.get_run(candidate_run_id)
    if baseline is None or candidate is None:
        missing = baseline_run_id if baseline is None else candidate_run_id
        raise ValueError(f"Run {missing} not found")

    differences: list[str] = []
    for attr in ("question_set", "agent_commit", "llm_model"):
        before, after = getattr(baseline, attr), getattr(candidate, attr)
        if before != after:
            differences.append(f"{attr}: {before!r} -> {after!r}")

    report = DriftReport(
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        system_fixed=not differences,
        system_differences=differences,
    )

    before_scores = {s.question_id: s for s in db.get_scores_for_run(baseline_run_id)}
    after_scores = {s.question_id: s for s in db.get_scores_for_run(candidate_run_id)}

    for question_id, after in after_scores.items():
        before = before_scores.get(question_id)
        if before is None:
            continue  # question added since the baseline; nothing to compare
        before_ctx: dict[str, Any] = dict(before.context or {})
        after_ctx: dict[str, Any] = dict(after.context or {})

        retrieval_changed = _retrieval_fingerprint(before_ctx) != _retrieval_fingerprint(after_ctx)
        before_v = _verdicts_by_fact(before_ctx)
        after_v = _verdicts_by_fact(after_ctx)
        kinds = _fact_kinds(after_ctx) or _fact_kinds(before_ctx)
        texts = _fact_texts(after_ctx) or _fact_texts(before_ctx)

        for fact_id, after_verdict in after_v.items():
            before_verdict = before_v.get(fact_id)
            if before_verdict is None or before_verdict == after_verdict:
                continue
            if not (before_verdict == "yes" and after_verdict == "no"):
                continue  # only pass->fail warrants attribution
            kind = kinds.get(fact_id)
            report.changes.append(
                FactChange(
                    fact_id=fact_id,
                    question_id=str(question_id),
                    fact_kind=kind,
                    before=before_verdict,
                    after=after_verdict,
                    retrieval_changed=retrieval_changed,
                    verdict=classify_change(kind, retrieval_changed=retrieval_changed),
                    fact_text=texts.get(fact_id, ""),
                )
            )

    report.changes.sort(key=lambda c: (c.verdict.value, c.question_id, c.fact_id))
    return report
