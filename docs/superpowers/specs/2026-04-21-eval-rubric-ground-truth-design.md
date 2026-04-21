# Eval Rubric Ground Truth — Design Spec

**Date:** 2026-04-21
**Status:** Parallel to the production launch work; not a launch blocker.

## Purpose

Improve eval methodology so comparison results (old architecture vs. new, future changes vs. baseline) are trustworthy enough to base launch and ongoing quality decisions on. Address the current weakness: the LLM judge scores well-formatted-but-factually-wrong answers highly because it has no ground truth to check against.

## Success Criteria

1. When a human and the judge disagree on the same answer, disagreement is traceable to either (a) judge lacking ground truth for that question — fixable by annotating, or (b) judge misreading present ground truth — fixable by prompt tuning.
2. A named subset of the battery has authored required facts available to the judge at scoring time.
3. Annotators authoring required facts in Argilla is as fast or faster than writing free-form `feedback`, so the workflow is not a regression.
4. **Measurable agreement gate:** on a spot-check set of at least 10 answers (drawn from questions with annotated `required_facts`), a human reviewer confirms on a per-question basis whether the new judge's correctness score is better-aligned with the reviewer's reading than the baseline judge's score on the same answer. Pass threshold: reviewer records "better" on ≥ 60% of the spot-checked questions, "worse" on ≤ 10%. The remainder may be "tied" or "mixed." Recorded in `docs/superpowers/plans/2026-04-21-eval-rubric-validation.md`.

## Non-goals

- Not redesigning the 5-dimension rubric (correctness, completeness, relevance, citation_quality, hedging stay).
- Not solving eval for highly dynamic queries where "correct" changes hourly — those are scoped via the `ground_truth_stability` label.
- Not replacing Argilla or building a new annotation UI.
- Not LLM-generating required facts (humans author; LLM-authored ground truth defeats the purpose).
- Not a launch blocker under any circumstance.

## Relationship to Launch Work

This plan runs **parallel to the production-launch hardening work**. Specifically:

- Launch evidence is human side-by-side review of battery answers (old vs. new architecture), plus objective latency metrics, plus documented privacy architecture. Those three stand on their own.
- This rubric work contributes opportunistically: if required-facts annotation is partially or fully in place at launch time, the judge's scores become stronger *supplementary* evidence next to the human review. If not in place, launch proceeds on human review alone.
- Natural interleaving: rubric phases slot into launch-plan waits (dependency upgrade, UKY endpoint dependencies).
- Hard line: if Phase 3 (judge-prompt integration) is not complete by the launch's side-by-side eval, it is deferred post-launch. The launch does not wait.
- The launch work does not pause to accommodate rubric progress.

## Current-State Context

Relevant to anyone executing this plan.

- **Rubric and judge:** `src/eval/rubric.py` defines 5 scoring dimensions with a weighted composite. `src/eval/judge.py` calls an OpenAI-compatible LLM with `build_judge_prompt()` and parses JSON back into `JudgeResult`.
- **Question schema:** `src/eval/questions.py` loads `EvalQuestion` from JSON files in `eval/questions/`. Existing per-question metadata includes `expected_type` (`static`/`dynamic`/`combined`) and `expected_tools`. No `reference_answer` or `required_facts` field today.
- **Argilla integration:** `src/eval/argilla_push.py` creates the `eval-production` dataset with rating questions per rubric dimension, a `decision` label question (approved/needs_revision/rejected), and an optional `feedback` text question. `argilla_pull.py` exists for syncing human scores back — it pulls ratings, not reference answers.
- **Existing annotations:** none. The dataset can be recreated with a new schema without migration concerns.
- **Judge context today:** already receives `rag_context`, `tool_results`, and `node_trace` when available. The correctness-prefers-tool-results guidance (commit `a309b6c`) is in the prompt. The gap is ground truth, not MCP-data visibility.

## Architecture

Three components, existing code extended rather than replaced.

### Component A — Argilla schema extension

Add three new fields alongside the existing ones in `create_eval_dataset()`:

- `required_facts` — `TextQuestion`, `required=False`, `use_markdown=True`. Reviewer authors a markdown bulleted list of facts any correct answer must contain. Atomic claims, one per bullet.
- `ground_truth_stability` — `LabelQuestion`, `required=False`, labels: `stable`, `time_bound`, `not_applicable`, `not_reviewed`. Captures whether the authored facts are stable over time. Default suggestion `not_reviewed` so un-annotated records are unambiguous.
- `ground_truth_valid_until` — new metadata property (`TermsMetadataProperty` or similar), ISO date string. Required when stability is `time_bound`, ignored otherwise. Judge skips expired facts.

**Fail-closed rule for `time_bound` integrity.** A record with `ground_truth_stability == time_bound` but a missing, empty, or malformed `ground_truth_valid_until` is treated as invalid: the cache drops the record with a logged warning. Rationale: `time_bound` declares the facts will go stale; without a date we can't honor that contract, and falling back to "use indefinitely" defeats the label.

Existing fields unchanged. Existing rating questions, decision label, feedback text, all the metadata properties — untouched.

Dataset guidelines text gains a "Ground truthing" subsection explaining the new fields and when to use each stability label.

Dataset migration: the existing `eval-production` dataset is recreated with the new schema. No annotations exist, so no data loss.

### Component B — Argilla pull-back module

New file: `src/eval/argilla_ground_truth.py`.

```python
@dataclass(frozen=True)
class RequiredFacts:
    question_id: str
    facts: list[str]                  # parsed bullets
    stability: str                    # "stable" | "time_bound"
    valid_until: date | None
    annotator: str | None
    annotated_at: datetime


class GroundTruthCache:
    def __init__(
        self,
        argilla_url: str,
        argilla_api_key: str,
        dataset_name: str,
    ) -> None: ...

    async def prime(self, question_ids: list[str]) -> None:
        """Fetch all relevant records in one pass at the start of an eval run."""

    def get(self, question_id: str) -> RequiredFacts | None:
        """Return cached facts for a question, or None if none available/valid."""
```

Semantics:

- `prime()` queries Argilla for records matching `question_id IN question_ids` with `required_facts` populated and `ground_truth_stability ∈ {stable, time_bound}`.
- For each question: pick the most-recent record with populated `required_facts`. Recency is established by the following cascade: (1) `updated_at` field on the Argilla record if present; (2) `inserted_at` (or equivalent creation timestamp) if `updated_at` is missing; (3) the record's `id` if IDs are sortable by insertion order — verify during Phase 2 whether the Argilla SDK guarantees this; (4) if all three are unavailable, log a warning and select the first-found record deterministically (stable sort by `id` lexicographic). The cascade is documented and tested so behavior is never undefined.
- Filter out `time_bound` facts past their `valid_until`. Records with `stability == time_bound` but missing/malformed `valid_until` are dropped per the fail-closed rule above.
- Parse `required_facts` markdown bullets into `list[str]`. One fact per bullet. Empty list treated as no ground truth.
- Cache is in-memory, per-run only. No disk cache.
- Connection failure: log a warning, cache stays empty, judge falls back to current behavior. The eval run never fails because of ground-truth unavailability.

### Component C — Judge prompt extension

`src/eval/rubric.py` — extend `build_judge_prompt()` signature:

```python
def build_judge_prompt(
    query: str,
    answer: str,
    rag_context: str | None = None,
    tool_results: str | None = None,
    node_trace: str | None = None,
    required_facts: list[str] | None = None,   # NEW
) -> str:
```

When `required_facts` is non-empty, prepend a "Required Facts" section to the prompt before the existing "Context the Agent Had Access To" section:

```
## Required Facts

A correct answer to this question must accurately represent all of the following:

- <fact 1>
- <fact 2>

When scoring correctness:
- Score 5 only if every required fact is accurately represented.
- Score 1-2 if any required fact is missing or contradicted.
- In your correctness justification, note each required fact and whether the answer covered it.
```

When `required_facts` is empty or None, the prompt is identical to today's.

Judge output schema gains one optional field on the correctness dimension:

```json
{
  "correctness": {
    "score": <1-5>,
    "justification": "...",
    "required_facts_coverage": [
      {"fact": "Delta has A100 GPUs", "covered": true, "note": "answer mentions A100s"},
      {"fact": "...", "covered": false, "note": "..."}
    ]
  },
  ...
}
```

Missing field in the response is treated as `[]` — backward compatible with old judge responses and with the no-ground-truth path.

`src/eval/judge.py` — `Judge.score()` passes `required_facts` through to `build_judge_prompt`. `parse_judge_response()` extracts `required_facts_coverage` if present. `JudgeResult` gains an optional `required_facts_coverage` field.

### Data Flow

```
Eval run starts
  ↓
GroundTruthCache.prime(all_question_ids)
  ↓
For each question:
  ↓
  agent.generate_answer()  (unchanged)
  ↓
  facts = cache.get(question_id)
  ↓
  judge.score(query, answer, ..., required_facts=facts.facts if facts else None)
  ↓
  store judge scores + per-fact coverage + question metadata

At end of run:
  ↓
  push_scores_to_argilla()  (unchanged — new records include same shape)
  ↓
  annotators may update/author required_facts on new records
  ↓
  next run sees updated facts via prime()
```

## Phasing and Deliverables

### Phase 1 — Schema extension (~half day)

- Update `create_eval_dataset()` in `src/eval/argilla_push.py`: add `required_facts` (TextQuestion), `ground_truth_stability` (LabelQuestion), `ground_truth_valid_until` metadata.
- Update guidelines text with ground-truthing guidance for annotators.
- Recreate `eval-production` (delete existing empty dataset, recreate with new schema).
- Unit tests for schema construction.

### Phase 2 — Ground-truth pull module (~1 day)

- Create `src/eval/argilla_ground_truth.py` with `RequiredFacts` dataclass and `GroundTruthCache` class.
- Implement `prime()`, `get()`, stability + `valid_until` filtering, markdown-bullet parsing.
- Graceful degradation on Argilla connection failure.
- Unit tests (mocked Argilla client).

### Phase 3 — Judge prompt integration (~half day)

- Extend `build_judge_prompt()` with `required_facts` parameter and the new prompt section.
- Extend `parse_judge_response()` to tolerate and capture optional `required_facts_coverage`.
- Extend `JudgeResult` with optional `required_facts_coverage` field.
- Wire `Judge.score()` to accept and pass through `required_facts`.
- Update the eval scorer (`src/eval/scorer.py` — the per-question run+judge orchestrator, not `runner.py` which only handles the agent invocation) to prime the cache at run start and fetch per-question facts at scoring time.
- Unit tests for prompt-with-facts, parse-with-coverage, parse-without-coverage.

### Phase 4 — Bootstrap annotation (time-bounded human work)

- Run the current battery against current-prod to populate Argilla with records for annotation.
- Annotate required facts, priority order:
  1. Static questions (`expected_type == "static"`) — stable, highest leverage.
  2. Combined questions with stable doc components.
  3. Dynamic questions only where trivially groundable (most get `not_applicable`).
- Budget: one focused 2-3 hour session for the initial pass. Expected yield ~60-80% of the battery annotated, remainder `not_applicable` or `not_reviewed`.

### Phase 5 — Validation (~half day)

**Approach:** replay, not re-run. Regenerating agent answers introduces model/output variance that confounds the judge comparison. Instead, re-score an existing run's stored answers using the new ground-truth-aware judge, and compare against the baseline judge's scores on *identical inputs*. This isolates the rubric delta from any agent-side drift.

- Implement a lightweight "replay" path in `src/eval/scorer.py` (or a new helper) that: (a) reads stored `answer_text`, `rag_context`, `tool_results`, `node_trace` from a chosen baseline run in the eval DB; (b) re-invokes the judge on each stored answer with the new ground-truth cache primed; (c) writes the new scores as a separate replay run in the DB. The agent is not re-invoked.
- Pick a baseline run: the most recent production eval run with `required_facts` authored on at least some of its questions (bootstrapped in Phase 4).
- Compare the baseline's original judge scores against the replay run's new scores on the same (question_id, answer_text) pairs.
- Investigate the top N score deltas (by absolute value). Verify they reflect intended behavior — specifically: is the correctness score now catching a factual error the baseline missed?
- Human spot-check: pick 10+ questions with annotated `required_facts`. For each, the reviewer reads the question, answer, baseline judge's correctness+justification, and new judge's correctness+justification+required_facts_coverage. Reviewer records per-question whether the new score is "better", "worse", "tied", or "mixed" relative to their own reading.
- Evaluate the success-criterion #4 pass threshold (≥ 60% better, ≤ 10% worse).
- Capture findings in `docs/superpowers/plans/2026-04-21-eval-rubric-validation.md`, including the baseline run ID, replay run ID, delta table, and spot-check tally.

### Phasing vs. launch

- Phase 1 can happen anytime in parallel with launch work.
- Phases 2-3 can land anytime; together they form the working-but-un-ground-truthed state.
- Phase 4 slots into launch-plan waits (UKY `/retrieve`, UKY LLM hosting).
- Phase 5 must run before the launch's side-by-side eval if Phases 1-3 landed — otherwise the launch eval doesn't benefit from this work.
- Any phase not completed by launch is deferred post-launch. No launch delay.

## Deliverables

- Code changes in `src/eval/*` (Phases 1-3) with tests alongside.
- New `eval-production` Argilla dataset with extended schema (Phase 1).
- Bootstrap annotations in Argilla (Phase 4).
- Validation results in `docs/superpowers/plans/2026-04-21-eval-rubric-validation.md` (Phase 5).

## Not Deliverables

- No retroactive re-judging of past runs.
- No automation of required-facts authoring.
- No change to rubric dimensions or weights.
- No changes to agent code — this is eval-layer only.

## Open Questions for Implementation

1. Does the Argilla Python SDK 2.x support multi-record queries filtered by metadata (needed for efficient `prime()`), or does it require per-record lookups? If the latter, batch size will determine run-start latency. Worth verifying in Phase 2.
2. Which recency fields does the Argilla SDK expose on record objects — `updated_at`, `inserted_at`, `id` ordering? The cascade described in Component B specifies the fallback order; Phase 2 confirms which tiers are actually available and adjusts the code accordingly. Tested with a unit test for each tier of the cascade.
3. Does the annotator UI surface markdown-rendered `required_facts` at review time (so future annotators see what's already been authored)? If not, minor Argilla field tweaks may be needed.

These are verified during Phase 2 and do not block the spec.
