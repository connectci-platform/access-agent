# Production Launch Hardening — Umbrella Plan

> **For agentic workers:** This is the umbrella coordinating plan for the launch-hardening effort. Execution is split across per-phase implementation plans; some phases are purely process and don't need code-level plans. Steps below are tracked with `- [ ]` where relevant, but most items here are phase-level status rather than TDD tasks.

**Goal:** coordinate the 9 launch-hardening phases, sequence them correctly, and point at the right implementation plans for each.

**Spec:** `docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md`.

**Parallel work:** eval rubric plan (`docs/superpowers/plans/2026-04-21-eval-rubric-ground-truth.md`) runs alongside this but does not block launch.

---

## Phase status at-a-glance

| # | Phase | Status | Implementation plan |
| - | ----- | ------ | ------------------- |
| 0 | Dependency upgrade | ready to execute | `docs/superpowers/plans/2026-04-21-launch-phase-0-deps-upgrade.md` |
| 1 | Side-effect safety audit + `READ_ONLY` guard | ready to execute | `docs/superpowers/plans/2026-04-21-launch-phase-1-safety-audit.md` |
| 2 | Staging environment | needs infra decisions first (see below) | not yet written |
| 3 | Tool-calling loop on OpenAI-compatible | blocked on Phase 0 landing | not yet written |
| 4 | UKY `/retrieve` integration | blocked on UKY delivery | not yet written |
| 5 | UKY LLM swap | blocked on UKY hosting | not yet written |
| 6 | Privacy data-flow document | ready to start (investigation can run now) | not yet written |
| 7 | Side-by-side evidence package | blocked on Phase 3 at minimum, ideally 5 | not yet written |
| 8 | Leadership checkpoint + production cutover | blocked on Phase 7 | not yet written |

---

## Dependency graph

```
Phase 0 (deps)
    ↓
Phase 1 (safety)
    ↓
Phase 2 (staging) ←── requires Phase 1's READ_ONLY guard
    │
Phase 3 (tool-calling loop) ─── needs Phase 0 landed; can parallelize with Phase 2
    ↓
Phase 4 (UKY /retrieve) ─────── blocked on UKY
    ↓
Phase 5 (UKY LLM swap) ───────── blocked on UKY
    ↓
Phase 6 (privacy doc) ────────── investigation can start anytime; finalizes after Phase 5
    ↓
Phase 7 (evidence package) ───── requires Phases 2, 3, 6; ideally also 4, 5
    ↓
Phase 8 (checkpoint + cutover) ─ requires Phase 7 + leadership calibration done
```

## Pre-Phase-7 calibration (must happen before Phase 7)

Per the spec's launch criteria, the numbers behind each criterion are negotiated with leadership before any evidence is collected. Drew owns the conversation; output goes in `docs/launch/2026-XX-launch-bar-calibration.md` (exact filename set when written).

- [ ] Propose quality thresholds to leadership: sample size N (~10-15), max individual dissent rate X (~20%).
- [ ] Propose speed thresholds: latency multipliers M₁ (~1.5×) and M₂ (~2.0×) vs current prod.
- [ ] Confirm privacy sign-off form (email vs. signed doc), and that the three signatories are PIs + Vikram.
- [ ] Write final agreed numbers into the calibration doc. Commit.

If leadership pushes back on any threshold, capture their counter in the doc and align Phase 7 evidence collection accordingly. Do not start Phase 7 without this.

## Phase 0 — Dependency upgrade

**Status:** ready.
**Plan:** `docs/superpowers/plans/2026-04-21-launch-phase-0-deps-upgrade.md`.
**Summary:** bump `langgraph` to 1.x, `langchain-core` to 1.x, `langchain-openai` to 1.x. Adapt `create_react_agent` → `create_agent` in `domain_agent.py`. Verify full test suite green on new pins.
**Blocks:** Phases 1, 3.

## Phase 1 — Side-effect safety audit + `READ_ONLY` guard

**Status:** ready.
**Plan:** `docs/superpowers/plans/2026-04-21-launch-phase-1-safety-audit.md`.
**Summary:** three deliverables — a checklist of write-capable capabilities, a committed audit document at `docs/security/write-capability-audit.md`, and a `READ_ONLY=true` env var that force-disables write capabilities in `CapabilityRegistry`.
**Blocks:** Phase 2 (staging uses `READ_ONLY=true` by default).

## Phase 2 — Staging environment

**Status:** blocked on infrastructure decisions Drew needs to make or delegate.
**Plan:** not yet written.

Before a per-phase plan can be written, these decisions need to land:

- [ ] **VM provisioning path.** Which cloud provider / ops workflow will stand up the staging VM? Procurement timeline.
- [ ] **DNS choice.** Exact subdomain on elytra.net (spec suggests `access-agent-staging.elytra.net`; final name to be set here).
- [ ] **Frontend coordination.** Frontend team aware they'll need a staging chatbot UI pointed at the staging agent URL; status of that work on their side.
- [ ] **GHCR access from staging host.** Confirm the `GHCR_PAT` that prod uses will work for staging, or whether a new PAT is needed.

Once those land, writing-plans produces the Phase 2 implementation plan covering: VM docker-compose stack, Caddy + Let's Encrypt config, `.github/workflows/deploy-staging.yml`, modifications to `deploy-production.yml` (remove push trigger, require `image_tag` input, print validated tag in staging output), hardened per-capability smoke test.

**Blocks:** Phase 7.

## Phase 3 — Tool-calling loop on OpenAI-compatible

**Status:** blocked on Phase 0 landing.
**Plan:** not yet written (will be written after Phase 0 merges).

**Summary:** replaces `plan + execute + evaluate + recover` nodes with a single `tool_calling_loop` node using OpenAI's native tool-calling API. MCP tools converted to OpenAI function definitions. Feature flag `USE_TOOL_CALLING_LOOP=true`. Routing in `src/agent/graph.py` branches on the flag. `$step_N` resolver in `execute.py` retires. Uses OpenAI's `gpt-4o` (or current equivalent) during development; swaps to UKY's vLLM endpoint via env change in Phase 5.

**Blocks:** Phase 7.

## Phase 4 — UKY `/retrieve` integration

**Status:** blocked on UKY shipping the endpoint.
**Plan:** not yet written.

**Summary:** update `src/services/uky_client.py` with a `retrieve()` method returning chunks; swap synthesis from "consume UKY paragraph" to "consume chunks alongside MCP tool results in a single synthesis pass"; remove `rag_answer_node` deflection detection; update eval runner to capture chunk metadata.

If UKY slips past launch, this phase defers — agent continues using `uky_client.ask()` (paragraph mode). Launch still ships on the old RAG consumption pattern.

**Blocks:** Phase 5 (ordering), Phase 7 (ideally, not strictly).

## Phase 5 — UKY LLM swap

**Status:** blocked on UKY hosting a model via vLLM.
**Plan:** not needed as a code-level plan. Config change only:

- [ ] Set `LLM_PROVIDER=vllm`
- [ ] Set `VLLM_BASE_URL=<UKY URL>`
- [ ] Set `VLLM_API_KEY=<UKY key>`
- [ ] Set `VLLM_MODEL_NAME=<deployed model>`
- [ ] Verify UKY enabled the appropriate tool-calling parser on their vLLM.
- [ ] Run full eval on staging against the new endpoint.

If UKY slips past launch: explicit leadership sign-off on launching with OpenAI temporarily, acknowledging the privacy story is compromised until the swap lands post-launch.

**Blocks:** Phase 7 (ideally).

## Phase 6 — Privacy data-flow document

**Status:** investigation can start now; document finalizes after Phase 5.
**Plan:** not needed as a code-level plan. Tasks:

- [ ] **Investigate Honeycomb trace content.** Review every `span.set_attribute` in `src/telemetry/spans.py` and every `add_span_event` call across the codebase. Document whether user-query content, RAG chunks, or synthesis output are sent to Honeycomb. If they are, decide: scrub at span level before export, or name Honeycomb as trusted infrastructure.
- [ ] **Draft `docs/privacy/data-flow.md`.** One-page diagram + narrative per spec section 6. Leave extension slots for future security architecture work.
- [ ] **Review with PIs + Vikram.** Collect sign-off in a durable form (email or equivalent) before Phase 8.

**Blocks:** Phase 7 (the privacy document is part of the evidence package), Phase 8.

## Phase 7 — Side-by-side evidence package

**Status:** blocked on Phases 2, 3, 6; ideally also 4, 5.
**Plan:** not yet written.

**Summary:** run combined battery (+ others if time) against (a) current prod configuration and (b) new-architecture-on-staging, within a tight temporal window. Push both runs to Argilla with distinguishing `agent_branch` metadata. Generate HTML comparison report via Joe's framework. Flag dynamic-question deltas with caveats. Spot-check ≥ N questions (N from calibration) with human review. Package with the privacy document.

**Blocks:** Phase 8.

## Phase 8 — Leadership checkpoint + production cutover

**Status:** blocked on Phase 7.
**Plan:** not needed as a code-level plan. Tasks:

- [ ] Share HTML report + Argilla dataset IDs + privacy document with PIs + Vikram.
- [ ] Collect written go/no-go from each in a durable form.
- [ ] On unanimous go: schedule cutover window (business hours, low-traffic period).
- [ ] Merge the prod-cutover config change to main (sets `USE_TOOL_CALLING_LOOP=true` and UKY env vars in `docker-compose.prod.yml`).
- [ ] Verify staging deploy of the cutover image is green.
- [ ] Manually dispatch `deploy-production.yml` with the validated image tag.
- [ ] Monitor Honeycomb over the documented post-cutover window (24h) for latency regressions, error rates, classifier misrouting, synthesis failures.
- [ ] Record observations; note any follow-up items for post-launch hardening.

**Rollback mechanism:** manually dispatch `deploy-production.yml` with the prior image tag.

---

## Post-launch hardening (not in this plan)

Named in the spec for context:

- MCP-server-level deny for write operations (defense-in-depth at the deployment boundary).
- MCP response snapshot/replay for eval comparisons (removes environment drift in Phase 7-like work).
- Extension of `docs/privacy/data-flow.md` into a fuller security architecture document.

Each becomes its own brainstorm + spec + plan when prioritized.

---

## Review and handoff

This umbrella plan is intended to be reviewed by the team before they start implementing. Phase 0 and Phase 1 plans have full TDD detail and can be executed immediately. Phases 2-8 are deliberately left as status placeholders until their prerequisites land or their external gates clear.
