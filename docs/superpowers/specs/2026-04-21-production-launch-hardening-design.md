# Production Launch Hardening — Design Spec

**Date:** 2026-04-21
**Status:** Primary launch-track spec. Parallel workstream: `2026-04-21-eval-rubric-ground-truth-design.md` (not a blocker).

## Purpose

Harden the access-agent codebase, deploy pipeline, and evidence package so it can be flipped into production with the new architecture (native tool calling + UKY-hosted primitives) and hold up under real user traffic.

## Launch Criteria

From leadership (2 PIs + Vikram at UKY). Three criteria, each with a precommitted rubric (concrete numbers in §Pre-Phase-7 Calibration below).

1. **Quality** — leadership convinced the new architecture's answers are at least as good as current prod's. Primary evidence: human side-by-side review of battery answers (old vs. new) in Argilla; HTML report summary via Joe's framework. Supplementary: judge scores (strengthened by the parallel rubric work if it lands). Pass rubric: leadership reviews N spot-checked side-by-side pairs from the comparison; each of the three stakeholders casts per-question "at least as good" yes/no; go gate is a unanimous majority-yes with no individual voting no on more than X% of sampled questions. N and X set during pre-Phase-7 calibration.
2. **Speed** — latency close enough to current prod that the quality/privacy wins justify it. Evidence: objective p50 and p95 latency on the same battery, reported alongside quality. Pass rubric: p50 new ≤ M₁× p50 current-prod; p95 new ≤ M₂× p95 current-prod. Multipliers M₁ and M₂ set during pre-Phase-7 calibration.
3. **Privacy** — documented data-flow diagram showing user-query content stays in-network (UKY-hosted retrieval, UKY-hosted LLM). Pass rubric: all three stakeholders sign off on the privacy document in writing.

### Pre-Phase-7 Calibration

Before Phase 7 produces any evidence, a short calibration conversation with leadership precommits the concrete thresholds in the rubrics above. Specifically:

- Quality: sample size N (suggest 10–15), and max individual-dissent rate X (suggest 20%).
- Speed: latency multipliers M₁ (suggest 1.5×) and M₂ (suggest 2.0×) vs. current prod baseline.
- Privacy: confirm the three sign-offs are recorded in a durable form (email or equivalent) and who the signatories are.

The suggested numbers are our opening proposal; final numbers come from leadership agreement. Captured in `docs/launch/2026-XX-launch-bar-calibration.md` (or equivalent) before Phase 7 starts. Without this calibration, Phase 8 go/no-go cannot be evaluated objectively.

## Non-goals

- Not production-level SRE (on-call rotations, paging, formal SLOs). This is research/service infrastructure, not a 24/7 product.
- Not gradual rollout with canarying real traffic. Old architecture continues serving until the flip; new architecture receives real traffic at full scale post-cutover.
- Not solving eval methodology (covered by the parallel rubric plan).
- Not achieving a specific numeric quality improvement. Human side-by-side + leadership judgment is the primary bar.
- Not depending on a specific UKY delivery date. If UKY endpoints slip, the spec accommodates either (a) waiting, or (b) launching on OpenAI temporarily with explicit leadership sign-off on the compromised privacy story.

## Success Criteria (internal to this plan)

- Staging environment exists with a green smoke test on every main-merge.
- Production deploys require a manual `workflow_dispatch`, not an automatic trigger.
- Every write-capable capability is enumerated in a committed audit document, and a `READ_ONLY=true` env var is enforced in code.
- Privacy data-flow document exists and has been reviewed by the three leadership stakeholders.
- HTML comparison report + Argilla datasets produced for old-vs-new architecture on at least the combined battery.
- Production cutover completed with new architecture serving real traffic; post-cutover monitoring window documented.

## Current-State Context

Relevant to anyone executing this plan.

- **Deploy pipeline today:** `.github/workflows/deploy-production.yml` fires on every push to main, builds a multi-arch Docker image, pushes to GHCR, SSHes to `PRODUCTION_HOST`, pulls and restarts compose, runs a smoke test. No staging environment exists.
- **Smoke test today:** one generic query ("smoke test") checked for `success=true`. Classifier is unlikely to route it as a write action, but this is not guaranteed — the smoke test has no safety guard against accidental ticket creation or announcement mutation.
- **Capability registry** (`src/agent/domains/capabilities.py`) already supports `DISABLED_CAPABILITIES` env var with deny-wins semantics. Missing: a `READ_ONLY=true` convenience that disables all write-capable capabilities at once.
- **Dependency pins:** `langgraph>=0.2.0`, `langchain-core>=0.3.0`, `langchain-openai>=0.2.0`. Current upstream is LangGraph 1.1.x / langchain-core 1.3.x — two major versions behind. Upgrade is a prerequisite to any tool-calling-loop work (see `2026-04-20-framework-verification.md`).
- **LLM providers wired:** `OpenAIProvider` (default `gpt-4o`), `OpenAICompatibleProvider` (points at `VLLM_BASE_URL`, currently unused), `access_ai` (UKY RAG facade, used as RAG service only).
- **Tool-calling today:** JSON-in-text planning in `plan_node`, hand-rolled `$step_N` reference resolver in `execute_node`. This is what the tool-calling-loop phase replaces.
- **UKY primitives status:** Vikram plans to ship `/retrieve` + docs MCP endpoints this week; LLM hosting on Grace Hopper system on a longer timeline. See `docs/collaboration/2026-04-20-uky-message-draft.md`.
- **Eval harness:** 4 batteries (~170 lines total of questions), judge runs with OpenAI-compatible LLM. HTML report framework handles run-to-run comparison (Joe's work).
- **Argilla:** `eval-production` dataset, rating-oriented today, ground-truth extension in progress via parallel rubric plan.

## Phase Structure

Nine phases across four tracks.

```
Track 1 (Foundation, serial):
  Phase 0 — Dependency upgrade
       ↓
  Phase 1 — Side-effect safety audit

Track 2 (Infrastructure):
  Phase 2 — Staging environment
       ↑ (depends on Phase 1)

Track 3 (New architecture, parallel to Track 2):
  Phase 3 — Tool-calling loop on OpenAI-compatible
       ↓
  Phase 4 — UKY /retrieve integration  [gated on UKY]
       ↓
  Phase 5 — UKY LLM swap               [gated on UKY]

Track 4 (Evidence + cutover, serial, at end):
  Phase 6 — Privacy document
       ↓
  Phase 7 — Side-by-side evidence
       ↓
  Phase 8 — Leadership checkpoint + production cutover
```

### Phase 0 — Dependency upgrade (prerequisite)

- Update `pyproject.toml`: `langgraph>=1.1.0,<2.0.0`, `langchain-core>=1.3.0,<2.0.0`, `langchain-openai>=1.1.0,<2.0.0`.
- Run `uv lock --upgrade && uv sync`.
- Adapt breaking changes: `create_react_agent` → `create_agent` in `src/agent/nodes/domain_agent.py`; any `StateGraph` / `AsyncPostgresSaver` signature shifts.
- Run the full test suite; fix regressions in-place.
- Lifted from the earlier Shape B plan's Phase 0; that plan file is deleted as part of Phase 3 scope.

### Phase 1 — Side-effect safety audit

Three deliverables, tight scope:

**1a. Checklist.** Enumerate every write-capable capability in the registry:
- `manage_announcements` (domain: announcements) — create/update/delete announcements.
- `open_ticket` (domain: jsm) — create JSM tickets.
- `report_login_problem`, `report_security` (domain: jsm) — also ticket-creating.
- Any other capability whose backend performs a POST/PUT/DELETE to an external service.

For each: classifier prompt rules that gate it (explicit imperative language required), backend MCP server, auth requirements, what `acting_user` is required to be non-anonymous.

**1b. Document.** `docs/security/write-capability-audit.md`. Enumerates the checklist, the env-var controls (`ENABLED_CAPABILITIES`, `DISABLED_CAPABILITIES`), the `READ_ONLY` mechanism (introduced in 1c), and explicitly states: "the production smoke test, when run with `READ_ONLY=true`, cannot create tickets or announcements regardless of classifier behavior."

Deliberately does not include the "how to temporarily enable writes on staging" testing workflow — that belongs in a dev-facing staging README, separately.

**1c. Code guard.** Add `READ_ONLY: bool = False` to `src/config.py`. In `CapabilityRegistry.__init__` (or equivalent bootstrapping site), when `settings.READ_ONLY is True`, force-add every write-capable capability ID into the disabled set. Log prominently at startup:

```
READ_ONLY=true active — write capabilities disabled: manage_announcements, open_ticket, ...
```

Tests: a unit test confirming that with `READ_ONLY=true`, `is_capability_enabled("manage_announcements")` returns `False` even when `ENABLED_CAPABILITIES` names it.

### Phase 2 — Staging environment

**Infrastructure:**
- New VM, smaller than prod (4 vCPU / 8GB RAM should be plenty). Operational cost and procurement are not specified here.
- DNS: `access-agent-staging.elytra.net` (or similar on the elytra.net domain; specific subdomain decided during execution).
- Caddy with Let's Encrypt automatic cert.
- Separate Postgres container with its own database.
- Separate Redis container.
- Docker-compose stack mirroring the prod file, with staging-specific env.

**GitHub Actions changes:**
- New secrets: `STAGING_HOST`, `STAGING_SSH_KEY`. Reuse existing `GHCR_PAT`.
- New workflow: `.github/workflows/deploy-staging.yml`. Triggers on `push: branches: [main]` and `workflow_dispatch`. Deploys to `STAGING_HOST`, runs the hardened smoke test against the staging container.
- Modify `.github/workflows/deploy-production.yml`: remove `push: branches: [main]` trigger; change to `workflow_dispatch` only. Declare `inputs.image_tag` as `required: true` — the operator must paste in the exact image tag/digest that was validated on staging. The prod workflow pulls that specific tag, not `:latest`, preventing a race where a newer main commit has built a fresh image since staging validation. The staging workflow's final step prints the validated image tag/digest prominently for the operator to copy.

**Smoke-test hardening (Phase 2 sub-task):**

The current single-query smoke test is replaced with a capability-aware set. Representative queries, each asserted independently:
- Static: "What is ACCESS?"
- Combined (compute): "What resources have A100 GPUs?"
- Dynamic (events): "What events are coming up?"
- Dynamic (status): "Is Delta down right now?"
- Combined (software): "What software is on Bridges-2?"

Each returns `success=true`, logs a usage row, produces a non-empty answer. Report per-query pass/fail in the smoke-test output. Runs on staging every main-merge, on prod on every manual promotion.

Staging runs with `READ_ONLY=true` by default. Override via env for write-capability testing (documented in a staging README, not here).

### Phase 3 — Tool-calling loop on OpenAI-compatible

Replaces `plan + execute + evaluate + recover` with a single `tool_calling_loop` node using OpenAI's native tool-calling API.

**Key points** (full detail becomes the implementation plan):
- New file `src/agent/nodes/tool_calling_loop.py`.
- MCP tools converted to OpenAI tool definitions (not Anthropic's `input_schema` shape — OpenAI's `function` shape).
- Uses `OpenAIProvider` during development (OpenAI's `gpt-4o` or current equivalent), switches to `OpenAICompatibleProvider` pointed at UKY's vLLM when Phase 5 lands.
- Feature flag `USE_TOOL_CALLING_LOOP=true` gates new vs. old code path.
- Routing changes in `src/agent/graph.py` branch on the flag. Old path remains functional.
- Hand-rolled `$step_N` resolver in `execute.py` is retired (the loop chains tool outputs natively).
- The eval battery must pass on the new path; per-capability smoke covers the common flows.

**Prior artifact:** the Shape B tool-calling-loop plan (targeting Anthropic SDK direct) was removed when this spec was written; its architectural reasoning lives in the three research docs (`2026-04-20-tool-calling-models.md`, `2026-04-20-pipeline-and-orchestration.md`, `2026-04-20-framework-verification.md`). Phase 3's implementation plan is drafted fresh against the OpenAI-compatible target.

### Phase 4 — UKY `/retrieve` integration

Gated on UKY shipping the endpoint.

- Update `src/services/uky_client.py` to add a `retrieve(query, rp_name?)` method returning `{chunks, in_scope?}` per the contract Vikram is building.
- Update the synthesis path: instead of consuming UKY's paragraph response, consume raw chunks. Single synthesis pass inside the tool-calling loop (or immediately after it).
- Remove `rag_answer_node`'s deflection-detection heuristics — not needed when we control synthesis directly.
- Update eval runner to capture chunk metadata (source URL, doc title, score) in the `rag_context` snapshot for audit.

If UKY slips past the launch window with `/retrieve` unavailable, Phase 4 defers; agent continues consuming paragraphs via the existing `uky_client.ask()` method. Launch still ships.

### Phase 5 — UKY LLM swap

Gated on UKY hosting the chosen model via vLLM.

Config change only — no code:
- `LLM_PROVIDER=vllm`
- `VLLM_BASE_URL=<UKY-provided>`
- `VLLM_API_KEY=<UKY-provided>`
- `VLLM_MODEL_NAME=<Qwen3.6-35B-A3B or whichever variant UKY deploys>`

UKY configures the vLLM tool-calling parser (`qwen3_coder` / `llama4_pythonic` / etc.) on their side. We verify the end-to-end flow on staging before promoting to prod. Full eval run on staging against the new model produces the comparison data that feeds Phase 7.

If UKY slips past launch, Phase 5 defers; agent continues using OpenAI. Privacy story is compromised and must be explicitly called out in Phase 6; leadership signs off on the compromise or delays launch.

### Phase 6 — Privacy document

`docs/privacy/data-flow.md`. One-page diagram + narrative.

**Data flow diagram** shows:
```
User query → Drupal frontend (JWT issued) → access-agent → UKY /retrieve
                                                        ↘ UKY vLLM LLM
                                                        ↘ MCP tool servers
                                          → Response
```

With clear labels on which boxes are "in-network" (ACCESS-CI-operated or UKY-operated infrastructure) versus external.

**Narrative sections:**
- What leaves the network: Turnstile tokens (Cloudflare), Honeycomb traces (see verification task below).
- What's logged locally: `usage_logs` table (per-query records), LangGraph PostgreSQL checkpointer (multi-turn conversation state per session_id), audit trails.
- Retention posture: what's kept how long, what's deleted.
- What does *not* leave after Phase 5 lands: user query text, retrieved chunks, LLM outputs, tool results.

**Verification task for Phase 6 execution:** check every `span.set_attribute` in `src/telemetry/spans.py` and every `add_span_event` call across the codebase to determine whether user-query content, RAG chunks, or synthesis output are sent to Honeycomb. Document the finding in the privacy doc. If query content leaves, either (a) scrub it at the span level before export, or (b) explicitly name Honeycomb as trusted infrastructure and justify.

Document structure leaves obvious slots for future extension into a fuller security architecture document, but that extension is out of scope here.

### Phase 7 — Side-by-side evidence

Two artifacts:

**1. HTML report (Joe's existing framework).** Compares a recent current-prod eval run against a new-architecture-on-staging eval run. Summary metrics (pass rate per battery, latency p50/p95), per-capability rollup, embedded privacy diagram, link to Argilla for question-by-question detail. Generated once via Joe's tooling; regenerable on demand.

**2. Argilla datasets.** Both runs pushed with `agent_branch` metadata distinguishing old vs. new. Filterable per-capability / per-battery / per-delta. Existing rating schema (supplemented by rubric plan's ground-truth work if it lands) serves detail review.

**Eval methodology:**

The old-vs-new comparison risks being confounded by environment drift (MCP tool responses change minute-to-minute, especially for dynamic queries like "what events are coming up?"). Apply the following controls:

- **Temporal control.** Run both configurations back-to-back within a tight window (same hour, same day). Document the exact start timestamp for each run in the report.
- **Battery coverage.** `combined_battery.json` at minimum; all four batteries if time permits.
- **Current-prod run:** executes against current prod's configuration.
- **New-architecture run:** executes on staging with `USE_TOOL_CALLING_LOOP=true` and UKY primitives configured (or OpenAI temporarily if UKY slipped — flagged explicitly in the report).
- **Dynamic-question flagging.** Questions marked `expected_type == "dynamic"` are called out separately in the report. Their deltas are interpreted with explicit caution: "this delta may reflect MCP upstream data change between runs." Static and combined-but-stable questions carry the primary comparison weight.
- **Human spot-check** on the N questions leadership will review per the calibrated sample (see §Pre-Phase-7 Calibration); plus any dramatic automated deltas (top-K by absolute score difference). Notes captured in the HTML report.
- **Future improvement (not in this launch):** snapshot MCP responses during the baseline run and replay them for the new-architecture run. Removes temporal confounds entirely. Meaningful implementation work; worth considering if the rubric plan or a future launch iteration wants a stricter comparison.

### Phase 8 — Leadership checkpoint + production cutover

**Checkpoint:**
- Share HTML report + Argilla dataset IDs + privacy document with PIs and Vikram.
- Collect written go/no-go from each in a durable form (email or equivalent).
- Address questions; re-run evidence if leadership requests follow-up.

**Cutover, on unanimous go:**
- Merge prod-cutover config change to main (sets `USE_TOOL_CALLING_LOOP=true` in `docker-compose.prod.yml`, plus any UKY endpoint env vars).
- Main merge triggers staging deploy + smoke test.
- On staging green, manually dispatch the prod workflow (`workflow_dispatch` on `deploy-production.yml`) to deploy to prod.
- Cutover window: business hours, during period of low expected traffic.

**Post-cutover monitoring:**
- Defined monitoring window (e.g., first 24 hours). Watch Honeycomb for:
  - Latency regressions vs. pre-cutover baseline.
  - Error rates on tool calls.
  - Classifier misrouting (unusual distributions of query_type, domain, capability_id).
  - Synthesis failures.
- No formal SLO. Documented "here's what we're watching and here's the rollback trigger." Rollback mechanism: manually dispatch the prod workflow with the previous image tag.

## Deliverables

- Code changes across Phases 0-5 (implementation plans produced per phase).
- `docs/security/write-capability-audit.md` (Phase 1b).
- `deploy-staging.yml` workflow; modified `deploy-production.yml`; staging VM + DNS + cert provisioned (Phase 2).
- `docs/privacy/data-flow.md` (Phase 6).
- HTML eval-comparison report (Phase 7).
- Argilla datasets for old + new architecture runs (Phase 7).
- Written go decisions from 3 leadership stakeholders (Phase 8).
- Post-cutover monitoring notes (Phase 8).

## Not Deliverables

- On-call rotation / paging setup.
- Formal SLOs or error budgets.
- Traffic-splitting canary infrastructure.
- Secondary region or multi-region deployment.
- Automated rollback triggers (rollback is a manual dispatch).
- MCP-server-level deny for write operations (see Post-Launch Hardening).

## Post-Launch Hardening

Not gating the launch, but named here so they don't get lost:

- **Deployment-boundary write-deny on staging.** The current Phase 1 guard enforces `READ_ONLY=true` at the access-agent code layer via the capability registry. A layered defense would add MCP-server-side deny: staging's MCP connection uses an API key the MCP servers recognize as "staging, no writes allowed." If the agent's code-level guard regresses or is bypassed (classifier bug, refactor mistake, new write-capable tool added without gating), the MCP server layer still denies. Requires coordination with the MCP-servers repo. Good defense-in-depth; not worth shipping half-baked pre-launch.
- **MCP response snapshot/replay for eval comparisons.** Flagged in Phase 7.
- **Fuller security-architecture document.** The privacy doc in Phase 6 leaves slots for this; the extension covers logging retention, audit trails, tenancy, and threat model. Leadership requested A-level scope for this launch; post-launch extends.

## Relationship to Parallel Work

- **Rubric plan** (`2026-04-21-eval-rubric-ground-truth-design.md`): runs in parallel during UKY waits. If it lands before Phase 7, supplements the HTML report with stronger judge scoring. If not, Phase 7 runs on human side-by-side + current judge.
- **UKY work** (Vikram's team): `/retrieve` this week, MCP endpoints this week, LLM hosting on a longer timeline. Phases 4 and 5 are gated on their delivery. Communication channel is the message draft at `docs/collaboration/2026-04-20-uky-message-draft.md` and whatever follow-up the teams establish.
- **Frontend chatbot-UI staging** (separate repo, separate owners): end-to-end staging requires a chatbot UI instance pointed at `access-agent-staging.elytra.net`. Scope, infrastructure, and timeline for that UI-side staging are owned by the frontend team and tracked in their repo. This plan coordinates but does not specify it. Coordination touchpoints: agreeing on the staging agent URL before Phase 2 lands (so the UI team can configure their staging build), and having a working UI-against-staging pairing before Phase 7's evidence collection, since some of the human side-by-side review may benefit from real chat-UI interaction rather than raw API responses.

## Open Questions for Implementation

- Specific staging subdomain on elytra.net — decided during Phase 2 execution.
- Exact staging VM specs and provider — operational choice, not spec-relevant.
- Honeycomb trace content — verified and addressed during Phase 6.
- Exact UKY-deployed model — depends on Vikram's choice from the shortlist we provided.
- Cutover timing — decided at Phase 8 checkpoint based on leadership availability and traffic patterns.

These are verified and resolved during execution; none block the spec.
