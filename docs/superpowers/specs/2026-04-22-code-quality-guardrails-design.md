# Code Quality Guardrails — Design Spec

**Date:** 2026-04-22
**Status:** Pre-launch track; installs guardrails before the new-architecture build (launch Phase 3 and beyond) so code quality doesn't regress while moving fast. Parallel to `2026-04-21-production-launch-hardening-design.md` and `2026-04-21-eval-rubric-ground-truth-design.md`.

## Purpose

The launch-hardening work ships substantial new code over the next 1-2 weeks (tool-calling loop replacing plan/execute/evaluate/recover, UKY integration, synthesis changes, staging infrastructure). Without guardrails in place *before* that build starts, code quality can regress invisibly — new untested code lands, new supply-chain risk arrives with new dependencies, new secrets get handled carelessly.

This spec installs eight low-cost, high-leverage guardrails that catch quality regressions on the new work without punishing the existing 42% coverage baseline.

## Non-goals

- Not a codebase-wide coverage push. The baseline is 42% line coverage; lifting it to 80% is multi-week work unrelated to the launch.
- Not mutation testing, not performance regression gates, not contract tests against external services — all valuable, all explicitly deferred to post-launch.
- Not a redesign of the existing pre-commit hooks (ruff, mypy, pytest-at-pre-push stay as-is).
- Not a CI speed optimization (the existing ~16s pytest run is fine).
- Not any change to the production deploy pipeline beyond the launch spec's existing requirements.

## Success Criteria

- Every PR touching Python code gets a coverage delta comment and is subject to a 100%-on-changed-lines hard gate; bypass only via an auditable `coverage-override-approved` label applied by a repo admin.
- CI fails on any **high or critical** CVE in new or existing deps; medium/low severities are surfaced in the CI log as annotations but do not block merge.
- CI fails on accidentally-committed secrets; any suppression must go through a committed, reviewed `.gitleaks.toml` allowlist (not CI-level overrides).
- No test is **silently** skipped in CI. Two resolution classes acceptable: (a) test runs in the blocking PR job via stubbing (Strategy B), or (b) test runs nightly against live dependencies with failure notifications that open a GitHub issue or Slack alert within 24 hours (Strategy A). Strategy A is acceptable for tests whose value requires real upstream dependencies; stubbing would defeat their purpose.
- `main` branch protection is in place: force-pushes blocked, direct pushes blocked, at least one PR review required, required status checks pass before merge. A snapshot of the protection settings is committed as `docs/security/branch-protection-snapshot.json`; a periodic (weekly) CI job compares current settings to the snapshot and fails if they drift.
- Lockfile drift from `pyproject.toml` is a hard CI failure, not a silent mismatch.
- Dependabot opens PRs for dep updates on a weekly cadence.

## Pre-launch guardrails

### 1. Coverage diff-gate on PRs

Enforce: **changed lines in a PR must be 100% covered by tests**.

Tool: [`diff-cover`](https://github.com/Bachmann1234/diff_cover) or equivalent.

Mechanism: CI runs pytest with coverage, generates `coverage.xml`, runs `diff-cover coverage.xml --compare-branch=origin/main --fail-under=100`. Fails the job if any changed line is uncovered.

Why 100%: the simplest defensible rule — "if you added a line, you tested it." Any other threshold requires a judgment call about which PRs are "trivial enough" to exempt, and those judgment calls drift over time. The TDD discipline used across the launch and rubric plans already produces near-100% coverage on the tested slice, so this is closer to "codify what we already do" than "demand something new."

Escape valves:

- **`# pragma: no cover`** for lines where coverage is genuinely infeasible. Examples: retry/fallback branches that require injecting faults at the SDK level, defensive exception handlers that should never fire in practice, conditional imports. The existing `pyproject.toml` `[tool.coverage.report] exclude_lines` list already covers the common cases (`if TYPE_CHECKING:`, `def __repr__`, `raise NotImplementedError`, `if __name__ == .__main__.:`) — those lines don't need manual pragmas. Overuse is a code-review concern reviewers push back on during review.
- **`coverage-override-approved` label** for genuinely unusual PRs where 100% isn't appropriate (pure-refactor PRs that confuse diff-cover's line-change detection; tool-config bumps that touch no behavior). The label skips the coverage gate in CI. **Only repo admins can apply it.** Every application is visible in GitHub's audit log, so the "we bypassed the gate" is always observable, not silent. PR description must include the justification. Regular reviewers cannot apply the label — they must request an admin do so.

The bar is strict; the escape valves are auditable. A pattern of frequent label-based overrides is a signal that either (a) the threshold is wrong for this codebase, or (b) test discipline is slipping — the team should notice and course-correct.

### 2. Coverage-comment bot on PRs

Every PR gets a GitHub comment showing the coverage delta: what lines were added, which are covered, which aren't. Visibility drives behavior even without a gate.

Tool: `py-cov-action/python-coverage-comment-action` or equivalent.

Implementation: GitHub Action step in the existing `ci.yml` workflow. Posts one comment per PR, updates on each push.

### 3. Resolve the two CI-skipped tests

CI currently runs `pytest` with `--ignore=tests/test_e2e.py --ignore=tests/test_classify.py`. These tests require live API keys (OpenAI and/or MCP servers). Silently skipped tests hide classifier and end-to-end regressions on exactly the new work about to happen.

**The problem is silent skipping, not non-blocking execution.** Two resolution strategies, picked per-test:

- **Strategy B (preferred when feasible): stubbing.** Mock the LLM and MCP calls at the HTTP layer (`pytest-httpx` is already a dev dep). Tests run in every blocking CI job with no live dependency. Appropriate for tests that assert control flow, classifier JSON-schema compliance, routing decisions, synthesis prompt structure — things whose value doesn't depend on the specific LLM output.
- **Strategy A (when stubbing would defeat the test's purpose): nightly job with notifications.** A separate `.github/workflows/nightly.yml` runs the test once per day with `OPENAI_API_KEY`, `MCP_API_KEY`, and whatever else it needs, via GitHub Actions secrets. Failures open a GitHub issue automatically (or post to a Slack webhook), so regressions surface within 24 hours. Strategy A is acceptable because the failure is *surfaced*, not silent; the test is not merge-blocking but it is audit-visible.

Walk each test in the implementation plan, decide strategy with justification, apply. Acceptance criterion: no test is silently skipped after this lands. The CI log output must name every skipped test and the skip reason (either "stubbed-ok" or "deferred-to-nightly-workflow-<name>").

Decision framework for Strategy A vs. B:
- Does the test's value require real upstream data or real LLM behavior? → Strategy A.
- Does the test assert control flow that can be verified with mocked responses? → Strategy B.

Pure "real end-to-end against real services" tests are genuinely Strategy A. Most other test shapes should be Strategy B.

### 4. `uv audit` in CI

Audit the dependency tree for known CVEs on every CI run. `uv` ships an audit command; CI runs it against the locked dependency set.

Implementation: one step in `ci.yml`: `uv audit` (or equivalent subcommand — verify current `uv` CLI in Phase 2). **Fails the job on any HIGH or CRITICAL CVE.** Medium- and low-severity CVEs are reported in the job log as annotations (via `::warning::` output) but do not fail the job.

Rationale for the severity cut: transitive deps frequently carry medium/low CVEs with no available patch; blocking merge on them causes teams to either maintain a growing suppression list (noise) or bypass the tool (worse). High/critical CVEs are the real supply-chain risks worth blocking on.

**Escape valve:** a committed `.audit-ignore.yml` (or equivalent format supported by the tool) allows explicit, reviewed suppression of specific CVE IDs where the team has assessed and accepted the risk, with a required comment explaining why. Suppressions are reviewed in PRs like any other config change.

If `uv audit` doesn't exist in the installed `uv` version, fall back to `pip-audit` with equivalent severity flags against the `uv.lock`-resolved environment.

### 5. Gitleaks (pre-commit + CI)

Scan commits for accidentally-committed secrets: API keys, JWT signing keys, database passwords, OAuth tokens, etc.

Pre-commit hook: adds to `.pre-commit-config.yaml`. Runs on every commit locally; catches secrets before they hit the network.

CI job: also runs on every PR as a backstop for developers who bypass pre-commit with `--no-verify`. Scans the PR commits + any file changes.

Ignore list: `.gitleaks.toml` configuration file excludes deliberate fixtures (`tests/fixtures/fake-jwt.pem` etc. if any). Phase 4 of the plan enumerates and documents the ignore list.

### 6. Dependabot config

Opens PRs weekly for dep updates: Python deps, GitHub Actions, and Docker base images.

Config: `.github/dependabot.yml` with three ecosystem entries (`pip` or `uv`, `github-actions`, `docker`).

**Verification required:** GitHub added `uv` ecosystem support to Dependabot relatively recently. If the installed Dependabot version doesn't recognize `uv.lock`, fall back to `pip` ecosystem — the lockfile format is compatible enough for CVE tracking, even if not perfect for pin updates. Phase 1 of the plan verifies this before committing to the config shape.

Update cadence: weekly. Grouped updates for minor/patch versions; individual PRs for majors so reviewers can assess breaking-change risk.

### 7. Branch protection on `main`

GitHub repository setting, not a code change. Enforces:

- Pull request review required before merge (at least 1 reviewer).
- Required status checks must pass: the existing `ci.yml` workflow, plus any new checks added by items 1-6 above.
- Force-pushes to `main` blocked.
- Direct pushes to `main` blocked (merges only).

Exceptions: repo admins retain the ability to override in emergencies, but every override is visible in the audit log.

**Operational verification required.** Branch protection is a critical control that lives in GitHub's UI, not in the repo. Settings can drift: someone tweaks a rule, GitHub releases a UI change that resets a default, an admin disables a check temporarily and forgets to re-enable it. The spec mitigates drift with two ingredients:

1. **Snapshot committed to the repo.** At setup time, export the protection settings via `gh api repos/OWNER/REPO/branches/main/protection` and commit the output as `docs/security/branch-protection-snapshot.json`. Also add a short human-readable summary at `docs/security/branch-protection.md` explaining the intended ruleset.
2. **Periodic drift check.** A weekly scheduled GitHub Action (`.github/workflows/branch-protection-drift-check.yml`) fetches current settings via `gh api` and diffs them against the committed snapshot. Any drift fails the job and (via `${{ secrets.SLACK_WEBHOOK_URL }}` or equivalent) posts an alert. Planned drift is legitimized by updating the snapshot in a PR, reviewed like any other change.

The implementation plan treats both snapshot and drift-check workflow as deliverables, alongside the manual GitHub-settings configuration.

### 8. `uv sync --locked` in CI

Verify that `uv.lock` is in sync with `pyproject.toml`. Catches the case where a developer changed `pyproject.toml` but forgot to regenerate the lockfile.

Implementation: the existing `uv sync` step in `ci.yml` becomes `uv sync --locked`. If lockfile is out of sync, the step fails.

May already be implicit depending on the installed `uv` version. Phase 5 of the plan verifies current behavior and adjusts only if needed.

## Approach notes

### Why thresholds at these numbers

- **100% on changed lines**: "test what you write" is the simplest rule. `# pragma: no cover` and admin-applied `coverage-override-approved` label are the intentional-exception mechanisms; both are auditable. Avoids threshold-drift from permissive numbers. See the rationale and escape-valve semantics in §Pre-launch guardrails item 1.
- **Weekly Dependabot cadence**: aggressive enough to keep deps fresh, rare enough to not drown reviewers. Daily PRs are noise at this project's scale.

### Why these guardrails and not others

Considered and explicitly excluded from pre-launch (deferred to post-launch):

- **MCP tool-schema contract tests** — most valuable when MCP schemas are changing frequently (Phase 4 / UKY integration). Defer until that work starts.
- **Docker image vulnerability scan (Trivy)** — valuable but adds CI minutes; not about catching new-code quality slippage.
- **CodeQL / Semgrep** — conditional on repo plan; low marginal value on top of `uv audit`.
- **Mutation testing** — slow, high-maintenance. Right tool for correctness-critical paths, not a pre-launch guardrail.
- **Performance regression gates** — eval infrastructure doesn't support the comparison shape yet; see launch spec Phase 7 for related work.
- **CODEOWNERS file** — ceremony at current team size; worth revisiting when team grows.
- **Signed commits requirement** — high developer friction, marginal security value at current trust model.
- **SBOM generation** — compliance-driven; no current requirement.

## Post-launch hardening (for the launch spec's existing post-launch section)

Items deferred here are additive to the list already in `2026-04-21-production-launch-hardening-design.md`:

- Codebase-wide coverage threshold (stretch goal as coverage grows).
- Mutation testing for correctness-critical paths (classifier, routing, synthesis reconciliation).
- Contract tests against UKY endpoints once they exist.
- Docker image vulnerability scan in CI (Trivy or similar).
- CodeQL / Semgrep scanning.
- Performance regression gates in the eval pipeline.
- Honeycomb alerting for error-rate spikes.
- Structured log schema validation.
- Span attribute schema validation.
- Python patch-version pinning in CI.
- SBOM generation (when compliance requires).
- Secret rotation checklist (docs work).
- CHANGELOG.md if the Conventional Commits output needs consolidation.
- CODEOWNERS file when team grows past ~5 active contributors.

## Deliverables

- `.github/workflows/ci.yml` updated: coverage report, coverage comment, 100% diff-gate (with `coverage-override-approved` label bypass), `uv audit` (fails on HIGH/CRITICAL), gitleaks job, `uv sync --locked`.
- `.github/workflows/nightly.yml` created (if Strategy A chosen for any skipped tests; includes issue-opening or Slack-alert on failure).
- `.github/workflows/branch-protection-drift-check.yml` created — weekly scheduled diff against the committed snapshot.
- `.pre-commit-config.yaml` updated: gitleaks hook.
- `.gitleaks.toml` created with documented allowlist.
- `.audit-ignore.yml` (or equivalent format) for reviewed CVE suppressions, empty initially.
- `.github/dependabot.yml` created.
- `tests/test_e2e.py` and `tests/test_classify.py` resolved per Strategy B (stubbed) or Strategy A (migrated to nightly), each with a documented decision rationale.
- Branch protection applied on `main` (manual GitHub config) + `docs/security/branch-protection-snapshot.json` (exported via `gh api`) + `docs/security/branch-protection.md` (human-readable summary).
- Plan file: `docs/superpowers/plans/2026-04-22-code-quality-guardrails.md` (bite-sized TDD tasks).

## Relationship to Parallel Work

- **Launch hardening** (`2026-04-21-production-launch-hardening-design.md`): these guardrails go in before Phase 3 of that plan (tool-calling loop work) so the new build lands on guardrails-present CI. Items 7 (branch protection) and 8 (`--locked`) can land immediately; others take a PR each.
- **Eval rubric** (`2026-04-21-eval-rubric-ground-truth-design.md`): unaffected. Rubric work is eval-layer code that also benefits from the diff-gate but doesn't need special treatment.

## Open Questions for Implementation

1. Does Dependabot's current `uv` support read `uv.lock` correctly? Phase 1 verifies before committing to ecosystem config.
2. Is `uv audit` a current CLI command or still pending? If pending, fall back to `pip-audit`. Phase 1 verifies.
3. Which strategy (A: nightly, or B: stubbing) for each of `test_e2e.py` and `test_classify.py`? Decision made in Phase 3 of the implementation plan by whoever owns those tests.
4. Does CI already enforce `--locked`? Phase 5 greps the existing workflow before adding.

These are verified and resolved during execution; none block the spec.
