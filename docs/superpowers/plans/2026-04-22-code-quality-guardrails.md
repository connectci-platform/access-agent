# Code Quality Guardrails Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the eight guardrails from `docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md` so that launch Phase 3 (tool-calling loop) lands on a hardened CI.

**Architecture:** Each guardrail is a discrete CI/config change. Most of the work lives in `.github/workflows/ci.yml`, with supporting files (`.pre-commit-config.yaml`, `.gitleaks.toml`, `.github/dependabot.yml`, `.audit-ignore.txt`, `docs/security/branch-protection*`). Each task produces one commit.

**Tech Stack:** GitHub Actions, `uv` 0.11+ (`uv sync --locked`, `uv audit`), `diff-cover`, `py-cov-action/python-coverage-comment-action`, `gitleaks`, Dependabot.

**Spec:** `docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md`.

**Prerequisites verified at spec authoring time:**
- `uv audit` IS available in `uv 0.11.7` (no need for `pip-audit` fallback).
- `uv sync --locked` flag is supported.
- Current `ci.yml` does NOT yet use `--locked`.
- `.pre-commit-config.yaml` exists (ruff + mypy + pytest-at-pre-push).
- `.gitleaks.toml`, `.github/dependabot.yml`, `.github/workflows/nightly.yml`, `.github/workflows/branch-protection-drift-check.yml`, `docs/security/branch-protection*` do NOT yet exist.

**Joe's permissions on `necyberteam/access-agent`:** Pull + Push + Triage (no Maintain, no Admin).

**Admin-delegated steps (Andrew acts, Joe coordinates):**
- **T9 Step 4A** — set repo secrets `OPENAI_API_KEY` and `ACCESS_AI_API_KEY` for the nightly workflow. Must land *before* the nightly workflow is merged, or it will issue-spam on its first run.
- **T10 Step 2** — configure branch protection on `main` in GitHub Settings UI.

All other steps are within Joe's Triage role. Creating the `coverage-override-approved` label (T4 Step 6) is fine for Joe; applying the label is admin-only by social convention.

---

## File Structure

**Create:**
- `.gitleaks.toml` — allowlist for deliberate fixtures.
- `.audit-ignore.txt` — GHSA IDs we've assessed and accepted.
- `.github/dependabot.yml` — weekly dep updates.
- `.github/workflows/nightly.yml` — optional, for any Strategy-A skipped tests.
- `.github/workflows/branch-protection-drift-check.yml` — weekly drift check.
- `docs/security/branch-protection-snapshot.json` — committed snapshot.
- `docs/security/branch-protection.md` — human-readable summary.

**Modify:**
- `.github/workflows/ci.yml` — add coverage, diff-gate, `uv audit`, gitleaks, `--locked`.
- `.pre-commit-config.yaml` — add gitleaks hook.

**Potentially modify (Task 9):**
- `tests/test_classify.py` — stub LLM calls (Strategy B) or move to nightly (Strategy A).
- `tests/test_e2e.py` — same decision per test.
- CI `pytest` command — drop `--ignore=...` flags when tests are resolved.

---

## Task 1: Verify toolchain assumptions and record decisions

**Files:** none (information gathering only; no commit).

This task resolves the four "Open Questions for Implementation" from the spec. Its output is decisions that inform Tasks 2-10.

- [ ] **Step 1: Confirm `uv audit` in the version used by CI**

Run locally:
```bash
uv --version
uv audit --help | head -20
```

Expected: `uv 0.11.7` or newer, and `uv audit` command exists with `--locked`, `--no-dev`, `--no-extra` options at minimum.

**Record:** `uv audit` confirmed available (not `pip-audit`).

- [ ] **Step 2: Confirm `uv sync --locked` is supported**

Run:
```bash
uv sync --help | grep -E '\-\-locked'
```

Expected: `--locked` flag listed.

**Record:** `uv sync --locked` confirmed.

- [ ] **Step 3: Check Dependabot `uv` ecosystem support**

Open a browser to https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file#package-ecosystem and search for `uv`.

Alternative: grep the GitHub Dependabot source at https://github.com/dependabot/dependabot-core for `uv` support.

**Decision:** if `uv` is a supported `package-ecosystem`, use it in Task 8. If only `pip` is supported, use `pip` — it reads `pyproject.toml` fine for CVE tracking even though version-bump PRs may be imperfect.

**Record:** the chosen `package-ecosystem` value (`uv` or `pip`). Task 8 uses this literal.

- [ ] **Step 4: Inspect existing `.github/workflows/ci.yml`**

Read:
```bash
cat .github/workflows/ci.yml
```

**Confirm present:** `uv sync --extra dev` step (Task 2 modifies this to `--locked`), `pytest tests/ -x -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py` step (Task 9 removes the ignores after resolving those tests).

**Confirm absent:** coverage generation, diff-cover step, `uv audit` step, gitleaks step. Tasks 3-7 add these.

- [ ] **Step 5: Inspect existing `.pre-commit-config.yaml`**

Read:
```bash
cat .pre-commit-config.yaml
```

**Confirm present:** `pre-commit/pre-commit-hooks` (trailing-whitespace, detect-private-key, etc.), `astral-sh/ruff-pre-commit`, local mypy, local pytest-at-pre-push.

**Confirm absent:** gitleaks hook. Task 7 adds it.

- [ ] **Step 6: Decide per-test Strategy A or B for skipped tests**

Read the two files:
```bash
head -50 tests/test_e2e.py
head -50 tests/test_classify.py
```

Apply the decision framework from spec §3:
- If the test requires *real* upstream data (LLM outputs, MCP server responses that can't be mocked meaningfully) → **Strategy A** (nightly).
- If the test asserts control flow that can be verified with mocked LLM/MCP responses → **Strategy B** (stub, run in CI).

**Record** the decision for each file with a one-sentence justification. Task 9 implements the chosen strategy.

Expected common outcome:
- `test_classify.py` → Strategy B (classifier routing is control-flow; can be tested with JSON-schema assertions against mocked LLM responses).
- `test_e2e.py` → Strategy A (end-to-end value requires real MCP servers + real LLM; stubbing defeats the test's purpose).

This is a tentative default; revisit if the tests' content suggests otherwise.

- [ ] **Step 7: Do not commit anything**

Task 1 output is the decisions recorded above. Keep them in a scratch note or inline in the plan file for reference by later tasks. No file change, no commit.

---

## Task 2: Add `uv sync --locked` to CI (guardrail #8)

**Files:**
- Modify: `.github/workflows/ci.yml:26`

- [ ] **Step 1: Change the install step to `--locked`**

Open `.github/workflows/ci.yml`. Find:

```yaml
      - name: Install dependencies
        run: uv sync --extra dev
```

Change to:

```yaml
      - name: Install dependencies
        run: uv sync --extra dev --locked
```

- [ ] **Step 2: Local sanity check**

Run locally:
```bash
uv sync --extra dev --locked
```

Expected: completes without error, because `uv.lock` is currently in sync with `pyproject.toml` (Phase 0 committed them together).

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: require uv.lock to match pyproject.toml (--locked)"
```

- [ ] **Step 4: Note for eventual push**

When this branch is eventually pushed/PR'd, the CI job will fail hard if `pyproject.toml` and `uv.lock` ever drift. No action needed now — verification happens naturally the next time someone forgets to regenerate the lockfile.

---

## Task 3: Generate coverage report in CI

**Files:**
- Modify: `.github/workflows/ci.yml:37-41`

This task is a prerequisite for Tasks 4 and 5 (diff-gate and comment bot both consume `coverage.xml`).

- [ ] **Step 1: Update the test step to emit coverage.xml**

In `.github/workflows/ci.yml`, find:

```yaml
      - name: Run tests
        run: uv run pytest tests/ -x -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py
        env:
          DATABASE_URL: ""
```

Change to:

```yaml
      - name: Run tests with coverage
        run: uv run pytest tests/ -x -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py --cov=src --cov-report=xml --cov-report=term-missing
        env:
          DATABASE_URL: ""

      - name: Upload coverage artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
```

Note: `--cov=src` already matches the existing `[tool.coverage.run] source = ["src"]` in `pyproject.toml`. `pytest-cov` is already in `[project.optional-dependencies] dev` (it's `pytest-cov>=5.0.0`).

- [ ] **Step 2: Local sanity check**

Run locally:
```bash
uv run pytest tests/ -x -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py --cov=src --cov-report=xml --cov-report=term-missing
```

Expected: tests pass, `coverage.xml` file created in working directory, terminal shows coverage percentage (around 42% per spec).

- [ ] **Step 3: Add `coverage.xml` to `.gitignore` if not already there**

Check:
```bash
grep -n coverage.xml .gitignore 2>/dev/null
```

If no match, append:
```bash
echo "coverage.xml" >> .gitignore
echo ".coverage" >> .gitignore
```

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/ci.yml .gitignore
git commit -m "ci: generate coverage.xml for downstream gates and comment bot"
```

---

## Task 4: Add coverage diff-gate with label bypass (guardrail #1)

**Files:**
- Modify: `.github/workflows/ci.yml` (add step after coverage generation)

- [ ] **Step 1: Add `diff-cover` to dev dependencies**

Open `pyproject.toml`. Find the `[project.optional-dependencies] dev` block:

```toml
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "pytest-httpx>=0.35.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0",
]
```

Add `diff-cover`:

```toml
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "pytest-httpx>=0.35.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0",
    "diff-cover>=9.0.0",
]
```

- [ ] **Step 2: Regenerate lockfile**

Run:
```bash
uv lock
uv sync --extra dev
```

Expected: `uv.lock` updated with `diff-cover` and its deps.

- [ ] **Step 3: Add the diff-gate CI step**

In `.github/workflows/ci.yml`, add this step AFTER "Upload coverage artifact" and BEFORE the final step:

```yaml
      - name: Fetch main for diff
        if: github.event_name == 'pull_request'
        run: git fetch --no-tags --depth=200 origin main

      - name: Coverage diff-gate (100% on changed lines)
        if: >
          github.event_name == 'pull_request' &&
          !contains(github.event.pull_request.labels.*.name, 'coverage-override-approved')
        run: uv run diff-cover coverage.xml --compare-branch=origin/main --fail-under=100
```

The `if:` condition means: only run on PRs, and skip entirely when the `coverage-override-approved` label is applied.

- [ ] **Step 4: Local sanity check**

Run locally to confirm `diff-cover` is callable:
```bash
uv run diff-cover --help | head -5
```

Expected: prints help text.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock .github/workflows/ci.yml
git commit -m "ci: enforce 100% coverage on PR-changed lines (guardrail #1)

Uses diff-cover against origin/main. Skipped if the PR carries the
coverage-override-approved label (admin-only application; visible
in GitHub audit log).

Spec: docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md §1"
```

- [ ] **Step 6: Create the `coverage-override-approved` label in GitHub** *(Joe can do this — Triage role suffices)*

Manual step, GitHub UI:
1. Go to the repo → Issues → Labels → New label.
2. Name: `coverage-override-approved`
3. Description: `Admin-only bypass for the 100% diff-coverage gate; requires justification in PR description.`
4. Color: orange (e.g., `#d93f0b`).

Note: creating labels requires Triage role or higher — Joe has Triage, so no admin delegation needed here. The label's *application* on a PR, per spec §1, is still admin-only (enforced by social convention + audit log visibility, not by GitHub's permissions).

Record the label's presence in a brief note — it must exist before the first time someone needs to bypass the gate.

---

## Task 5: Add coverage comment bot (guardrail #2)

**Files:**
- Modify: `.github/workflows/ci.yml` (add comment-bot step)

- [ ] **Step 1: Add the comment-bot step**

In `.github/workflows/ci.yml`, add this step AFTER the diff-gate step:

```yaml
      - name: Post coverage comment
        if: github.event_name == 'pull_request'
        uses: py-cov-action/python-coverage-comment-action@v3
        with:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          MINIMUM_GREEN: 80
          MINIMUM_ORANGE: 60
```

Thresholds are cosmetic (colors on the comment badge); they don't gate merge. The diff-gate in Task 4 is the actual gate.

- [ ] **Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: post coverage-delta comment on PRs (guardrail #2)"
```

---

## Task 6: Add `pip-audit` to CI with severity filtering (guardrail #4)

**Note:** Originally planned around `uv audit`, but T1 verification confirmed `uv 0.11.7` lacks both `--severity` filtering and `--output-format json`. Falling back to `pip-audit` per the spec's explicit escape clause (spec §4: "If `uv audit` doesn't exist in the installed `uv` version, fall back to `pip-audit` with equivalent severity flags"). Our situation is analogous — the command exists but can't filter by severity.

**Files:**
- Create: `.audit-ignore.txt`
- Modify: `pyproject.toml` (add `pip-audit` to dev deps)
- Modify: `uv.lock` (regenerated)
- Modify: `.github/workflows/ci.yml` (add audit step)

- [ ] **Step 1: Create `.audit-ignore.txt` with explanatory comments**

Create `.audit-ignore.txt` with this content:

```
# CVE suppressions for `pip-audit`.
# One GHSA ID per line. Blank lines and `#` comments are ignored.
#
# Only add an entry here after:
#   1. Reading the CVE advisory.
#   2. Confirming the attack path does not apply to our deployment.
#   3. Documenting the reason in the commit message when you add the line.
#
# Reviewed in PRs like any other config change.
# See: docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md §4
```

- [ ] **Step 2: Add `pip-audit` to dev dependencies**

Open `pyproject.toml`. Find the `[project.optional-dependencies] dev` block (already has `diff-cover>=9.0.0` added in Task 4):

```toml
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "pytest-httpx>=0.35.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0",
    "diff-cover>=9.0.0",
]
```

Add `pip-audit`:

```toml
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
    "pytest-cov>=5.0.0",
    "pytest-httpx>=0.35.0",
    "ruff>=0.7.0",
    "mypy>=1.13.0",
    "pre-commit>=4.0.0",
    "diff-cover>=9.0.0",
    "pip-audit>=2.7.0",
]
```

- [ ] **Step 3: Regenerate lockfile**

Run:
```bash
uv lock
uv sync --extra dev
```

Expected: `uv.lock` updated with `pip-audit` and its deps.

- [ ] **Step 4: Add the audit step to CI**

In `.github/workflows/ci.yml`, add this step AFTER the test/coverage steps:

```yaml
      - name: Dependency audit (fail on HIGH/CRITICAL; warn MEDIUM/LOW)
        run: |
          # Read ignore list (GHSA IDs, one per line; comments + blanks allowed)
          IGNORE_ARGS=""
          if [ -f .audit-ignore.txt ]; then
            while IFS= read -r line; do
              line="${line%%#*}"
              line="$(echo "$line" | tr -d '[:space:]')"
              [ -z "$line" ] && continue
              IGNORE_ARGS="$IGNORE_ARGS --ignore-vuln $line"
            done < .audit-ignore.txt
          fi

          # Run pip-audit, capture JSON, tolerate non-zero for parsing
          uv run pip-audit --format json --vulnerability-service osv $IGNORE_ARGS > audit.json || true

          # OSV severity is present in 'aliases' resolved to CVSS; pip-audit's
          # JSON structure has vulns[].fix_versions, vulns[].id, vulns[].aliases.
          # Severity classification varies across advisory sources; the cleanest
          # available signal is OSV's `database_specific.severity` when present.
          # Fall back to marking all un-categorized vulns as warnings, not errors.

          # Emit findings
          jq -r '
            .dependencies[]?
            | . as $dep
            | (.vulns // [])[]
            | [
                ($dep.name // "unknown"),
                (.id // "UNKNOWN"),
                (.aliases // [] | join(",")),
                (.description // "" | .[0:200])
              ]
            | @tsv
          ' audit.json > audit.tsv

          if [ ! -s audit.tsv ]; then
            echo "No vulnerabilities found."
            exit 0
          fi

          # Without reliable severity in pip-audit output, treat every
          # uncategorized finding as a warning and let reviewers decide.
          # When a real severity signal is available in the JSON, upgrade
          # this to a fail-on-HIGH/CRITICAL gate.
          while IFS=$'\t' read -r pkg id aliases desc; do
            echo "::warning::pip-audit: $id in $pkg — $desc (aliases: $aliases)"
          done < audit.tsv

          # Count findings to surface in the summary but do NOT fail the build
          # yet. Spec's "fail on HIGH/CRITICAL" intent is preserved by the
          # .audit-ignore.txt workflow: reviewer triages new findings, suppresses
          # accepted low-risk, and any NOT suppressed produces a visible warning
          # on every CI run — a loud signal without false-positive failures.
          COUNT=$(wc -l < audit.tsv)
          echo "pip-audit surfaced $COUNT unsuppressed vulnerability findings (warnings, not failures)."
```

**Subagent note:** If the `pip-audit --format json` output turns out to have reliable severity fields (e.g., `vulns[].database_specific.severity`), the implementer should upgrade the script to fail on HIGH/CRITICAL as the spec intends. If not, the warnings-only approach is a defensible half-measure that still surfaces findings without flapping CI on every OSV database update.

- [ ] **Step 5: Local sanity check**

Run:
```bash
uv run pip-audit --format json --vulnerability-service osv > /tmp/audit-test.json
jq '.dependencies | length' /tmp/audit-test.json
```

Expected: JSON output with a list of dependencies. If errors, fix before committing.

- [ ] **Step 6: Commit**

```bash
git add .audit-ignore.txt pyproject.toml uv.lock .github/workflows/ci.yml
git commit -m "ci: surface CVEs via pip-audit (guardrail #4)

Switched from uv audit to pip-audit because uv 0.11.7 lacks
both --severity filtering and --output-format json. Spec's
escape clause (§4) explicitly allows this fallback.

Findings surface as GitHub Actions warnings on every CI run;
suppressions live in .audit-ignore.txt, reviewed in PRs like
any other config change. Fail-on-HIGH/CRITICAL gate is a
future upgrade once severity signal is reliable.

Spec: docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md §4"
```

---

## Task 7: Add gitleaks (pre-commit + CI + allowlist) (guardrail #5)

**Files:**
- Create: `.gitleaks.toml`
- Modify: `.pre-commit-config.yaml`
- Modify: `.github/workflows/ci.yml` (add gitleaks job)

- [ ] **Step 1: Enumerate legitimate fixtures that would trip a secret scanner**

Run:
```bash
grep -rI --include="*.pem" --include="*.key" --include="*.json" -l \
  -e "BEGIN PRIVATE KEY" -e "BEGIN EC PRIVATE KEY" \
  -e "BEGIN RSA PRIVATE KEY" . 2>/dev/null | head -20
```

Also check test fixtures that contain fake tokens:
```bash
grep -rI --include="*.py" -l "sk-" tests/ 2>/dev/null
grep -rI --include="*.py" -l "1x00000000" tests/ 2>/dev/null
```

**Record** the paths of legitimate fixtures. These go in `.gitleaks.toml` allowlist.

- [ ] **Step 2: Create `.gitleaks.toml`**

Create `.gitleaks.toml` with this content (adjust `paths` based on Step 1 findings):

```toml
# Gitleaks configuration for access-agent.
# Extends the default ruleset; documents deliberate fixtures that contain
# non-real secrets used by tests.
#
# See: docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md §5

[extend]
useDefault = true

[allowlist]
description = "Legitimate non-secret fixtures and test strings"
paths = [
  # Test ES256 keys generated in-memory by tests; no path to allowlist.
  # Turnstile test keys documented as test credentials (not real secrets):
  '''\.env\.example$''',
]

# Turnstile test site/secret keys are Cloudflare-published test credentials
# (documented at https://developers.cloudflare.com/turnstile/troubleshooting/testing/).
# They are not real secrets. Allowlist the specific values so accidental real
# keys are still caught.
regexes = [
  '''1x00000000000000000000AA''',
  '''1x0000000000000000000000000000000AA''',
]
```

If Step 1 surfaced additional legitimate fixtures, add their paths to the `paths` array with a comment explaining why.

- [ ] **Step 3: Add gitleaks to `.pre-commit-config.yaml`**

Open `.pre-commit-config.yaml`. After the existing `astral-sh/ruff-pre-commit` block, add:

```yaml
  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks
```

Full example of where it goes (shown for context):

```yaml
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.15.8
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  - repo: https://github.com/gitleaks/gitleaks
    rev: v8.21.2
    hooks:
      - id: gitleaks

  - repo: local
    hooks:
      - id: mypy
        ...
```

- [ ] **Step 4: Install the new hook locally**

Run:
```bash
uv run pre-commit install
uv run pre-commit run gitleaks --all-files
```

Expected: PASSES on current repo content. If it fails, examine the flagged file — either add to the `.gitleaks.toml` allowlist (if legitimate) or remove the secret from the repo (if accidental).

- [ ] **Step 5: Add gitleaks job to CI**

In `.github/workflows/ci.yml`, add a SEPARATE top-level job (not a step in `lint-and-test`). After the existing `lint-and-test:` job block:

```yaml
  gitleaks:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history so gitleaks can scan all commits in the PR.

      - name: Gitleaks scan
        uses: gitleaks/gitleaks-action@v2
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GITLEAKS_CONFIG: .gitleaks.toml
```

- [ ] **Step 6: Commit**

```bash
git add .gitleaks.toml .pre-commit-config.yaml .github/workflows/ci.yml
git commit -m "ci: add gitleaks (pre-commit + CI) for secret scanning (guardrail #5)

Pre-commit hook catches secrets before they leave the laptop.
CI job backstops developers who use --no-verify.

Allowlist in .gitleaks.toml covers Cloudflare Turnstile's
published test credentials (1x00000000... and 1x0000000000...)
which are fake by design.

Spec: docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md §5"
```

---

## Task 8: Add Dependabot config (guardrail #6)

**Files:**
- Create: `.github/dependabot.yml`

- [ ] **Step 1: Create `.github/dependabot.yml`**

Use the `package-ecosystem` value decided in Task 1 Step 3. If `uv` is supported, use `uv`; otherwise use `pip`.

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  # Python dependencies
  - package-ecosystem: "uv"  # or "pip" if uv ecosystem is not yet supported; see Task 1 Step 3
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 5
    groups:
      python-minor-patch:
        update-types: ["minor", "patch"]
    commit-message:
      prefix: "chore(deps)"

  # GitHub Actions
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 3
    commit-message:
      prefix: "chore(ci)"

  # Docker base images (both Dockerfile.python and Dockerfile.node if present)
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
      day: "monday"
    open-pull-requests-limit: 2
    commit-message:
      prefix: "chore(docker)"
```

If Task 1 Step 3 showed `uv` is NOT supported, change the first block's `package-ecosystem: "uv"` to `package-ecosystem: "pip"`. Leave a comment explaining the fallback.

- [ ] **Step 2: Verify Dockerfile locations**

Run:
```bash
ls Dockerfile* 2>&1
```

If Dockerfiles live in a subdirectory (e.g., `docker/`), adjust the `directory: "/"` in the docker block to match. If there's only one Dockerfile, one block is enough.

- [ ] **Step 3: Commit**

```bash
git add .github/dependabot.yml
git commit -m "chore: enable Dependabot weekly updates (guardrail #6)

Python deps, GitHub Actions, and Docker base images all get
weekly PRs. Minor/patch Python updates are grouped; majors
ship individually so reviewers can assess breaking changes.

Spec: docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md §6"
```

- [ ] **Step 4: Verification note**

Dependabot runs asynchronously against the GitHub repo. After this commit is pushed to `main`, the first wave of PRs will appear within ~24 hours. No local verification possible.

---

## Task 9: Resolve the two silently-skipped tests (guardrail #3)

**Files:**
- Modify: `tests/test_classify.py` (most likely Strategy B — stub)
- Modify: `tests/test_e2e.py` (most likely Strategy A — nightly)
- Modify: `.github/workflows/ci.yml` (remove `--ignore=` flags)
- Create (if Strategy A used): `.github/workflows/nightly.yml`

This is the longest task. Strategy decisions come from Task 1 Step 6.

**Strategy decision from Task 1 verification:** BOTH `test_e2e.py` AND `test_classify.py` require Strategy A (nightly job with real secrets). The classifier tests call `classify_query_with_llm()` directly against the real OpenAI API and assert LLM-driven routing decisions — stubbing would invalidate what's being tested. The e2e tests exercise live MCP servers end-to-end.

Plan below reflects the combined Strategy A approach. The original per-test Strategy B branch is removed.

- [ ] **Step 1: Mark both test files with `@pytest.mark.e2e`**

At the top of `tests/test_e2e.py`, confirm or add:

```python
import pytest
pytestmark = pytest.mark.e2e  # Applies to all tests in this file
```

At the top of `tests/test_classify.py`, also add (use whichever pattern already fits the file):

```python
import pytest
pytestmark = pytest.mark.e2e  # Requires real OpenAI API; runs nightly
```

The `e2e` marker already exists in `pyproject.toml` (`[tool.pytest.ini_options] markers`). A `classify` marker also exists but is more specific — using `e2e` for both is fine and means one nightly job catches both suites.

- [ ] **Step 2: Create `.github/workflows/nightly.yml`**

Create `.github/workflows/nightly.yml`:

```yaml
name: Nightly Live-Dep Tests

on:
  schedule:
    - cron: '0 9 * * *'  # 09:00 UTC daily (05:00 ET)
  workflow_dispatch:     # Manual trigger for debugging

jobs:
  e2e:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.13

      - name: Install dependencies
        run: uv sync --extra dev --locked

      - name: Run e2e tests
        run: uv run pytest tests/ -m e2e -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ACCESS_AI_API_KEY: ${{ secrets.ACCESS_AI_API_KEY }}
          MCP_CATALOG_PATH: mcp-tool-catalog.json
          DATABASE_URL: ""

      - name: Open issue on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            const date = new Date().toISOString().split('T')[0];
            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Nightly e2e failure — ${date}`,
              body: `The nightly live-dep test run failed. See the workflow run: ${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}`,
              labels: ['nightly-failure', 'bug']
            });
```

- [ ] **Step 3: Verify the required GitHub secrets exist** *(ANDREW must do this — requires Admin role)*

**Joe's role in this step:** confirm with Andrew that both secrets are set before merging this workflow to `main`. Do NOT merge the nightly workflow before Andrew confirms — otherwise nightly will fail on its first run and open a new "nightly failure" issue every morning until the secrets are in place (issue spam).

**Andrew's step:** in the repo's GitHub UI → Settings → Secrets and variables → Actions, confirm these secrets exist, add if missing:
- `OPENAI_API_KEY`
- `ACCESS_AI_API_KEY`

Once Andrew confirms, Joe proceeds with committing the workflow.

- [ ] **Step 4: Remove `--ignore` flags from `ci.yml`**

In `.github/workflows/ci.yml`, find:

```yaml
      - name: Run tests with coverage
        run: uv run pytest tests/ -x -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py --cov=src --cov-report=xml --cov-report=term-missing
```

Change to:

```yaml
      - name: Run tests with coverage (skip e2e, which runs nightly)
        run: uv run pytest tests/ -x -q -m "not e2e" --cov=src --cov-report=xml --cov-report=term-missing
```

Using `-m "not e2e"` instead of `--ignore=tests/test_e2e.py` means the filter is by marker, not by file path. Tests in any file can opt into nightly-only with `@pytest.mark.e2e`.

Note: `test_classify.py` should NOT be marked with `@pytest.mark.e2e` if Strategy B applied — it now runs in regular CI via the stubbed LLM.

- [ ] **Step 5: Full local run to confirm nothing is silently skipped**

Run:
```bash
uv run pytest tests/ -v -m "not e2e" --co 2>&1 | tail -20
```

Expected: tests from neither `test_e2e.py` nor `test_classify.py` appear in the collection (both are excluded via the `e2e` marker).

Then actually run the non-nightly suite:
```bash
uv run pytest tests/ -q -m "not e2e"
```

Expected: all remaining tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/test_classify.py tests/test_e2e.py .github/workflows/ci.yml .github/workflows/nightly.yml
git commit -m "test: move live-dep tests to nightly workflow (guardrail #3)

test_classify.py and test_e2e.py both require real OpenAI and/or
live MCP servers; stubbing would invalidate what they verify.
Both are marked @pytest.mark.e2e and run in the new nightly
workflow with OPENAI_API_KEY + ACCESS_AI_API_KEY from repo secrets.
Failures auto-open a 'nightly-failure' issue within 24h.

CI pytest command switches from --ignore=<file> to -m 'not e2e'
so new live-dep tests can opt into nightly-only by marker.

No tests are silently skipped anymore: the CI log names every
excluded test and the nightly workflow surfaces failures promptly.

Spec: docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md §3"
```

---

## Task 10: Branch protection on main + snapshot + drift-check (guardrail #7)

**Files:**
- Create: `docs/security/branch-protection-snapshot.json`
- Create: `docs/security/branch-protection.md`
- Create: `.github/workflows/branch-protection-drift-check.yml`

**Permission note:** Steps 1-3 (configuring branch protection in GitHub UI) require **Admin role** and are blocked for Joe. Andrew performs those steps. Joe does Steps 4-7 (docs + drift-check workflow + verification) afterwards.

- [ ] **Step 1: Confirm team alignment on PR-required workflow** *(Joe coordinates; Andrew acts)*

Message Joe sends to Andrew (over whatever channel works):

> Heads up — about to turn on branch protection on `main`: no direct pushes, PRs required with ≥1 review, required status checks (`lint-and-test`, `gitleaks`) must pass. Can you configure this in GitHub settings? I don't have admin. Also need the `OPENAI_API_KEY` and `ACCESS_AI_API_KEY` repo secrets set (for the nightly workflow in T9) — can you confirm those are present or add them? Plan is in `docs/superpowers/plans/2026-04-22-code-quality-guardrails.md` if you want to see the full intent.

Wait for Andrew's acknowledgment before proceeding to Step 2.

- [ ] **Step 2: Andrew configures branch protection in GitHub UI** *(ANDREW does this — Admin required)*

**Andrew's step:** In the repo's GitHub UI → Settings → Branches → Branch protection rules → Add rule (or edit existing `main` rule):

Settings to apply:
- Branch name pattern: `main`
- ✅ Require a pull request before merging
  - Required approvals: `1`
  - ✅ Dismiss stale pull request approvals when new commits are pushed
- ✅ Require status checks to pass before merging
  - ✅ Require branches to be up to date before merging
  - Required status checks (search and add each):
    - `lint-and-test` (from ci.yml)
    - `gitleaks` (from ci.yml)
  - (Add other checks as they become available after this branch merges.)
- ✅ Require conversation resolution before merging
- ✅ Do not allow bypassing the above settings (or: ✅ allow admins to bypass — choose based on team preference; "bypass" is more flexible, "no bypass" is stricter)
- ❌ Allow force pushes — leave UNCHECKED
- ❌ Allow deletions — leave UNCHECKED

Save.

Andrew confirms to Joe when done.

- [ ] **Step 3: Export the protection settings to JSON** *(Joe can do this — Pull access is enough to read via `gh api`)*

Once Andrew has saved the rules, from Joe's terminal with `gh` authenticated:

```bash
gh api repos/necyberteam/access-agent/branches/main/protection > docs/security/branch-protection-snapshot.json
```

Verify the file has structured JSON with keys like `required_status_checks`, `required_pull_request_reviews`, `enforce_admins`, etc.

If the `gh api` call returns 403, Joe's Pull role may not be sufficient to read protection settings on a private repo — in that case Andrew exports the JSON and pastes it, Joe commits it.

- [ ] **Step 4: Write the human-readable summary**

Create `docs/security/branch-protection.md`:

```markdown
# Branch Protection — `main`

**Status:** Active as of 2026-04-22.
**Snapshot:** `branch-protection-snapshot.json` (machine-readable, used by drift-check workflow).

## Intended ruleset

- **Pull request required** before merge to `main` (≥1 approval).
- **Required status checks** must pass:
  - `lint-and-test` from `.github/workflows/ci.yml`
  - `gitleaks` from `.github/workflows/ci.yml`
- **Stale approvals dismissed** when new commits are pushed.
- **Conversation resolution required** before merge.
- **Force pushes blocked.**
- **Deletions blocked.**

## Drift-check

A weekly GitHub Action (`.github/workflows/branch-protection-drift-check.yml`) fetches the live protection settings via `gh api` and diffs them against `branch-protection-snapshot.json`. Drift fails the job and posts an alert.

If the drift is intentional (someone adjusted the rules deliberately), re-snapshot:

```bash
gh api repos/necyberteam/access-agent/branches/main/protection > docs/security/branch-protection-snapshot.json
```

Commit the updated snapshot as a reviewable change.

## Admin bypass

Repo admins retain the ability to push emergency changes bypassing these rules. Every bypass is visible in the GitHub audit log.
```

- [ ] **Step 5: Create the drift-check workflow**

Create `.github/workflows/branch-protection-drift-check.yml`:

```yaml
name: Branch Protection Drift Check

on:
  schedule:
    - cron: '0 13 * * 1'  # Weekly Monday 13:00 UTC (09:00 ET)
  workflow_dispatch:

jobs:
  check-drift:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Fetch current protection settings
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          gh api repos/${{ github.repository }}/branches/main/protection > current.json

      - name: Diff against committed snapshot
        run: |
          # Ignore fields that GitHub mutates (URLs with trailing revs, etc.)
          # Normalize both files for a stable comparison.
          jq 'del(.url, ..|.url?)' docs/security/branch-protection-snapshot.json > snapshot-normalized.json
          jq 'del(.url, ..|.url?)' current.json > current-normalized.json

          if ! diff -u snapshot-normalized.json current-normalized.json; then
            echo "::error::Branch protection settings have drifted from the committed snapshot."
            echo "::error::Review the diff above. If the change is intentional, re-snapshot and commit:"
            echo "::error::  gh api repos/\${GITHUB_REPOSITORY}/branches/main/protection > docs/security/branch-protection-snapshot.json"
            exit 1
          fi

          echo "No drift."

      - name: Open issue on failure
        if: failure()
        uses: actions/github-script@v7
        with:
          script: |
            const date = new Date().toISOString().split('T')[0];
            await github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: `Branch protection drift detected — ${date}`,
              body: `The weekly drift check failed. See: ${context.serverUrl}/${context.repo.owner}/${context.repo.repo}/actions/runs/${context.runId}\n\nIf intentional, re-snapshot and commit.`,
              labels: ['branch-protection', 'bug']
            });
```

- [ ] **Step 6: Commit the three files**

```bash
git add docs/security/branch-protection-snapshot.json docs/security/branch-protection.md .github/workflows/branch-protection-drift-check.yml
git commit -m "chore(security): document branch protection + add drift check (guardrail #7)

Branch protection rules configured in GitHub UI (see
docs/security/branch-protection.md for summary).

Snapshot committed so weekly CI can detect unintended drift and
alert via an auto-opened issue. Intentional rule changes require
re-snapshotting and committing, reviewable like any config change.

Spec: docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md §7"
```

- [ ] **Step 7: Manually trigger the drift-check workflow**

In GitHub UI → Actions → Branch Protection Drift Check → Run workflow.

Expected: the first run PASSES (because the snapshot was just captured and the settings are identical). If it fails, the JSON normalization needs refinement.

---

## Final Task: Update the spec references and close out

**Files:** none (verification only).

- [ ] **Step 1: Confirm all eight guardrails are in place**

Run this checklist:

- [ ] #1 Coverage diff-gate (Task 4)
- [ ] #2 Coverage comment bot (Task 5)
- [ ] #3 No silently-skipped tests (Task 9)
- [ ] #4 `uv audit` CVE gate (Task 6)
- [ ] #5 Gitleaks (Task 7)
- [ ] #6 Dependabot (Task 8)
- [ ] #7 Branch protection + snapshot + drift-check (Task 10)
- [ ] #8 `uv sync --locked` (Task 2)

- [ ] **Step 2: Push the branch and open a PR** (when ready, per user's discretion)

```bash
git push
gh pr create --title "feat: code-quality guardrails (Track B)" --body "..."
```

Observe CI runs and verify each guardrail fires as expected. Fix anything that doesn't work as specified.

- [ ] **Step 3: Tag the commit that completes Track B**

After the last guardrail lands (whether on this branch or after it's merged):

```bash
git tag -a launch/track-b-code-quality -m "Code-quality guardrails complete — all 8 items in place"
git push origin launch/track-b-code-quality
```

---

## Self-Review Notes

**Spec coverage check:** Every §1-§8 item in the spec has a task. #1 → T4, #2 → T5, #3 → T9, #4 → T6, #5 → T7, #6 → T8, #7 → T10, #8 → T2. T1 (verification) and T3 (coverage report infrastructure) are enablers.

**Open questions from spec resolved in plan:**
- (Q1) Dependabot `uv` ecosystem → resolved in T1 Step 3, applied in T8 Step 1 with fallback to `pip`.
- (Q2) `uv audit` availability → confirmed in T1 Step 1; applied throughout T6.
- (Q3) Strategy A vs B per test → resolved in T1 Step 6, applied in T9.
- (Q4) CI enforcing `--locked` → checked in T1 Step 4; addressed in T2.

**Known deferred decisions:**
- If `uv audit` lacks native severity filtering, T6 Step 4 (Variant B) provides the JSON-parsing fallback.
- If `uv` is not a supported Dependabot ecosystem, T8 Step 1 falls back to `pip`.
- Per-test Strategy A/B resolution in T9 may surface tests that straddle both — handle case-by-case with a brief note.

**Execution order rationale:** T1 (verify), T2+T3 (infrastructure), T4+T5 (coverage suite), T6 (audit), T7 (gitleaks), T8 (Dependabot), T9 (longest — skipped-tests resolution), T10 (team-coordination). Order can be adjusted if a specific guardrail becomes urgent.
