# Launch Phase 0 — Dependency Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `langgraph` 0.2→1.x and matching langchain packages to their current major versions, adapting the single breaking change (`create_react_agent` → `create_agent`) without altering agent behavior.

**Architecture:** Pure dependency bump. No feature changes, no behavioral changes. The only production code change is a rename from `create_react_agent` to `create_agent` in `src/agent/nodes/domain_agent.py`. Everything else survives unchanged; the test suite verifies that.

**Tech Stack:** Python 3.11+, `uv`, `langgraph`, `langchain-core`, `langchain-openai`.

**Spec:** `docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md` (Phase 0).

**Umbrella plan:** `docs/superpowers/plans/2026-04-21-production-launch-umbrella.md`.

**Unblocks:** Phase 1 (safety audit + `READ_ONLY` guard), Phase 3 (tool-calling loop).

---

## File Structure

**Modify:**
- `pyproject.toml` — bump three version pins.
- `uv.lock` — regenerated via `uv lock --upgrade`.
- `src/agent/nodes/domain_agent.py` — rename `create_react_agent` to `create_agent` if referenced (verified in Task 3).

**Not touched** (verified green by tests):
- All other `src/**` files.
- All `tests/**` files (unless a test directly imports `create_react_agent`, addressed in Task 3).
- Docker configs, deploy workflows.

---

## Task 1: Create a dedicated branch

**Files:** none.

- [ ] **Step 1: Ensure main is up-to-date**

Run:
```bash
git fetch origin main
git checkout main
git pull --ff-only
```
Expected: at HEAD of `origin/main`.

- [ ] **Step 2: Create feature branch**

Run:
```bash
git checkout -b feature/launch-phase-0-deps-upgrade
```
Expected: on a fresh branch from main.

- [ ] **Step 3: Verify clean baseline**

Run:
```bash
uv sync
uv run pytest -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py
```
Expected: all tests pass on current pins. If not, stop and investigate — the upgrade should start from a green baseline. (`test_e2e.py` and `test_classify.py` are skipped per CI convention; they require live API keys.)

---

## Task 2: Bump the version pins

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update the three relevant pins**

Open `pyproject.toml`. Find the `dependencies = [` block. Update these three lines (keep all other dependencies untouched):

```toml
    "langgraph>=1.1.0,<2.0.0",
    "langchain-core>=1.3.0,<2.0.0",
    "langchain-openai>=1.1.0,<2.0.0",
```

Replace the existing lines:
```
    "langgraph>=0.2.0",
    "langchain-core>=0.3.0",
    "langchain-openai>=0.2.0",
```

All other dependencies and the whole `[project.optional-dependencies]` block remain as-is.

- [ ] **Step 2: Regenerate the lockfile**

Run:
```bash
uv lock --upgrade
```
Expected: the lockfile is updated. No errors. If dependency resolution fails (e.g., a transitive package has an upper bound that conflicts with langchain-core 1.x), stop and surface the exact error; do not force-resolve.

- [ ] **Step 3: Install the new lockfile**

Run:
```bash
uv sync
```
Expected: all packages reinstalled to the new versions. No errors.

- [ ] **Step 4: Record resolved versions in a pre-commit verification step**

Run:
```bash
uv pip show langgraph langchain-core langchain-openai | grep -E '^(Name|Version):'
```
Expected output includes:
```
Name: langgraph
Version: 1.1.x (or whatever 1.x is current)
...
Name: langchain-core
Version: 1.3.x
...
Name: langchain-openai
Version: 1.1.x
```
Record these exact versions for the commit message.

---

## Task 3: Adapt `create_react_agent` → `create_agent`

**Files:**
- Modify: `src/agent/nodes/domain_agent.py` (if it imports `create_react_agent`)
- Possibly modify: `tests/**` (if any test imports `create_react_agent` directly)

- [ ] **Step 1: Run the test suite to surface breakage**

Run:
```bash
uv run pytest -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py 2>&1 | tee /tmp/phase-0-first-run.log
```

Likely outcome: at least one test fails with `ImportError: cannot import name 'create_react_agent' from 'langgraph.prebuilt'`, or a deprecation-turned-error from LangGraph 1.x.

If ALL tests pass as-is, skip to Step 4. The upgrade happened to be a no-op for our usage.

- [ ] **Step 2: Update `domain_agent.py`**

Open `src/agent/nodes/domain_agent.py`. Find the import of `create_react_agent`:

```python
from langgraph.prebuilt import create_react_agent
```

Replace with:

```python
from langgraph.prebuilt import create_agent
```

Find the call site(s):

```python
agent = create_react_agent(model=llm, tools=tools, state_modifier=system_prompt)
# or similar — kwarg names may differ by version
```

The LangGraph 1.x replacement is `create_agent`. Check LangGraph 1.x migration notes (search for "create_react_agent" in their CHANGELOG.md or upgrade guide) for exact kwarg renames. The most likely rename is `state_modifier` → `prompt`, but verify against the actually-installed version's docstring:

```bash
uv run python -c "from langgraph.prebuilt import create_agent; help(create_agent)" | head -40
```

Apply the correct signature to the call. Show the after-edit line in the final diff so reviewers can verify.

- [ ] **Step 3: Search for other callers**

Use the Grep tool (not bash) to find any remaining references:
- Search pattern: `create_react_agent`
- Include glob: `*.py`

Expected: only the `domain_agent.py` reference (and possibly test mocks). If other files are found, apply the same rename.

- [ ] **Step 4: Run the full test suite**

Run:
```bash
uv run pytest -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py
```
Expected: all previously-passing tests pass. If any new failures appear, they're framework-upgrade-related — fix them in this task, don't defer.

- [ ] **Step 5: Quick smoke test on a simple query**

Spin up the agent locally and verify a basic query works end-to-end:

```bash
uv run python test_agent.py "What is ACCESS?"
```

Expected: a coherent textual response about ACCESS-CI. No stack traces. Any deprecation warnings printed to stderr should be captured for later triage but not block this task.

---

## Task 4: Type-check and lint

**Files:** none (verification only).

- [ ] **Step 1: Run mypy**

Run:
```bash
uv run mypy src/
```
Expected: no new errors introduced by the upgrade. Pre-existing errors (if any in the baseline run from Task 1 Step 3) are acceptable; new ones are not.

- [ ] **Step 2: Run ruff**

Run:
```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
```
Expected: both pass. Neither should need changes from an upgrade; if either flags something, the rename in Task 3 probably needs minor cleanup.

---

## Task 5: Commit and push

**Files:** none (git operations).

- [ ] **Step 1: Review the staged diff**

Run:
```bash
git add pyproject.toml uv.lock src/agent/nodes/domain_agent.py
git diff --cached
```
Expected: the diff shows the three pin bumps, a regenerated `uv.lock` (large file but mechanical), and the `create_react_agent` → `create_agent` rename with any kwarg adjustments.

If any files beyond those three show up, investigate — an unintended change may have slipped in.

- [ ] **Step 2: Commit**

Run:
```bash
git commit -m "$(cat <<'EOF'
chore(deps): upgrade langgraph 0.2→1.x, langchain-core/openai to 1.x

Pins updated:
- langgraph: >=0.2.0 → >=1.1.0,<2.0.0
- langchain-core: >=0.3.0 → >=1.3.0,<2.0.0
- langchain-openai: >=0.2.0 → >=1.1.0,<2.0.0

create_react_agent renamed to create_agent in domain_agent.py per
LangGraph 1.x migration guide.

Full test suite passes on new pins. No behavioral changes intended;
this is prerequisite work for the tool-calling loop (Phase 3).

Spec: docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md
EOF
)"
```
Expected: one commit with the three modified files.

- [ ] **Step 3: Push**

Run:
```bash
git push -u origin feature/launch-phase-0-deps-upgrade
```
Expected: branch pushed to origin.

- [ ] **Step 4: Open a PR**

Run:
```bash
gh pr create --title "chore(deps): upgrade langgraph to 1.x (launch Phase 0)" --body "$(cat <<'EOF'
## Summary
- Bumped `langgraph` from 0.2 → 1.x, plus `langchain-core` 0.3 → 1.x and `langchain-openai` 0.2 → 1.x.
- Renamed `create_react_agent` → `create_agent` in `src/agent/nodes/domain_agent.py` per the LangGraph 1.x migration guide.
- No behavior change intended. Full test suite passes on new pins.

Prerequisite for the tool-calling loop work (launch Phase 3).

Spec: `docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md`.
Umbrella plan: `docs/superpowers/plans/2026-04-21-production-launch-umbrella.md`.

## Test plan
- [ ] CI green on the upgraded pins.
- [ ] Local smoke test passes: `uv run python test_agent.py "What is ACCESS?"`.
- [ ] Reviewer spot-checks `domain_agent.py` diff for correct kwarg-rename.
EOF
)"
```
Expected: PR URL printed. Hand off for review.
