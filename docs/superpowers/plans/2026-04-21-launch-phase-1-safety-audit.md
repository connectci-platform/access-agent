# Launch Phase 1 — Safety Audit + `READ_ONLY` Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enumerate every write-capable capability in a committed audit document and add a `READ_ONLY=true` env var that force-disables all write capabilities in `CapabilityRegistry` at startup, with a prominent log line.

**Architecture:** Three deliverables: (1) a markdown audit document at `docs/security/write-capability-audit.md` enumerating write paths and controls; (2) a new `READ_ONLY` setting in `src/config.py`; (3) a small code change in `src/agent/domains/capabilities.py` that, when `READ_ONLY=true`, force-adds the known write-capability IDs into the disabled set before the registry is built. Logs a loud startup line when active.

**Tech Stack:** Python 3.11+, pytest, existing `CapabilityRegistry` and `settings` infrastructure.

**Spec:** `docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md` (Phase 1).

**Umbrella plan:** `docs/superpowers/plans/2026-04-21-production-launch-umbrella.md`.

**Prerequisites:** Phase 0 merged (dependency upgrade).

**Unblocks:** Phase 2 staging (uses `READ_ONLY=true` by default).

---

## Known write-capable capabilities

Verified from source (`src/agent/domains/jsm.py`, `src/agent/domains/announcements.py`):

- `manage_announcements` — create/update/delete announcements (POSTs to announcements MCP).
- `open_ticket` — create a JSM support ticket.
- `report_login_problem` — create a JSM ticket for login issues.
- `report_security` — create a JSM ticket for security concerns.

All four are domain-agent-backed (announcements / jsm) and are only reached when the classifier sets `domain` to the corresponding value AND (for announcements) `capability_id == "manage_announcements"`.

This list is the source of truth for the `READ_ONLY` guard and for the audit document. If new write-capable capabilities are added in the future, they must be added to both `WRITE_CAPABILITY_IDS` (defined in this plan) and to the audit document.

---

## File Structure

**Create:**
- `docs/security/write-capability-audit.md` — the audit document.
- `tests/test_capabilities_read_only.py` — unit tests for the `READ_ONLY` guard.

**Modify:**
- `src/config.py` — add `READ_ONLY: bool = False`.
- `src/agent/domains/capabilities.py` — define `WRITE_CAPABILITY_IDS` constant; extend `_build_registry()` to apply `READ_ONLY` to the disabled set; add startup log.
- `.env.example` — document the new `READ_ONLY` env var.

**Not touched:**
- Classifier prompt rules (already gate write capabilities behind explicit imperative language; the audit documents this behavior, no change needed).
- Existing `ENABLED_CAPABILITIES` / `DISABLED_CAPABILITIES` semantics.
- Domain agent code itself.

---

## Task 1: Create a dedicated branch

**Files:** none.

- [ ] **Step 1: Ensure main is up-to-date with Phase 0 merged**

Run:
```bash
git fetch origin main
git checkout main
git pull --ff-only
```
Expected: at HEAD of `origin/main`, with Phase 0 commits present. Verify with:
```bash
git log -n 5 --oneline
```

- [ ] **Step 2: Create feature branch**

Run:
```bash
git checkout -b feature/launch-phase-1-safety-audit
```

- [ ] **Step 3: Verify clean baseline**

Run:
```bash
uv sync
uv run pytest -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py
```
Expected: all tests pass.

---

## Task 2: Add the `READ_ONLY` setting

**Files:**
- Modify: `src/config.py`
- Modify: `.env.example`

- [ ] **Step 1: Add `READ_ONLY` to the `Settings` class**

Open `src/config.py`. Find the existing capability-registry-related settings:

```python
    # Capability registry
    # ENABLED_CAPABILITIES: comma-separated capability IDs to enable.
    # ...
    ENABLED_CAPABILITIES: str = ""
    # DISABLED_CAPABILITIES: ...
    DISABLED_CAPABILITIES: str = ""
```

Immediately after `DISABLED_CAPABILITIES`, add:

```python
    # READ_ONLY: when True, force-disables all write-capable capabilities
    # (manage_announcements, open_ticket, report_login_problem, report_security)
    # by adding them to the disabled set at registry build time. Intended for
    # staging environments, the smoke test window, and any deploy where
    # inadvertent writes would be unacceptable. Overrides nothing else.
    READ_ONLY: bool = False
```

- [ ] **Step 2: Update `.env.example`**

Find the capability-registry block in `.env.example` (search for `ENABLED_CAPABILITIES`). Add after the `DISABLED_CAPABILITIES` line:

```bash
# READ_ONLY=true disables all write-capable capabilities at startup
# (manage_announcements, open_ticket, report_login_problem, report_security).
# Recommended for staging and for any deploy where accidental writes would
# be unacceptable. Default false. See docs/security/write-capability-audit.md.
# READ_ONLY=false
```

- [ ] **Step 3: Commit**

Run:
```bash
git add src/config.py .env.example
git commit -m "feat(config): add READ_ONLY setting for capability safety guard"
```

---

## Task 3: Write failing tests for the `READ_ONLY` guard

**Files:**
- Create: `tests/test_capabilities_read_only.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for the READ_ONLY capability guard (launch Phase 1).

The guard adds every write-capable capability ID into the disabled set
at registry build time when settings.READ_ONLY is True. Verifies that:

1. Without READ_ONLY, write capabilities remain enabled (baseline).
2. With READ_ONLY=true, write capabilities are disabled in the registry.
3. With READ_ONLY=true, read-only capabilities are NOT affected.
4. The WRITE_CAPABILITY_IDS constant lists exactly the known write caps.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture
def fresh_registry(monkeypatch):
    """Rebuild the capability registry from scratch under the current settings.

    The registry is a module-level singleton cached in _registry. We reset
    it before each test so settings changes are picked up.
    """
    from src.agent.domains import capabilities as caps_mod

    # Reset the module-level singleton before each test.
    monkeypatch.setattr(caps_mod, "_registry", None, raising=False)

    def _build():
        # Reset again to be extra-safe against test interleaving.
        monkeypatch.setattr(caps_mod, "_registry", None, raising=False)
        return caps_mod.get_capability_registry()

    return _build


def test_write_capability_ids_constant_matches_known_writes():
    """The WRITE_CAPABILITY_IDS set must list exactly the known write capabilities."""
    from src.agent.domains.capabilities import WRITE_CAPABILITY_IDS

    assert WRITE_CAPABILITY_IDS == frozenset({
        "manage_announcements",
        "open_ticket",
        "report_login_problem",
        "report_security",
    })


def test_baseline_write_caps_enabled_without_read_only(monkeypatch, fresh_registry):
    """Sanity check: with READ_ONLY=false and no ENABLED/DISABLED env, writes are on."""
    monkeypatch.setattr("src.config.settings.READ_ONLY", False, raising=False)
    monkeypatch.setattr("src.config.settings.ENABLED_CAPABILITIES", "", raising=False)
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    registry = fresh_registry()

    for cap_id in ("manage_announcements", "open_ticket",
                   "report_login_problem", "report_security"):
        assert registry.get_by_id(cap_id) is not None, f"{cap_id} should be enabled"


def test_read_only_disables_all_write_capabilities(monkeypatch, fresh_registry):
    """With READ_ONLY=true, every write capability is absent from the registry."""
    monkeypatch.setattr("src.config.settings.READ_ONLY", True, raising=False)
    monkeypatch.setattr("src.config.settings.ENABLED_CAPABILITIES", "", raising=False)
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    registry = fresh_registry()

    for cap_id in ("manage_announcements", "open_ticket",
                   "report_login_problem", "report_security"):
        assert registry.get_by_id(cap_id) is None, (
            f"{cap_id} must be disabled when READ_ONLY=true"
        )


def test_read_only_does_not_affect_read_capabilities(monkeypatch, fresh_registry):
    """Read-only capabilities stay enabled when READ_ONLY=true."""
    monkeypatch.setattr("src.config.settings.READ_ONLY", True, raising=False)
    monkeypatch.setattr("src.config.settings.ENABLED_CAPABILITIES", "", raising=False)
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    registry = fresh_registry()

    # A sampling of read-only capabilities that should NOT be touched.
    for cap_id in ("ask_question", "check_allocations", "search_software",
                   "check_system_status", "browse_events"):
        assert registry.get_by_id(cap_id) is not None, (
            f"{cap_id} is read-only and should stay enabled under READ_ONLY=true"
        )


def test_read_only_overrides_explicit_enabled_list(monkeypatch, fresh_registry):
    """If READ_ONLY=true, even ENABLED_CAPABILITIES naming a write cap can't resurrect it."""
    monkeypatch.setattr("src.config.settings.READ_ONLY", True, raising=False)
    monkeypatch.setattr(
        "src.config.settings.ENABLED_CAPABILITIES",
        "ask_question,manage_announcements,open_ticket",
        raising=False,
    )
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    registry = fresh_registry()

    assert registry.get_by_id("ask_question") is not None  # allowed read
    assert registry.get_by_id("manage_announcements") is None  # blocked by READ_ONLY
    assert registry.get_by_id("open_ticket") is None  # blocked by READ_ONLY


def test_read_only_logged_at_startup(monkeypatch, fresh_registry, caplog):
    """READ_ONLY=true must produce a prominent log line at registry build time."""
    import logging

    monkeypatch.setattr("src.config.settings.READ_ONLY", True, raising=False)
    monkeypatch.setattr("src.config.settings.ENABLED_CAPABILITIES", "", raising=False)
    monkeypatch.setattr("src.config.settings.DISABLED_CAPABILITIES", "", raising=False)

    with caplog.at_level(logging.WARNING):
        fresh_registry()

    assert any(
        "READ_ONLY" in rec.message and "active" in rec.message.lower()
        for rec in caplog.records
    ), "Expected a WARNING-level log line announcing READ_ONLY=true at startup"
```

- [ ] **Step 2: Run to confirm failures**

Run:
```bash
uv run pytest tests/test_capabilities_read_only.py -v
```
Expected: `ImportError: cannot import name 'WRITE_CAPABILITY_IDS' from 'src.agent.domains.capabilities'` on the first test, then subsequent tests fail for similar reasons.

---

## Task 4: Implement `WRITE_CAPABILITY_IDS` and the `READ_ONLY` guard

**Files:**
- Modify: `src/agent/domains/capabilities.py`

- [ ] **Step 1: Add the `WRITE_CAPABILITY_IDS` constant**

Open `src/agent/domains/capabilities.py`. Find the `_ATTRIBUTION_FALLBACK_ORDER` constant near the top of the file. Immediately after it, add:

```python
# ── Write-capable capabilities ────────────────────────────────────────────

# Capabilities whose backend performs writes (POST/PUT/DELETE) against an
# external system. Source of truth for the READ_ONLY guard; enumerated in
# docs/security/write-capability-audit.md. If a new write-capable capability
# is added, it MUST be added here AND in the audit document.
WRITE_CAPABILITY_IDS: frozenset[str] = frozenset({
    "manage_announcements",   # announcements domain: create/update/delete
    "open_ticket",            # jsm domain: create support ticket
    "report_login_problem",   # jsm domain: create login-issue ticket
    "report_security",        # jsm domain: create security-concern ticket
})
```

- [ ] **Step 2: Apply `READ_ONLY` in `_build_registry()`**

Find `_build_registry()` (around line 559). Locate the block that parses `DISABLED_CAPABILITIES`:

```python
    disabled_set: set[str] = set()
    if settings.DISABLED_CAPABILITIES:
        disabled_set = {s.strip() for s in settings.DISABLED_CAPABILITIES.split(",") if s.strip()}
```

Immediately after that block, insert:

```python
    # Phase 1 safety guard: READ_ONLY forcibly adds every write capability to
    # the disabled set. This runs BEFORE the filter loop, so the deny-wins
    # semantics of the existing filter automatically picks it up.
    if settings.READ_ONLY:
        disabled_set = disabled_set | set(WRITE_CAPABILITY_IDS)
        logger.warning(
            "READ_ONLY=true active — write capabilities disabled: %s",
            ", ".join(sorted(WRITE_CAPABILITY_IDS)),
        )
```

- [ ] **Step 3: Run the tests**

Run:
```bash
uv run pytest tests/test_capabilities_read_only.py -v
```
Expected: all six tests PASS.

- [ ] **Step 4: Run the full test suite to make sure nothing else regressed**

Run:
```bash
uv run pytest -q --ignore=tests/test_e2e.py --ignore=tests/test_classify.py
```
Expected: all tests pass. No regressions from the `READ_ONLY` addition.

- [ ] **Step 5: Commit**

Run:
```bash
git add src/agent/domains/capabilities.py tests/test_capabilities_read_only.py
git commit -m "$(cat <<'EOF'
feat(capabilities): add READ_ONLY guard for write capabilities

When READ_ONLY=true, the capability registry force-disables the four
known write capabilities (manage_announcements, open_ticket,
report_login_problem, report_security) at build time. Logs a prominent
WARNING on startup when active.

Intended for staging (default) and any deploy where accidental writes
would be unacceptable. Works on top of existing ENABLED/DISABLED env
vars; READ_ONLY always wins.

Spec: docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md
EOF
)"
```

---

## Task 5: Write the safety-audit document

**Files:**
- Create: `docs/security/write-capability-audit.md`

- [ ] **Step 1: Create the directory if it doesn't exist**

Run:
```bash
mkdir -p docs/security
```

- [ ] **Step 2: Write the audit document**

Create `docs/security/write-capability-audit.md` with the following content:

```markdown
# Write-Capable Capability Audit

**Date:** 2026-04-21
**Status:** Living document. Update whenever a write-capable capability is added, removed, or renamed.
**Spec:** `docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md` (Phase 1).

## Purpose

Enumerate every capability in the access-agent that performs writes against an external system, and document the layered controls that prevent accidental or malicious invocation. This document is the pre-launch artifact that supports the launch criterion "privacy and security of the new architecture."

## Scope

"Write" here means any capability whose backend performs a POST, PUT, or DELETE against an external service. Read-only queries against MCP servers (search, lookup, status, list) are not in scope.

## Enumeration

| Capability ID | Domain | Backend (MCP server) | Effect | Classifier gate |
| ------------- | ------ | -------------------- | ------ | --------------- |
| `manage_announcements` | `announcements` | `announcements` | Create / update / delete ACCESS-CI announcements | Classifier must set `domain == "announcements"` AND `capability_id == "manage_announcements"`. Classifier prompt requires explicit imperative language: "create an announcement", "update this announcement", "delete the announcement" (see `src/agent/nodes/classify.py`). |
| `open_ticket` | `jsm` | `jsm` | Create a new JSM support ticket | Classifier must set `domain == "jsm"` AND `capability_id == "open_ticket"`. Prompt requires explicit ticket-creation verbs ("open a ticket", "file a ticket", "submit a ticket"). Described problems without those verbs do NOT trigger this capability. |
| `report_login_problem` | `jsm` | `jsm` | Create a login-issue JSM ticket | Same classifier rule as `open_ticket`, scoped to login problems. |
| `report_security` | `jsm` | `jsm` | Create a security-concern JSM ticket | Same classifier rule, scoped to security issues. |

The source of truth for this list is the `WRITE_CAPABILITY_IDS` constant in `src/agent/domains/capabilities.py`. Any change to that set must be reflected in this table (enforced by code review).

## Defense layers

Three layers protect against unintended writes. They are additive; an attacker or a regression would need to defeat all three simultaneously.

### Layer 1: Classifier prompt rules

The classifier LLM's prompt (`src/agent/nodes/classify.py`) explicitly states that write capabilities are only triggered by explicit imperative language. Example rules from the prompt:

> Only set `domain` to `"jsm"` when the user uses explicit action language requesting ticket creation.

> A user describing a problem is NOT the same as requesting a ticket. Only set `domain` to `"jsm"` when the user uses explicit action language requesting ticket creation.

This is an LLM-level rule, not a hard guarantee. Prompt regressions or model changes could theoretically shift behavior. Layers 2 and 3 backstop it.

### Layer 2: Operator-controlled deny-list (`DISABLED_CAPABILITIES`)

Operators can explicitly disable any capability via the `DISABLED_CAPABILITIES` environment variable. Deny wins over `ENABLED_CAPABILITIES`. Useful for targeted disabling without broad policy changes.

### Layer 3: `READ_ONLY` global guard

Setting `READ_ONLY=true` in the environment forcibly adds every `WRITE_CAPABILITY_IDS` member to the disabled set at registry build time. This happens before any filter is applied; no caller can re-enable a write capability when `READ_ONLY=true` is set — even `ENABLED_CAPABILITIES` naming a write capability is overridden.

When active, the process logs a prominent WARNING at startup:

```
READ_ONLY=true active — write capabilities disabled: manage_announcements, open_ticket, report_login_problem, report_security
```

### Staging default

The staging environment runs with `READ_ONLY=true` by default. This guarantees the smoke test (and any manual testing against staging) cannot create tickets or announcements regardless of classifier behavior or developer experimentation.

### Production default

Production runs with `READ_ONLY=false` so real users can file tickets and manage announcements when the flow warrants it. The classifier prompt (Layer 1) is what actually gates individual requests in production.

## Smoke-test safety statement

The production smoke test, when the process is started with `READ_ONLY=true`, cannot create tickets or announcements regardless of classifier behavior. In practice the production smoke test runs against the normal production process (`READ_ONLY=false`) with a generic query ("smoke test") that the classifier does not route to any write capability. The per-capability smoke test set (Phase 2) explicitly avoids the four write capabilities above.

## What this document does NOT cover

- Logging retention and audit trails (covered in the privacy data-flow document, Phase 6).
- Authentication and authorization for specific users invoking write capabilities (covered by the existing JWT and `acting_user` infrastructure).
- MCP-server-level deny for write operations as an additional defense layer (flagged as post-launch hardening in the launch spec).
- Testing of write capabilities in staging (intentional opt-in; documented separately in a staging README when staging is stood up in Phase 2).

## Change log

- 2026-04-21: Initial audit. Four write capabilities enumerated; `WRITE_CAPABILITY_IDS` constant and `READ_ONLY` guard landed together.
```

- [ ] **Step 3: Commit**

Run:
```bash
git add docs/security/write-capability-audit.md
git commit -m "docs(security): add write-capability audit for launch"
```

---

## Task 6: Verify end-to-end

**Files:** none (verification only).

- [ ] **Step 1: Confirm `READ_ONLY=true` actually disables writes via `/capabilities` endpoint**

Locally, start the agent with `READ_ONLY=true`:

```bash
READ_ONLY=true uv run uvicorn src.main:app --reload --port 8001 &
SERVER_PID=$!
sleep 3
```

Check the capabilities endpoint (adjust path if needed — the exact route lives in `src/api/routes.py`):

```bash
curl -s http://localhost:8001/api/v1/capabilities | python -m json.tool | grep -E '"id"' | sort
```

Expected: the output does NOT include `manage_announcements`, `open_ticket`, `report_login_problem`, or `report_security`. It DOES include `ask_question`, `check_allocations`, `search_software`, `check_system_status`, etc.

Clean up:
```bash
kill $SERVER_PID
```

- [ ] **Step 2: Confirm without `READ_ONLY`, writes are present (regression check)**

```bash
uv run uvicorn src.main:app --reload --port 8001 &
SERVER_PID=$!
sleep 3
curl -s http://localhost:8001/api/v1/capabilities | python -m json.tool | grep -E '"id"' | sort
kill $SERVER_PID
```

Expected: output includes the four write capabilities (baseline behavior unchanged when `READ_ONLY` is unset or false).

- [ ] **Step 3: Commit empty verification note**

```bash
git commit --allow-empty -m "chore: verify READ_ONLY=true hides write capabilities from /capabilities"
```

---

## Task 7: Open the pull request

**Files:** none.

- [ ] **Step 1: Push the branch**

```bash
git push -u origin feature/launch-phase-1-safety-audit
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --title "feat(safety): Phase 1 — write-capability audit + READ_ONLY guard" --body "$(cat <<'EOF'
## Summary
- Enumerated the four write-capable capabilities (`manage_announcements`, `open_ticket`, `report_login_problem`, `report_security`) in a new `WRITE_CAPABILITY_IDS` constant.
- Added `READ_ONLY=true` env var that force-disables all write capabilities at capability-registry build time; logs a WARNING at startup when active.
- Committed `docs/security/write-capability-audit.md` documenting the layered defense (classifier rules → DISABLED_CAPABILITIES → READ_ONLY) and staging/production defaults.

Spec: `docs/superpowers/specs/2026-04-21-production-launch-hardening-design.md`.
Umbrella plan: `docs/superpowers/plans/2026-04-21-production-launch-umbrella.md`.

Part of the launch-hardening track; unblocks Phase 2 (staging, which will default to `READ_ONLY=true`).

## Test plan
- [ ] CI green.
- [ ] Unit tests: `tests/test_capabilities_read_only.py` (6 cases covering constant, baseline, disabling, read-only unaffected, ENABLED override, startup log).
- [ ] Manual: `READ_ONLY=true uv run uvicorn src.main:app` — `/capabilities` endpoint does not list write capabilities.
- [ ] Manual: without `READ_ONLY`, `/capabilities` returns the full list including writes.
EOF
)"
```
Expected: PR URL printed. Hand off for review.
