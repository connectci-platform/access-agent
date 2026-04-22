# Branch Protection — `main`

**Status:** **Pending activation.** Andrew will configure the protection rules in GitHub UI when he has time; Joe will then capture the live settings as `branch-protection-snapshot.json` in this directory and update this status to "Active as of YYYY-MM-DD".

**Spec:** `docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md` §7.

## Intended ruleset

Once Andrew applies it in GitHub UI (repo Settings → Branches → Add rule for branch `main`):

- **Require a pull request** before merging to `main`, with at least **1 approval**.
- **Dismiss stale approvals** when new commits are pushed to the PR.
- **Require status checks** to pass before merging:
  - `lint-and-test` (from `.github/workflows/ci.yml`)
  - `gitleaks` (from `.github/workflows/ci.yml`)
- **Require branches to be up to date** before merging.
- **Require conversation resolution** before merging.
- **Block force pushes.**
- **Block deletions.**
- **Admins may bypass** in genuine emergencies (every bypass is visible in the GitHub audit log).

## Drift check

A weekly GitHub Action (`.github/workflows/branch-protection-drift-check.yml`) fetches the live protection settings via `gh api` and diffs them against the committed snapshot. Drift fails the job and opens a GitHub issue.

**Until the snapshot is captured**, the drift-check workflow runs but exits early with a log message — so it doesn't spuriously fail during the pre-activation window.

## Updating the snapshot

If Andrew changes the protection rules intentionally in the future (or when first activating):

```bash
gh api repos/necyberteam/access-agent/branches/main/protection > docs/security/branch-protection-snapshot.json
```

Commit the updated snapshot as a reviewable change. The drift check will then accept the new baseline.
