# Branch Protection — `main`

**Status:** **Configured but not enforced.** As of 2026-04-22, the intended ruleset below has been saved in GitHub's UI, but GitHub does not enforce branch-protection rulesets on private repositories owned by free-tier organizations. Saving the ruleset returned the advisory: *"Your rulesets won't be enforced on this private repository until you upgrade this organization account to GitHub Team."*

Enforcement activates automatically — without reconfiguration — if any of the following happens:

- `necyberteam` upgrades from GitHub Free to GitHub Team (~$4/user/month).
- This repository is made public.
- The repository moves to an org on a paid plan.

Until then, the CI-layer guardrails (100% coverage diff-gate, gitleaks, pip-audit, `uv sync --locked`) still run on every PR and still fail the job when violated — they just cannot block merge without branch-protection-level enforcement. Team norms (review before merge, no force-push to `main`) carry the rest. The drift-check workflow no-ops cleanly while no snapshot exists.

**Revisit trigger:** consider the Team upgrade when the active-contributor count grows past ~5, or when new contributors join whose merge judgment isn't yet calibrated to the project's standards.

**Spec:** `docs/superpowers/specs/2026-04-22-code-quality-guardrails-design.md` §7.

## Intended ruleset (configured, not enforced)

Saved in the GitHub UI on 2026-04-22:

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

The snapshot cannot be captured until the ruleset becomes enforced (`gh api ... /branches/main/protection` returns 403 on unenforced configurations). Once the enforcement blocker above is resolved:

```bash
gh api repos/necyberteam/access-agent/branches/main/protection > docs/security/branch-protection-snapshot.json
```

Commit the snapshot as a reviewable change. The drift check will accept the new baseline and start diffing against it weekly.

For intentional future ruleset changes after that point, repeat the snapshot commit in the same PR as the UI change.
