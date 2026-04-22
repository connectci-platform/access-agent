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
