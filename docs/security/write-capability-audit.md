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

### Events writes (no `WRITE_CAPABILITY_IDS` entry — guarded by tool name only)

The `events` MCP server exposes write tools, but they are **not** owned by any registry write capability: the only events capability in `GENERAL_CAPABILITIES` is `browse_events`, which is read-only. Events writes reach the agent solely through the `tool_calling_loop`, which builds tools directly from the MCP catalog. They are therefore guarded by their MCP tool name in `WRITE_MCP_TOOL_NAMES` (Layer 3, loop path), with no corresponding `WRITE_CAPABILITY_IDS` id. The legacy chain never surfaces these tools, so a capability-id entry there would gate nothing; the tool-name deny-list is the load-bearing control.

| MCP tool name | Server | Effect | Guard |
| ------------- | ------ | ------ | ----- |
| `register_for_event` | `events` | Register the acting user for an event | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `cancel_registration` | `events` | Cancel the acting user's event registration | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `create_event` | `events` | Create an event (organizer) | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `update_event` | `events` | Update an event (organizer) | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `delete_event` | `events` | Delete an event (organizer; preview-by-default, writes when `confirmed:true`) | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `restore_event` | `events` | Restore a deleted event (organizer) | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `send_for_review` | `events` | Submit an event for review (organizer) | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `cancel_occurrence` | `events` | Cancel one occurrence (organizer; preview-by-default, writes when `confirmed:true`) | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `restore_occurrence` | `events` | Restore a cancelled occurrence (organizer) | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `edit_occurrence` | `events` | Edit one occurrence (organizer). A location-only edit, or a date edit on a dark/draft occurrence, applies **immediately** with no preview and no `confirmed` flag; only a date change on a live occurrence with existing registrants previews and requires `confirmed:true`. The call signature does not reveal which path an invocation takes, so the whole tool is stripped. | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |
| `add_occurrence` | `events` | Add an occurrence to an event (organizer) | `WRITE_MCP_TOOL_NAMES` (loop deny-list) |

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

The legacy `plan→execute→evaluate→recover→synthesize` chain enforces this guard at the capability-registry level. The new `tool_calling_loop` (Phase 3) builds tools directly from the MCP catalog and never sees the registry, so it applies the same guard via a parallel deny-list of MCP tool names (`WRITE_MCP_TOOL_NAMES` in `src/agent/domains/capabilities.py`). Both code paths are covered by tests in `tests/test_capabilities_read_only.py` and `tests/test_tool_calling_loop.py::test_read_only_strips_write_tools_from_loop_registry` — the audit's "READ_ONLY blocks all writes" claim is machine-verified on both paths.

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
- 2026-04-29: Extended `READ_ONLY` guard to the `tool_calling_loop` code path (Phase 3). Added `WRITE_MCP_TOOL_NAMES` deny-list in `capabilities.py` and a filter in `tool_calling_loop_node` so the loop's tool registry honors `READ_ONLY=true` the same way the legacy chain does. Without this, the loop bypassed the guard entirely. Machine-verified by `tests/test_tool_calling_loop.py::test_read_only_strips_write_tools_from_loop_registry`.
- 2026-08-15: Closed a guard hole for events writes. When the events-CRUD organizer tools shipped in access_mcp they were not added to `WRITE_MCP_TOOL_NAMES`, so under `READ_ONLY=true` the `tool_calling_loop` left all nine callable and write-counting undercounted them. Added `create_event`, `update_event`, `delete_event`, `restore_event`, `send_for_review`, `cancel_occurrence`, `restore_occurrence`, `edit_occurrence`, and `add_occurrence` to `WRITE_MCP_TOOL_NAMES`. Documented the events writes (organizer + the pre-existing registration writes) in the new "Events writes" section above; these are guarded by tool name only and have no `WRITE_CAPABILITY_IDS` entry because no registry write capability owns them. `WRITE_CAPABILITY_IDS` is unchanged. Machine-verified by `tests/test_tool_calling_loop.py::test_read_only_strips_events_organizer_write_tools_from_loop` and `tests/test_capabilities_read_only.py::test_write_mcp_tool_names_covers_all_known_write_tools`.
