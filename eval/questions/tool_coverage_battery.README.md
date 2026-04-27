# `tool_coverage_battery.yaml`

Small eval battery (14 questions) that exercises each MCP server at
least once with a question that needs live data, plus a few stable
how-to questions that should be answered from docs without any tool
call.

Built to answer: **does the new tool-calling-loop agent call tools when
the question requires live data, and skip them when it doesn't?** The
existing `phase3_smoke_battery.json` wasn't designed for that question.

> **Format note:** This battery uses YAML for authoring ergonomics
> (multi-line strings, no escape hell). The loader at
> `src/eval/questions.py` dispatches on file extension and supports
> both `.yaml` and `.json`. Sibling batteries remain JSON.

## Schema

```yaml
- id: tc-<area>-NN
  question: user-facing query text
  capability_area: ...               # existing convention, used by runner
  battery: tool_coverage
  expected_tool: <tool_name>         # or null = RAG-sanity, no tool expected
  ground_truth_stability: stable     # or time_bound — see below
  required_facts:
    - plain-English claim the answer must make (atomic, durable)
  sources:
    - "tool: <name>(<args>) at <host>"
    - "doc: <url>"
  authoring_notes: |
    Snapshot data, regen logic, reviewer hints. NOT graded by the judge.
```

`required_facts` are **semantic**, not verbatim. When the scorer is
built, the judge prompt will be something like: "for each fact, rate
whether the candidate answer supports it (yes / partial / no),
regardless of exact wording." Don't write facts as regex strings —
write them as claims a human reviewer would check.

### `ground_truth_stability`

Per the rubric spec
(`docs/superpowers/specs/2026-04-21-eval-rubric-ground-truth-design.md`):

- **`stable`** — facts rarely change (eligibility rules, workflows,
  policy). Author once, refresh on documentation revision only.
- **`time_bound`** — facts depend on live system state (current
  outages, today's events, current allocation count). Refresh script
  regenerates the relevant `required_facts` from live tools before
  each eval run; the snapshot in `authoring_notes` shows what state
  was true at authoring.

### `required_facts` vs `authoring_notes`

- **`required_facts`** are what the judge grades against. Each is an
  atomic, independently-checkable claim. No clause-stacking, no
  transitive overlap between siblings, no snapshot data.
- **`authoring_notes`** are for human reviewers — worked examples,
  snapshot data ("at authoring time the tool returned X"), regen
  logic, edge-case framing. The judge does not see this field.

## Authoring

Every `TODO` in the file needs a human to replace with a real claim:

- **Tool-targeted questions:** run the `expected_tool` against the live
  MCP server with reasonable arguments, read the output, and describe
  what a correct answer should assert. Keep facts specific (name real
  resources, real project titles, real group IDs) so the judge can
  catch answers that are fluent but fabricated.
- **RAG-sanity questions:** confirm facts against canonical ACCESS
  documentation (operations.access-ci.org, support.access-ci.org,
  allocations.access-ci.org, resource user guides).
- **Edge cases** where docs don't settle it: ask someone on the ACCESS
  team.

## Not included (on purpose)

- **Write operations** (`create_announcement`, `delete_announcement`,
  JSM ticket creation). Side-effecting tools belong in an integration
  test, not an answer-quality battery.
- **Authenticated-user queries** (`get_user_data`, `get_my_events`,
  etc.). Auth state is a confounder that isn't about the tool-calling
  decision.
- **Compute-resources MCP.** No resource-lookup tool confirmed in the
  catalog we pulled — placeholder for now; add a question if one
  exists.

## Scorer status

Factoid-based scoring isn't implemented yet. Running this battery
through the current scorer uses the existing rubric-based composite —
which is fine for seeing that the runs complete and the tool-call
counts land. Real factoid grading is Track C work, separate from this
battery.
