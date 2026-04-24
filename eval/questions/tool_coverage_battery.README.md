# `tool_coverage_battery.json`

Small eval battery (14 questions) that exercises each MCP server at
least once with a question that needs live data, plus a few stable
how-to questions that should be answered from docs without any tool
call.

Built to answer: **does the new tool-calling-loop agent call tools when
the question requires live data, and skip them when it doesn't?** The
existing `phase3_smoke_battery.json` wasn't designed for that question.

## Schema

```jsonc
{
  "id": "tc-<area>-NN",
  "question": "user-facing query text",
  "capability_area": "...",          // existing convention, used by runner
  "battery": "tool_coverage",
  "expected_tool": "<tool_name>" | null,   // null = RAG-sanity, no tool expected
  "required_facts": [
    "plain-English claim the answer must make"
  ]
}
```

`required_facts` are **semantic**, not verbatim. When the scorer is
built, the judge prompt will be something like: "for each fact, rate
whether the candidate answer supports it (yes / partial / no),
regardless of exact wording." Don't write facts as regex strings —
write them as claims a human reviewer would check.

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
