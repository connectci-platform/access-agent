# Agent Answer Evaluation Pipeline

Scores agent answers against a 5-dimension rubric using an LLM judge. Used for:
- **Pre-production eval**: Run a question set against a candidate agent before deploying
- **Comparing configurations**: Side-by-side comparison of two eval runs

## Quick Start

```bash
# Run eval with the friendly battery (50 clean questions)
uv run python -m src.eval run --questions eval/questions/friendly_battery.json

# Run eval with real user queries (50 messy questions)
uv run python -m src.eval run --questions eval/questions/real_user_battery.json

# Compare two runs
uv run python -m src.eval compare --run-a <run-id> --run-b <run-id>
```

### Phase-3 parity check (tool-calling loop vs legacy chain)

For regression-checking the Phase-3 loop against the legacy chain, use the curated smoke battery:

```bash
# Baseline — legacy plan→execute chain
uv run python -m src.eval run \
  --system agent_full_legacy \
  --questions eval/questions/phase3_smoke_battery.json

# Candidate — tool-calling loop (the new default)
uv run python -m src.eval run \
  --system agent_full \
  --questions eval/questions/phase3_smoke_battery.json

# Compare + render
uv run python -m src.eval compare-judge \
  --baseline <baseline-run-id> --candidate <candidate-run-id> \
  -o comparisons/phase3-smoke.json
uv run python -m src.eval html --from-json comparisons/phase3-smoke.json -o /tmp/phase3-smoke.html
```

Capture the run IDs that the `run` commands print — you need them for compare-judge.

## Systems (`--system` choices)

The `run` command's `--system` flag selects which pipeline scores the questions. All four systems write to the same `eval_runs` / `eval_scores` tables; system is captured in metadata so comparison reports can distinguish them.

| System | What it runs | When to use |
|---|---|---|
| `agent_full` *(default)* | Full agent graph with the tool-calling loop — the Phase-3 default path. | Standard evaluation; grand-prix; Phase-3 candidate in parity comparisons. |
| `agent_full_legacy` | Full agent graph with the legacy `plan → execute → evaluate → recover → synthesize` chain. Forces `USE_TOOL_CALLING_LOOP=false` for the run. | Phase-3 parity baseline; debugging regressions introduced by the loop. |
| `agent_rag_only` | Skips the agent entirely; serves the top RAG match as the answer. | Measures the RAG-only baseline. |
| `raw_rag` | Queries UKY's `/ask` endpoint directly, no agent involvement. | Production-baseline comparisons (grand-prix). |

**Note on grand-prix historical compatibility.** Prior to 2026-04-23, `agent_full` meant the legacy chain. After, it means the tool-calling loop. Grand-prix runs recorded before vs after this date are NOT apples-to-apples in the `agent_full` column — if you need to compare against pre-2026-04-23 grand-prix runs, use `--system agent_full_legacy` for re-runs.

## Run IDs

Runs get semantic IDs of the form `{shortcode}-{YYYYMMDD}-{HHMMSS}-{hash6}`, where `shortcode` reflects the pipeline architecture:

| System | Shortcode | Example |
|---|---|---|
| `agent_full` | `loop` | `loop-20260423-143052-a1b2c3` |
| `agent_full_legacy` | `chain` | `chain-20260423-143107-8d7e4f` |
| `agent_rag_only` | `rag_only` | `rag_only-20260423-143122-3f91a8` |
| `raw_rag` | `raw_rag` | `raw_rag-20260423-143135-77c4de` |

The IDs are lex-sortable by timestamp, fit in the existing `String(36)` column (no migration), and make Argilla's "Eval Run ID" metadata filter self-describing. The 6-hex random suffix prevents collisions between same-second runs.

## Requirements

- `OPENAI_API_KEY` set in `.env` (for the agent and the judge LLM)
- MCP servers accessible (for tool-calling questions)
- PostgreSQL running (for storing results)

## Running Against Production

For the two-way baseline comparison described in
[Decision 007](https://github.com/necyberteam/access-qa-planning/blob/main/decisions/007-production-baseline-comparison.md),
the eval runs against **production infrastructure** (prod Postgres, prod MCP
servers, prod Argilla) from a workstation. The setup is deliberately separated
from `.env` so prod eval runs cannot accidentally use local values.

### One-time setup

Copy the template and fill in real production values:

```bash
cp .env.eval.prod.example .env.eval.prod
$EDITOR .env.eval.prod
```

`.env.eval.prod` is gitignored. Ask the team for the real values — do not
commit them.

### Running an eval

Prod Postgres lives in a Docker container on `mcp.access-ci.org` and is not
directly reachable from a workstation, so open an SSH tunnel first:

```bash
# Terminal 1: open the tunnel, leave it running
./scripts/eval-tunnel-open
```

The tunnel forwards local `5432` → `mcp.access-ci.org:5432`. If you already
run a local Postgres on 5432, set a different local port:

```bash
EVAL_PROD_DB_LOCAL_PORT=5433 ./scripts/eval-tunnel-open
```

and update `DATABASE_URL` in `.env.eval.prod` to match (`...@localhost:5433/...`).

The tunnel assumes the prod container publishes Postgres on the SSH host's
loopback (i.e. `localhost:5432` on `mcp.access-ci.org`). If that ever
changes — for example, if Postgres moves onto a Docker-internal network
only — the tunnel target in `scripts/eval-tunnel-open` will need updating.

Then from another terminal:

```bash
# Full-capabilities run (what we want to ship)
./scripts/eval-prod run \
  --questions eval/questions/friendly_battery.json \
  --push-argilla

# RAG-only baseline run (same agent, MCP capabilities disabled)
./scripts/eval-prod --rag-only run \
  --questions eval/questions/friendly_battery.json \
  --push-argilla

# Compare the two runs
./scripts/eval-prod compare --run-a <rag_only_id> --run-b <full_caps_id>
```

The `--rag-only` flag (which must appear **before** the eval subcommand)
sets `ENABLED_CAPABILITIES=ask_question,ask_xdmod_question,ask_about_resource`
so the agent runs with RAG backends only and no MCP tools. Both runs
otherwise use the same agent binary, classifier, and synthesis layer — the
only variable being tested is the value added by MCP tools.

Results land in the prod Argilla dataset `eval-baseline-comparison`, which
is kept separate from `eval-production` (the rolling production scoring
dataset) so the one-time baseline evidence doesn't mix with ongoing
quality tracking.

### Safety checks

The `eval-prod` wrapper refuses to run if:

- `.env.eval.prod` does not exist
- `ENVIRONMENT` is not `production` after loading the env file
- `DATABASE_URL` is unset or still contains the `USER:PASS` placeholder
- `ARGILLA_EVAL_DATASET` is unset

These catch the common mistake of running the wrapper before filling in
the env file.

## Question Sets

Eval questions live in `eval/questions/` as JSON files:

| File | Questions | Description |
|------|-----------|-------------|
| `friendly_battery.json` | 50 | Clean, well-phrased questions covering all capability areas |
| `real_user_battery.json` | 50 | Real user queries with typos, vague phrasing, pasted errors |
| `mcp_coverage_battery.json` | 21 | Targeted questions for MCP capabilities under-represented in the other batteries — system status, affinity groups, events, XDMoD, NSF awards, software discovery |
| `combined_battery.json` | 30 | Allocations MCP, compute-resources MCP, cross-resource software comparison, combined RAG+tools questions, edge cases, and messy multi-part questions |

**Total: 151 questions** across four batteries.

### Format

```json
[
  {
    "id": "friendly-001",
    "question": "How do I log in to Expanse using SSH?",
    "capability_area": "expanse",
    "battery": "friendly"
  }
]
```

Optional fields (used in `combined_battery.json`):

```json
{
  "id": "comb-022",
  "question": "my job keeps failing on anvil, are there any outages?",
  "capability_area": "messy_combined",
  "battery": "combined",
  "expected_type": "combined",
  "expected_tools": ["get_infrastructure_news", "get_resource_hardware"]
}
```

- `expected_type`: the expected classifier output (`static`, `dynamic`, `combined`). Used for post-hoc analysis of classifier accuracy, not at runtime.
- `expected_tools`: the MCP tools the question should trigger. Used for post-hoc analysis of planner accuracy.

### Adding Questions

Add questions to an existing JSON file or create a new one. Each question needs an `id`, `question`, and `capability_area`. Run against a subset with `--questions your_file.json`.

## Scoring Rubric

Five dimensions, each scored 1-5:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Correctness | 30% | Does the answer match its sources? Tool results take precedence over RAG when they conflict. |
| Completeness | 25% | Does it address all parts of the question? |
| Relevance | 20% | Does it stay on topic? |
| Citation quality | 15% | Are URLs present and from sources? |
| Appropriate hedging | 10% | Does it calibrate confidence to source quality? |

The judge evaluates whether the agent faithfully represented the information it had access to, not whether the source information itself is correct.

## Judge Configuration

The judge LLM is configurable via environment variables:

```bash
# Use default OpenAI (for pre-production eval with curated questions)
EVAL_JUDGE_MODEL=gpt-4o-mini

# Use on-premise LLM (required for production scoring with real user data)
EVAL_JUDGE_BASE_URL=http://uky-gpu:8000/v1
EVAL_JUDGE_MODEL=llama-3.1-70b-instruct
EVAL_JUDGE_API_KEY=not-needed
```

## Output

Each run prints a summary:

```
============================================================
  Eval Run: 89d5b65f-8131-4a99-bb9b-4d843edfe905
  Branch:   feature/eval-pipeline
  Commit:   abb2aa98
============================================================

  Questions: 50
  Scored:    48
  Skipped:   2

  Composite Score: 4.12 / 5.00

  Per Dimension:
    correctness          4.35  ████░
    completeness         3.90  ███░░
    relevance            4.50  ████░
    citation_quality     3.80  ███░░
    hedging              4.10  ████░

============================================================
```

Results are also stored in PostgreSQL (`eval_runs` and `eval_scores` tables) for querying and comparison.

## What It Records

Each eval run captures:
- Agent git branch and commit SHA
- Tool catalog snapshot (which MCP servers, how many tools)
- Judge model used
- Per-question scores with justifications
- Aggregate scores per dimension

This lets you compare "agent on branch A with 39 tools" vs "agent on branch B with 40 tools."

## Human Review via Argilla

Push scored answers to Argilla for human review. Judge scores appear as suggestions that reviewers can accept or override.

```bash
# Run eval and push to Argilla
uv run python -m src.eval run --questions eval/questions/friendly_battery.json --push-argilla

# Sync human annotations back to the database
uv run python -m src.eval argilla-sync --dataset eval-feature-eval-pipeline --run-id <run-id>

# Clean up branch dataset after merge
uv run python -m src.eval cleanup --branch feature/eval-pipeline
```

Requires Argilla server running and configured:
```bash
ARGILLA_URL=http://localhost:6900
ARGILLA_API_KEY=your-api-key
```

Install the Argilla SDK: `uv sync --extra eval`

## Reports

Generate markdown reports for different audiences:

```bash
# Team weekly report — dimension scores, worst answers, gaps
uv run python -m src.eval report --format team --since 7d

# Leadership monthly summary — composite trend, capability breakdown
uv run python -m src.eval report --format leadership --since 30d

# Per-resource report — scores for questions about a specific resource
uv run python -m src.eval report --format resource --resource Delta --since 30d
```

Reports use human scores where available, falling back to judge scores.

## Exploring Eval Data

Ask natural language questions about eval results:

```bash
uv run python -m src.eval ask "What are the worst scoring answers this week?"
uv run python -m src.eval ask "How does correctness compare across capability areas?"
uv run python -m src.eval ask "Show me questions where judge and human scores disagree"
```

The LLM generates SQL against the eval tables, runs it, and returns a natural language summary with data.

## All Commands

| Command | Purpose |
|---------|---------|
| `python -m src.eval run` | Pre-production eval (default: friendly battery) |
| `python -m src.eval run --push-argilla` | Eval + push to Argilla for human review |
| `python -m src.eval compare --run-a ID --run-b ID` | Compare two eval runs |
| `python -m src.eval report --format team\|leadership\|resource` | Generate report |
| `python -m src.eval ask "question"` | Query eval data with natural language |
| `python -m src.eval argilla-sync --dataset NAME --run-id ID` | Pull human scores from Argilla |
| `python -m src.eval cleanup --branch NAME` | Delete Argilla branch dataset |
