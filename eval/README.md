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

## Requirements

- `OPENAI_API_KEY` set in `.env` (for the agent and the judge LLM)
- MCP servers accessible (for tool-calling questions)
- PostgreSQL running (for storing results)

## Question Sets

Eval questions live in `eval/questions/` as JSON files:

| File | Questions | Description |
|------|-----------|-------------|
| `friendly_battery.json` | 50 | Clean, well-phrased questions covering all capability areas |
| `real_user_battery.json` | 50 | Real user queries with typos, vague phrasing, pasted errors |

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

### Adding Questions

Add questions to an existing JSON file or create a new one. Each question needs an `id`, `question`, and `capability_area`. Run against a subset with `--questions your_file.json`.

## Scoring Rubric

Five dimensions, each scored 1-5:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Correctness | 30% | Does the answer match its sources? |
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
