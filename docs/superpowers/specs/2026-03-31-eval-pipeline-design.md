# Agent Answer Evaluation Pipeline

**Date:** 2026-03-31
**Status:** Draft
**Repo:** access-agent

## Problem

The agent is deploying to production next week with new capabilities (UKY-first architecture, MCP tool calling, domain agents). There is no systematic way to:

1. Evaluate answer quality before deploying a new agent configuration
2. Monitor answer quality in production beyond success/failure rates
3. Provide stakeholders with meaningful reporting on how well the agent answers real questions
4. Compare two agent configurations to decide which is better

Existing infrastructure covers mechanical correctness (e2e tests check tool selection and keyword presence) and operational health (usage logs, response times, success rates), but not answer quality.

## Solution

An LLM-as-judge evaluation pipeline integrated into access-agent, with human review via Argilla and reporting for three stakeholder audiences.

## Architecture

```
                                   +------------------+
                                   |   Argilla        |
                                   |  (Human Review)  |
                                   |  - Dev datasets  |
                                   |    (per branch)  |
                                   |  - Prod dataset  |
                                   +--------+---------+
                                       push | pull
                                            |
+-------------+    +----------+    +--------+---------+    +-----------+
| Eval        | -> | LLM      | -> | eval_runs        | -> | Reports   |
| Question    |    | Judge    |    | eval_scores       |    | - Team    |
| Set (CSV)   |    | (rubric) |    | (PostgreSQL)      |    | - Leader  |
+-------------+    +----------+    +------------------+    | - RP      |
                        |                                   +-----------+
                        |
+-------------+    +----+-----+
| usage_logs  | -> | Prod     |
| (live       |    | Scorer   |
|  answers)   |    +----------+
+-------------+

+------------------------------------------+
| CLI                                      |
| - python -m src.eval run                 |
| - python -m src.eval score-production    |
| - python -m src.eval report              |
| - python -m src.eval compare             |
| - python -m src.eval ask "..."           |
| - python -m src.eval cleanup             |
+------------------------------------------+
```

Three independent consumers read from the same `eval_runs`/`eval_scores` tables: reports, Argilla sync, and the LLM query interface.

## Components

### 1. Scoring Rubric

Five dimensions, each scored 1-5, with configurable weights for composite score:

| Dimension | Weight | 1 (worst) | 5 (best) |
|-----------|--------|-----------|----------|
| Correctness | 30% | Contradicts sources or hallucinates | Faithfully represents all source material |
| Completeness | 25% | Misses the main point | Thoroughly covers the question |
| Relevance | 20% | Mostly irrelevant content | Focused and directly addresses the query |
| Citation quality | 15% | No URLs or hallucinated URLs | All relevant URLs preserved from sources |
| Appropriate hedging | 10% | Confidently wrong or hedges everything | Calibrated confidence matching source quality |

Weights are configurable in `src/eval/rubric.py` so they can be tuned as the team learns what matters most.

The same rubric is used by both the LLM judge and human reviewers in Argilla. This is what makes scores comparable.

### 2. LLM Judge

The judge receives:
- User query
- Agent answer
- RAG matches the agent had access to
- Tool results the agent received
- Node trace (classification, planning decisions, synthesis strategy)

It returns a structured JSON:
```json
{
  "correctness": { "score": 4, "justification": "Accurately represents GPU specs from source docs" },
  "completeness": { "score": 3, "justification": "Covers Delta but omits Bridges-2 which was also in sources" },
  "relevance": { "score": 5, "justification": "Directly addresses the question with no filler" },
  "citation_quality": { "score": 4, "justification": "3 of 4 source URLs preserved" },
  "hedging": { "score": 5, "justification": "Confident tone matches strong source coverage" },
  "composite": 4.0
}
```

The judge evaluates whether the agent was a good synthesizer of the information it had, not whether the information itself is correct. If UKY has outdated docs, the agent faithfully reporting outdated info scores well on correctness — that's a data quality problem, not an agent quality problem.

**Judge configuration:**
- `EVAL_JUDGE_BASE_URL` — LLM endpoint (cloud API or on-premise). Configurable per environment.
- `EVAL_JUDGE_MODEL` — model name, pinned per eval run and recorded in `eval_runs.llm_model`.
- Model version must be pinned (not "latest") to prevent score drift from model updates.

**Calibration and drift detection:**
- Maintain a frozen calibration set of ~20 questions with known human scores.
- Run the calibration set monthly (or after any judge model change).
- If judge-human agreement on the calibration set drops below 70% weighted agreement, alert the team and flag scores as potentially unreliable.
- Judge model version is recorded in every `eval_runs` row so score trends can be segmented by judge version.

**Malformed output handling:**
- If the judge returns invalid JSON or scores outside 1-5, the scorer retries once with a stricter prompt.
- If the retry also fails, the question is flagged with `source="judge_error"` in `eval_scores` (all dimension scores NULL, justification contains a truncated excerpt of the raw judge output — max 500 chars, with email patterns and obvious PII redacted before storage).
- Judge errors are counted per run; if >10% of questions fail, the run is flagged as unreliable in `eval_runs.metadata`.

### 3. Eval Question Set

~100-150 questions curated from real user queries in `chatbot_log_all_data.csv`:
- 10-15 questions per capability area (general knowledge, resource info, software, outages, XDMoD, allocations, events, announcements, NSF awards, support)
- Mix of easy, medium, and hard questions per area
- No expected answers — the judge scores against sources, not a reference answer

Stored as `eval/questions/base_eval_set.csv` with columns: `id, question, capability_area, difficulty, notes`.

**Curation process:**
1. LLM clusters and deduplicates the ~5,700 queries from chatbot log
2. Sample representative questions from each cluster
3. Human review for coverage and quality
4. Tag with capability area

**Maintenance:**
- Add 10-15 questions when a new capability is added
- Periodically promote low-scoring production queries into the set
- Can run eval against a subset (e.g., only allocations questions when testing a new allocations tool)

### 4. Pre-Production Eval Runner

`python -m src.eval run [--questions subset.csv] [--push-argilla]`

1. Loads eval question set (or subset)
2. Snapshots the environment:
   - Agent git branch and commit SHA
   - Tool catalog (which MCP servers responded, what tools are available)
   - LLM model and config
   - RAG configuration
3. Runs each question through `run_agent()` (same function the API uses)
4. Collects answer + RAG matches + tool results + node trace
5. Sends each Q/A + context to LLM judge
6. Stores results in `eval_runs` and `eval_scores`
7. Optionally pushes low-scoring answers to Argilla for human review (dataset named after branch, e.g., `eval-feature/new-tool`)
8. Prints summary report

### 5. Production Scorer

`python -m src.eval score-production [--since 24h] [--push-argilla]`

1. Pulls recent queries from `usage_logs` (query text, answer, metadata)
2. Retrieves context from stored node traces (RAG matches, tool results)
3. Sends each to LLM judge
4. Stores scores in `eval_scores` linked to `usage_logs` via `question_id`
5. Optionally pushes low-scoring answers to the persistent production Argilla dataset

Runs on a schedule (daily or weekly) alongside existing reporting.

**Missing context fallback:** Not all production queries will have full node traces available (e.g., traces may be truncated, or the query predates trace storage). When context is missing:
- If node trace is completely absent: skip the question and log it as `source="skipped"` with reason `"no_trace"`. Do not score — the judge cannot assess correctness without knowing what the agent saw.
- If node trace is partial (e.g., RAG matches present but tool results missing): score with available context, but record `context_completeness="partial"` in metadata. Reports can filter these out if needed.
- This means production scoring coverage will be <100% of queries. The coverage percentage is reported alongside scores.

**LLM privacy constraint:** Production scoring sends real user queries to the judge LLM. This requires either:
- An on-premise LLM (UKY GPU infrastructure is available for this), or
- An updated privacy policy covering LLM processing of user queries, or
- PII redaction before sending to an external LLM

Until one of these is in place, production LLM scoring is blocked. Human review via Argilla (internal access, no external LLM) is unaffected.

### 6. Human Review via Argilla

**Two dataset types:**

- **Branch datasets** (pre-production): Created per eval run, named after the branch (e.g., `eval-feature/new-allocations-tool`). Team reviews flagged answers before deciding to deploy. Deleted when the branch is merged or abandoned via `python -m src.eval cleanup --branch <name>`.

- **Production dataset** (persistent): Rolling set of production answers for ongoing review. Students or team members review on a continuous basis.

**Argilla record structure:**

Fields (what reviewers see):
- `user_query` — the question
- `agent_answer` — the agent's response (markdown)
- `rag_context` — RAG matches with similarity scores (markdown)
- `tool_results` — MCP tool execution results (markdown)
- `agent_reasoning` — node trace showing classification, planning, synthesis decisions (markdown)

Questions (annotation interface):
- `correctness` — Rating 1-5
- `completeness` — Rating 1-5
- `relevance` — Rating 1-5
- `citation_quality` — Rating 1-5
- `hedging` — Rating 1-5
- `decision` — Label: approved / needs_revision / rejected
- `feedback` — Free text

LLM judge scores appear as suggestions that reviewers can accept or override.

Metadata for filtering: model version, capability area, query type, agent confidence.

**Score sync:** A job pulls completed annotations from Argilla and writes human scores to `eval_scores` with `source="human"` alongside the judge's `source="judge"` entries, linked by `question_id`. This enables direct comparison in the same table.

### 7. Database Schema

**`eval_runs` table:**
```sql
CREATE TABLE eval_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    run_type VARCHAR(16) NOT NULL,          -- "pre_production" or "production"
    agent_commit VARCHAR(40),
    agent_branch VARCHAR(128),
    tool_catalog JSONB,                      -- snapshot of available tools
    llm_model VARCHAR(64),
    question_set VARCHAR(128),               -- CSV filename or "production"
    question_count INT,
    scores_summary JSONB,                    -- aggregate scores per dimension
    composite_score FLOAT,
    metadata JSONB                           -- additional config snapshot
);
```

**`eval_scores` table:**
```sql
CREATE TABLE eval_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    run_id UUID NOT NULL REFERENCES eval_runs(id),
    question_id VARCHAR(64) NOT NULL,        -- matches usage_logs.question_id
    source VARCHAR(16) NOT NULL CHECK (source IN ('judge', 'human', 'judge_error', 'skipped')),
    reviewer_id VARCHAR(128),                -- NULL for judge, user ID for humans
    question_text TEXT,
    answer_text TEXT,
    context JSONB,                           -- RAG matches, tool results, trace
    context_completeness VARCHAR(16),        -- "full", "partial", or NULL
    correctness INT CHECK (correctness BETWEEN 1 AND 5),
    completeness INT CHECK (completeness BETWEEN 1 AND 5),
    relevance INT CHECK (relevance BETWEEN 1 AND 5),
    citation_quality INT CHECK (citation_quality BETWEEN 1 AND 5),
    hedging INT CHECK (hedging BETWEEN 1 AND 5),
    composite_score FLOAT,
    justifications JSONB,                    -- per-dimension justification text
    feedback TEXT,                           -- human reviewer free text

    -- Prevent duplicate scores: separate constraints for machine vs. human rows
    -- because PostgreSQL treats NULL != NULL in unique constraints
);

-- Machine-generated scores (judge, judge_error, skipped): one per question per run
CREATE UNIQUE INDEX eval_scores_machine_unique
    ON eval_scores (run_id, question_id, source)
    WHERE reviewer_id IS NULL;

-- Human scores: one per reviewer per question per run
CREATE UNIQUE INDEX eval_scores_human_unique
    ON eval_scores (run_id, question_id, source, reviewer_id)
    WHERE reviewer_id IS NOT NULL;
```

**Every scoring job creates an `eval_runs` row** — including production scoring jobs. `run_id` is never NULL. A production scoring job for "last 24h" creates a run with `run_type="production"`, and all scores from that job reference it. This eliminates NULL-key ambiguity and makes every score traceable to a specific run.

**Join key convention:** The column is `question_id` everywhere — in `usage_logs`, `eval_scores`, and Argilla record metadata. This matches the existing `usage_logs.question_id` column. The eval pipeline never uses `query_id` as a column name.

**Multiple human annotations:** A single answer can receive scores from multiple reviewers (different `reviewer_id` values). Reports aggregate human scores by averaging across reviewers for the same `question_id`. Inter-rater agreement is reported when multiple reviewers score the same answer.

### 8. Reports

Three formats, all CLI-driven, feeding into existing email/Slack pipeline:

**Team report** (weekly):
- Average scores per dimension, trend vs. last week
- Worst-scoring answers with links to Argilla for review
- Judge vs. human agreement rate (where both exist)
- Content gaps: capability areas with lowest scores
- `python -m src.eval report --format team --since 7d`

**Leadership summary** (monthly):
- Composite quality score trend line
- Volume: queries handled, % reviewed by humans
- Capability area breakdown (table of composite scores)
- Notable improvements/regressions in plain language
- `python -m src.eval report --format leadership --since 30d`

**RP operator view** (per-resource, on demand):
- Scores filtered to questions mentioning a specific resource
- Lowest-scoring answers about that resource with justifications
- `python -m src.eval report --format resource --resource Delta --since 30d`

**Comparison report** (pre-production):
- Side-by-side scores from two eval runs
- Per-dimension delta, per-capability-area delta
- Regressions highlighted
- `python -m src.eval compare --run-a <id> --run-b <id>`

**Score source policy for reports:** Reports use human scores where available, falling back to judge scores. When both exist for the same question, human scores take precedence. The report includes a "coverage" line showing what percentage of scores are human vs. judge, so readers know how much human validation underlies the numbers.

### 9. LLM Query Interface (v1 — CLI)

`python -m src.eval ask "Why did allocation answers regress last week?"`

A small agent with read access to the eval tables via SQL. Handles ad-hoc exploratory questions:
- "Show me the 5 worst answers about Delta this month"
- "What capability areas improved since the last deploy?"
- "How do judge scores compare to human ratings for XDMoD questions?"

Returns a natural language answer with supporting data.

### 10. Future Phases (designed, not built in v1)

**Web UI with LLM** — Standalone lightweight web app for stakeholder exploration. Filters, charts, time period selection, plus an LLM chat interface for ad-hoc questions. Built as a small modular tool that reads from the eval tables.

**Agent self-eval capability** — The main ACCESS agent can answer questions about its own performance, gated by Drupal roles. "How am I doing on allocation questions?" becomes a capability available to team members and leadership.

## Module Structure

```
src/eval/
    __init__.py
    __main__.py      -- CLI entry point
    judge.py         -- LLM judge: scores a Q/A pair against the rubric
    runner.py        -- Runs eval set through the agent, collects answers
    scorer.py        -- Orchestrates: run questions, judge answers, store results
    rubric.py        -- Scoring dimensions, weights, judge prompt
    questions.py     -- Loads eval question sets from CSV
    report.py        -- Generates reports (team, leadership, RP)
    models.py        -- Data models for eval runs, scores, metadata
    argilla_sync.py  -- Push to / pull from Argilla
    ask.py           -- LLM query interface over eval tables
    db.py            -- Database operations for eval tables
eval/
    questions/
        base_eval_set.csv
```

## CLI Commands

| Command | Purpose |
|---------|---------|
| `python -m src.eval run` | Pre-production: run eval set, judge, report |
| `python -m src.eval run --questions allocations.csv` | Run subset |
| `python -m src.eval run --push-argilla` | Also push to Argilla for human review |
| `python -m src.eval score-production --since 24h` | Score recent production answers |
| `python -m src.eval report --format team --since 7d` | Team weekly report |
| `python -m src.eval report --format leadership --since 30d` | Leadership monthly |
| `python -m src.eval report --format resource --resource Delta` | Per-resource |
| `python -m src.eval compare --run-a <id> --run-b <id>` | Compare two runs |
| `python -m src.eval ask "..."` | Exploratory LLM query |
| `python -m src.eval cleanup --branch <name>` | Delete Argilla branch dataset |

## Observability Boundaries

- **Honeycomb/OpenTelemetry**: Operational observability — latency, errors, tool failures. Stays as-is.
- **GA4/Looker Studio**: Engagement — users, sessions, funnels. Stays as-is.
- **Eval pipeline (this spec)**: Quality — answer correctness, completeness, relevance. New layer.

These are complementary. The eval pipeline does not replace or overlap with existing observability.

## Data Governance

### PII and Sensitive Data

**Current state:** `usage_logs` stores `query_text` verbatim. User queries may contain PII (names, emails, account IDs). User identity is hashed (`user_hash`), but query content is not redacted. The `usage_logs` docstring says "No PII is stored" but this applies to user identity, not query content.

**Eval pipeline policy:** The eval pipeline inherits the same data sensitivity as `usage_logs`. It additionally stores `answer_text` and `context` (RAG matches, tool results, node traces). Tool results may contain user-specific data if the agent called personalized tools (e.g., allocation lookups).

**Mitigations:**
- **Pre-production eval:** Questions are curated from historical logs. Curation process must include a sanitization step: strip names, emails, account IDs, and any other PII from questions before inclusion in `base_eval_set.csv`. A CI lint check validates that the eval set contains no email patterns (`@`), no common PII markers. This ensures curated questions are safe to send to any LLM.
- **Production scoring:** Contains real user data. Access must be restricted to authorized team members.
- **Argilla access:** Argilla contains the same data as `eval_scores`. Reviewer access should be controlled. CILogon integration for Argilla is planned but not yet implemented.
- **LLM judge for production data:** Sending real user queries to an external LLM requires either an on-premise LLM, an updated privacy policy, or PII redaction. See "On-Premise LLM Infrastructure" section.

### Access Control

- `eval_runs` and `eval_scores` tables are in the same PostgreSQL database as `usage_logs`. Same access controls apply.
- Argilla access is controlled by its own user accounts (currently local, CILogon proxy planned).
- CLI commands require database credentials (same as existing reporting tools).
- Reports should not be sent to distribution lists that include people without data access authorization.

### Retention

- **Pre-production eval runs:** Retained indefinitely (small volume, useful for historical comparison).
- **Production eval scores:** Follow the same retention policy as `usage_logs` (currently unlimited). Consider a 12-month rolling window if volume becomes a concern.
- **Argilla branch datasets:** Deleted on branch merge/abandon via `cleanup` command.
- **Argilla production dataset:** Rolling; old records can be archived after human review is complete.

## On-Premise LLM Infrastructure

Production scoring requires an LLM that can process real user queries without sending them to external cloud APIs. UKY has GPU capacity available via SSH.

### Requirements

- OpenAI-compatible API endpoint (the judge module uses the same client interface regardless of backend)
- Model capable of reliable rubric-following and structured JSON output (Llama 3.1 70B or comparable recommended)
- Available for batch scoring workloads (not latency-sensitive — minutes per eval run is fine)
- Accessible from the production agent server (SSH tunnel, VPN, or direct network route)

### Recommended Setup

- **Inference server:** vLLM or Ollama on UKY GPU, exposed as OpenAI-compatible HTTP endpoint
- **Model:** Llama 3.1 70B-Instruct (strong rubric adherence, structured output support)
- **Access:** SSH tunnel from agent server to UKY inference endpoint, or reverse proxy with auth
- **Configuration:** `EVAL_JUDGE_BASE_URL=http://uky-gpu:8000/v1` and `EVAL_JUDGE_MODEL=llama-3.1-70b-instruct` in agent environment

### Phasing

1. **Immediate (pre-production eval):** Use cloud LLM (e.g., GPT-4o, Claude). Curated questions only, no PII.
2. **Near-term (production scoring):** Deploy on-premise LLM at UKY. Switch `EVAL_JUDGE_BASE_URL` to UKY endpoint for production scoring.
3. **Optional:** Use on-premise LLM for all eval (including pre-production) for consistency.

The eval pipeline is LLM-agnostic — switching between cloud and on-premise is a configuration change, not a code change.

## What This Does NOT Cover

- Changing the agent's behavior based on eval scores (no automatic feedback loop)
- Evaluating whether source documents (UKY, MCP tools) are correct — only whether the agent faithfully represents them
- Real-time scoring of production answers (scoring is batch, not inline with the request)
- The existing e2e test suite — it continues as-is for mechanical regression testing
