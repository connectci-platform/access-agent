"""LLM query interface over eval tables.

A small agent that translates natural language questions about eval data
into SQL queries and returns natural language answers.
"""

import logging
import re

from openai import OpenAI

from .db import EvalDB

logger = logging.getLogger(__name__)


def build_schema_description() -> str:
    """Describe the eval tables for the LLM."""
    return """
## Database Schema

### eval_runs
One row per eval run.
- id (UUID), created_at (timestamp), run_type ("pre_production"/"production")
- agent_commit, agent_branch, tool_catalog (JSON), llm_model, judge_model
- question_set, question_count, scores_summary (JSON), composite_score (float 0-1)

### eval_scores
One row per scored answer.
- id (UUID), created_at (timestamp), run_id (FK to eval_runs)
- question_id, source ("judge"/"human"/"judge_error"/"skipped"), reviewer_id
- question_text, answer_text, context (JSON)
- correctness (0-2), specificity (0-2, NULL when N/A), relevance (0-2),
  citation_quality (0-2), hedging (0-1)  -- 3-point ordinal (hedging 2-point), higher=better
- specificity_na (bool), answerable (bool; NULL=pre-v2; FALSE rows excluded from aggregates)
- rubric_version (2 for v2 rows), composite_score (float 0-1), justifications (JSON), feedback (text)

## Weights (v2 default, overridable): correctness 40%, specificity 25%, relevance 15%, citation_quality 12%, hedging 8%
## Composite is per-dimension normalized to [0,1] then weighted; specificity N/A rows are skipped.
"""


def build_ask_prompt(question: str) -> str:
    """Build the prompt for the LLM query agent."""
    return f"""You are a data analyst for an AI agent evaluation system.

{build_schema_description()}

## Rules
- Write PostgreSQL-compatible SQL to answer the question.
- Return the SQL in a ```sql code block, then explain what you found.
- source="judge" is the LLM judge, source="human" is a human reviewer.
- Human scores take precedence when both exist.
- Keep your answer concise.

## Question
{question}
"""


def ask(
    question: str,
    database_url: str,
    api_key: str | None = None,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
) -> str:
    """Ask a natural language question about eval data."""
    db = EvalDB(database_url)
    client = OpenAI(api_key=api_key or "not-needed", base_url=base_url)

    prompt = build_ask_prompt(question)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1000,
    )
    llm_response = response.choices[0].message.content or ""

    sql_match = re.search(r"```sql\s*\n?(.*?)\n?\s*```", llm_response, re.DOTALL)
    if sql_match:
        sql = sql_match.group(1).strip()
        # Safety: reject anything that isn't a read-only query.
        # Check for dangerous statements even within CTEs or subqueries.
        sql_upper = sql.upper()
        dangerous = [
            "INSERT",
            "UPDATE",
            "DELETE",
            "DROP",
            "ALTER",
            "CREATE",
            "TRUNCATE",
            "GRANT",
            "REVOKE",
        ]
        for keyword in dangerous:
            if re.search(rf"\b{keyword}\b", sql_upper):
                return f"Error: Only read-only queries are allowed (found {keyword}).\n\nLLM suggested:\n{sql}"
        if not (sql_upper.strip().startswith("SELECT") or sql_upper.strip().startswith("WITH")):
            return f"Error: Query must start with SELECT or WITH.\n\nLLM suggested:\n{sql}"
        try:
            rows, columns = db.execute_readonly_sql(sql)
            if rows:
                data_str = "\n".join(
                    str(dict(zip(columns, row, strict=False))) for row in rows[:20]
                )
                return f"{llm_response}\n\n---\n**Query Results ({len(rows)} rows):**\n```\n{data_str}\n```"
            return f"{llm_response}\n\n---\n**Query returned no results.**"
        except Exception as e:
            return f"{llm_response}\n\n---\n**SQL Error:** {e}"

    return llm_response
