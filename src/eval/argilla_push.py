"""Push scored answers to Argilla for human review.

Creates Argilla datasets with the eval rubric as annotation questions.
Judge scores appear as suggestions that reviewers can accept or override.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def dataset_name_for_branch(branch: str | None) -> str:
    """Generate Argilla dataset name from branch name."""
    if branch is None:
        return "eval-production"
    sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", branch)
    return f"eval-{sanitized}"


def build_argilla_record(
    question_id: str,
    question_text: str,
    answer_text: str,
    judge_scores: dict[str, int],
    composite_score: float,
    capability_area: str = "general",
    rag_context: str | None = None,
    tool_results: str | None = None,
    node_trace: str | None = None,
    run_id: str | None = None,
    agent_branch: str | None = None,
    agent_commit: str | None = None,
    judge_model: str | None = None,
    duration_ms: float | None = None,
) -> dict[str, Any]:
    """Build a record dict for Argilla."""
    return {
        "id": question_id,
        "fields": {
            "user_query": question_text,
            "agent_answer": answer_text,
            "rag_context": rag_context or "",
            "tool_results": tool_results or "",
            "agent_reasoning": node_trace or "",
        },
        "metadata": {
            "capability_area": capability_area,
            "composite_score": str(round(composite_score, 2)),
            "run_id": run_id or "",
            "agent_branch": agent_branch or "",
            "agent_commit": agent_commit or "",
            "judge_model": judge_model or "",
            "duration_ms": str(round(duration_ms, 1)) if duration_ms else "",
        },
        "suggestions": {
            **judge_scores,
        },
    }


def create_eval_dataset(argilla_url: str, argilla_api_key: str, dataset_name: str) -> Any:
    """Create an Argilla dataset with the eval rubric schema. Lazily imports argilla."""
    try:
        import argilla as rg
    except ImportError:
        logger.error("argilla package not installed. Install with: pip install argilla>=2.0.0")
        raise

    client = rg.Argilla(api_url=argilla_url, api_key=argilla_api_key)

    try:
        existing = client.datasets(name=dataset_name)
        if existing:
            logger.info(f"Dataset '{dataset_name}' already exists, reusing")
            return existing
    except Exception:
        pass

    settings = rg.Settings(
        guidelines=(
            "# How to Review\n\n"
            "Evaluate the agent's answer using the scoring rubric below. "
            "The LLM judge's scores appear as suggestions — accept or override based on your judgment.\n\n"
            "Use the **RAG Documents Retrieved** and **MCP Tool Results** panels to verify "
            "the agent's claims against its actual sources. The **Agent Trace** panel shows "
            "how the agent classified the query and which tools it chose.\n\n"
            "## Scoring Rubric (1-5 each)\n\n"
            "- **Correctness**: Does the answer accurately represent its sources? "
            "Score 5 if faithful to sources, even if the sources themselves are outdated. "
            "Score 1-2 if the agent hallucinated or contradicted its sources.\n"
            "- **Completeness**: Does the answer address all parts of the question? "
            "Score 5 if thorough, 1 if it misses the main point.\n"
            "- **Relevance**: Does the answer stay on topic? "
            "Score 5 if focused, 1 if mostly irrelevant padding.\n"
            "- **Citation Quality**: Are URLs present, valid, and from the source docs? "
            "Score 5 if all relevant URLs preserved, 1 if missing or hallucinated.\n"
            "- **Hedging**: Is the agent's confidence calibrated to its source quality? "
            "Score 5 if well-calibrated, 1 if confidently wrong or hedges everything.\n\n"
            "## Decision\n\n"
            "- **Approved**: Answer is good enough to serve to users. Minor quibbles are OK.\n"
            "- **Needs revision**: Answer has real problems (missing key info, wrong facts, bad URLs) "
            "but the agent was on the right track. Indicates the agent or its sources need improvement.\n"
            "- **Rejected**: Answer is actively wrong, misleading, or unhelpful. "
            "Would cause confusion or harm if served to a user.\n\n"
            "## Feedback\n\n"
            "Use the optional feedback field to explain *why* you scored differently from the judge, "
            "or to note issues the rubric doesn't capture (e.g., tone, formatting, missing context)."
        ),
        fields=[
            rg.TextField(name="user_query", title="User Query", required=True),
            rg.TextField(
                name="agent_answer", title="Agent Response", use_markdown=True, required=True
            ),
            rg.TextField(
                name="rag_context",
                title="RAG Documents Retrieved",
                use_markdown=True,
                required=False,
            ),
            rg.TextField(
                name="tool_results", title="MCP Tool Results", use_markdown=True, required=False
            ),
            rg.TextField(
                name="agent_reasoning",
                title="Agent Trace & Reasoning",
                use_markdown=True,
                required=False,
            ),
        ],
        questions=[
            rg.RatingQuestion(
                name="correctness",
                title="Correctness (1=contradicts sources, 5=faithful)",
                values=[1, 2, 3, 4, 5],
                required=True,
            ),
            rg.RatingQuestion(
                name="completeness",
                title="Completeness (1=misses main point, 5=thorough)",
                values=[1, 2, 3, 4, 5],
                required=True,
            ),
            rg.RatingQuestion(
                name="relevance",
                title="Relevance (1=off-topic, 5=focused)",
                values=[1, 2, 3, 4, 5],
                required=True,
            ),
            rg.RatingQuestion(
                name="citation_quality",
                title="Citation Quality (1=no/bad URLs, 5=all preserved)",
                values=[1, 2, 3, 4, 5],
                required=True,
            ),
            rg.RatingQuestion(
                name="hedging",
                title="Hedging (1=miscalibrated, 5=well-calibrated)",
                values=[1, 2, 3, 4, 5],
                required=True,
            ),
            rg.LabelQuestion(
                name="decision",
                title="Decision",
                labels=["approved", "needs_revision", "rejected"],
                required=True,
            ),
            rg.TextQuestion(name="feedback", title="Feedback (optional)", required=False),
        ],
        metadata=[
            rg.TermsMetadataProperty(name="capability_area", title="Capability Area"),
            rg.TermsMetadataProperty(name="composite_score", title="Composite Score"),
            rg.TermsMetadataProperty(name="run_id", title="Eval Run ID"),
            rg.TermsMetadataProperty(name="agent_branch", title="Agent Branch"),
            rg.TermsMetadataProperty(name="agent_commit", title="Agent Commit"),
            rg.TermsMetadataProperty(name="judge_model", title="Judge Model"),
            rg.TermsMetadataProperty(name="duration_ms", title="Duration (ms)"),
        ],
    )

    dataset = rg.Dataset(name=dataset_name, settings=settings)
    dataset.create()
    logger.info(f"Created Argilla dataset '{dataset_name}'")
    return dataset


def push_scores_to_argilla(
    records: list[dict[str, Any]],
    argilla_url: str,
    argilla_api_key: str,
    dataset_name: str,
) -> int:
    """Push scored records to Argilla. Returns count pushed."""
    try:
        import argilla as rg
    except ImportError:
        logger.error("argilla package not installed")
        return 0

    dataset = create_eval_dataset(argilla_url, argilla_api_key, dataset_name)

    argilla_records = []
    for r in records:
        suggestions = []
        for dim_name, score_val in r["suggestions"].items():
            suggestions.append(rg.Suggestion(question_name=dim_name, value=score_val))

        argilla_records.append(
            rg.Record(
                fields=r["fields"],
                metadata=r["metadata"],
                suggestions=suggestions,
                id=r["id"],
            )
        )

    dataset.records.log(argilla_records)
    logger.info(f"Pushed {len(argilla_records)} records to '{dataset_name}'")
    return len(argilla_records)


def delete_dataset(argilla_url: str, argilla_api_key: str, dataset_name: str) -> bool:
    """Delete an Argilla dataset (for branch cleanup)."""
    try:
        import argilla as rg
    except ImportError:
        logger.error("argilla package not installed")
        return False

    client = rg.Argilla(api_url=argilla_url, api_key=argilla_api_key)
    try:
        dataset = client.datasets(name=dataset_name)
        if dataset:
            dataset.delete()
            logger.info(f"Deleted Argilla dataset '{dataset_name}'")
            return True
        logger.warning(f"Dataset '{dataset_name}' not found")
        return False
    except Exception as e:
        logger.error(f"Failed to delete dataset '{dataset_name}': {e}")
        return False
