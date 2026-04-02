"""Pull human annotations from Argilla back to eval_scores."""

import logging
from typing import Any

from .db import EvalDB
from .rubric import DIMENSION_NAMES, compute_composite

logger = logging.getLogger(__name__)


def extract_scores_from_response(
    response: dict[str, Any],
    reviewer_id: str,
) -> dict[str, Any]:
    """Extract scores from an Argilla annotation response."""
    scores = {name: response[name] for name in DIMENSION_NAMES}
    decision = response.get("decision")
    feedback_text = response.get("feedback")
    # Combine decision and feedback into the feedback field
    # since EvalScore doesn't have a separate decision column
    feedback_parts = []
    if decision:
        feedback_parts.append(f"Decision: {decision}")
    if feedback_text:
        feedback_parts.append(feedback_text)
    return {
        "source": "human",
        "reviewer_id": reviewer_id,
        "correctness": scores["correctness"],
        "completeness": scores["completeness"],
        "relevance": scores["relevance"],
        "citation_quality": scores["citation_quality"],
        "hedging": scores["hedging"],
        "composite_score": compute_composite(scores),
        "feedback": "\n".join(feedback_parts) if feedback_parts else None,
    }


def sync_from_argilla(
    db: EvalDB,
    argilla_url: str,
    argilla_api_key: str,
    dataset_name: str,
    run_id: str | None = None,
) -> dict[str, int]:
    """Sync human annotations from Argilla to eval_scores."""
    try:
        import argilla as rg
    except ImportError:
        logger.error("argilla package not installed")
        return {"synced": 0, "skipped": 0, "errors": 1}

    client = rg.Argilla(api_url=argilla_url, api_key=argilla_api_key)
    stats = {"synced": 0, "skipped": 0, "errors": 0}

    try:
        dataset = client.datasets(name=dataset_name)
        if not dataset:
            logger.error(f"Dataset '{dataset_name}' not found")
            return {"synced": 0, "skipped": 0, "errors": 1}
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"synced": 0, "skipped": 0, "errors": 1}

    for record in dataset.records(with_responses=True):
        if not record.responses:
            stats["skipped"] += 1
            continue

        question_id = record.id

        # Argilla v2 returns flat list of Response objects, one per question per reviewer.
        # Group by user_id to reconstruct per-reviewer submissions.
        by_reviewer: dict[str, dict[str, Any]] = {}
        for resp in record.responses:
            uid = str(getattr(resp, "user_id", None) or "unknown")
            if uid not in by_reviewer:
                by_reviewer[uid] = {}
            qname = getattr(resp, "question_name", None)
            val = getattr(resp, "value", None)
            if qname and val is not None:
                by_reviewer[uid][qname] = val

        for reviewer_id, response_dict in by_reviewer.items():
            try:
                # Check all dimensions are present before proceeding
                missing = [name for name in DIMENSION_NAMES if name not in response_dict]
                if missing:
                    logger.warning(f"Record {question_id}: missing dimensions {missing}, skipping")
                    stats["errors"] += 1
                    continue

                extracted = extract_scores_from_response(response_dict, reviewer_id)

                db.add_score(
                    run_id=run_id,
                    question_id=str(question_id),
                    question_text=record.fields.get("user_query", ""),
                    answer_text=record.fields.get("agent_answer", ""),
                    **extracted,
                )
                stats["synced"] += 1

            except Exception as e:
                logger.warning(f"Failed to sync record {question_id}: {e}")
                stats["errors"] += 1

    logger.info(f"Argilla sync complete: {stats}")
    return stats
