"""Evaluate node - quality assessment of tool results.

This node evaluates whether tool results adequately answer the user's query.
If not helpful, it triggers a retry with different tools/parameters.
"""

import logging
from typing import Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ...config import settings
from ...llm import get_llm
from ..state import AgentState, QualityEvaluation

logger = logging.getLogger(__name__)

EVALUATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You evaluate whether tool results adequately answer the user's question.

Analyze the results and determine:
1. Do they contain information relevant to the question?
2. Can a useful answer be synthesized from this data?
3. Is there substantive content (not just errors or empty results)?

Respond with JSON only:
{{
  "is_helpful": true or false,
  "confidence": "high" | "medium" | "low",
  "reason": "brief explanation of your evaluation",
  "missing_information": "what's still needed to answer" or null if complete
}}

Guidelines:
- Mark as HELPFUL if the results contain relevant data that can answer the question, even partially
- Mark as NOT helpful only if results are empty, completely irrelevant, or all failed
- Partial information that addresses the question should be considered helpful
- Don't require perfect or complete answers - useful partial information is helpful""",
        ),
        (
            "human",
            """ORIGINAL QUESTION:
{query}

TOOLS EXECUTED:
{tools_summary}

RESULTS:
{results_summary}

Evaluate whether these results answer the question.""",
        ),
    ]
)


async def evaluate_node(state: AgentState) -> dict[str, Any]:
    """Evaluate if tool results adequately answer the query.

    This node:
    1. Checks if there are any results to evaluate
    2. Summarizes the tools and results for the LLM
    3. Gets LLM evaluation of result quality
    4. Returns quality evaluation for routing decisions

    Args:
        state: Current agent state with tool_results.

    Returns:
        Dict with quality_evaluation and incremented attempt_number.
    """
    query = state["query"]
    tool_results = state.get("tool_results", [])
    attempt_number = state.get("attempt_number", 0)

    def _trace(is_helpful, reason):
        return [{"node": "evaluate", "is_helpful": is_helpful, "attempt": attempt_number, "reason": reason[:100]}]

    # If no tools were needed, skip evaluation
    query_analysis = state.get("query_analysis")
    if query_analysis and not query_analysis.requires_tools:
        return {
            "quality_evaluation": QualityEvaluation(
                is_helpful=True,
                confidence="high",
                reason="No tools needed for this query",
            ),
            "attempt_number": attempt_number,
            "node_trace": _trace(True, "no_tools_needed"),
        }

    # Handle empty results
    if not tool_results:
        logger.warning("No tool results to evaluate")
        return {
            "quality_evaluation": QualityEvaluation(
                is_helpful=False,
                confidence="high",
                reason="No tool results available",
                missing_information="Tool execution required",
            ),
            "attempt_number": attempt_number + 1,
            "node_trace": _trace(False, "no_results"),
        }

    # Check if all tools failed
    all_failed = all(not r.success for r in tool_results)
    if all_failed:
        logger.warning("All tools failed - marking as unhelpful")
        errors = [r.error for r in tool_results if r.error]
        return {
            "quality_evaluation": QualityEvaluation(
                is_helpful=False,
                confidence="high",
                reason=f"All tools failed: {'; '.join(errors[:2])}",
                missing_information="Need successful tool execution",
            ),
            "attempt_number": attempt_number + 1,
            "node_trace": _trace(False, "all_tools_failed"),
        }

    # Build summaries for LLM evaluation
    tools_summary = _build_tools_summary(tool_results)
    results_summary = _build_results_summary(tool_results)

    # Get LLM evaluation
    llm = get_llm(temperature=0.1, max_tokens=500)
    chain = EVALUATION_PROMPT | llm | JsonOutputParser()

    try:
        result = await chain.ainvoke(
            {
                "query": query,
                "tools_summary": tools_summary,
                "results_summary": results_summary,
            }
        )

        evaluation = QualityEvaluation(
            is_helpful=result.get("is_helpful", False),
            confidence=result.get("confidence", "medium"),
            reason=result.get("reason", ""),
            missing_information=result.get("missing_information"),
        )

        logger.info(
            f"Quality evaluation: is_helpful={evaluation.is_helpful}, "
            f"confidence={evaluation.confidence}, reason={evaluation.reason[:50]}"
        )

        return {
            "quality_evaluation": evaluation,
            "attempt_number": attempt_number + 1 if not evaluation.is_helpful else attempt_number,
            "node_trace": _trace(evaluation.is_helpful, evaluation.reason),
        }

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return {
            "quality_evaluation": QualityEvaluation(
                is_helpful=True,
                confidence="low",
                reason=f"Evaluation error: {e}",
            ),
            "attempt_number": attempt_number,
            "node_trace": _trace(True, f"error: {e}"),
        }


def _build_tools_summary(results: list[Any]) -> str:
    """Build a summary of tools executed."""
    lines = []
    for r in results:
        status = "SUCCESS" if r.success else f"FAILED ({r.error})"
        lines.append(f"- {r.tool_name}: {status}")
    return "\n".join(lines)


def _build_results_summary(results: list[Any]) -> str:
    """Build a summary of tool results for evaluation.

    Uses settings for length limits. Modern LLMs have large context windows,
    so limits are generous.
    """
    import json

    max_length = settings.MAX_TOOL_RESULT_LENGTH
    max_single = settings.MAX_SINGLE_RESULT_LENGTH

    summaries = []
    for r in results:
        if r.success and r.data:
            try:
                data_str = json.dumps(r.data, indent=2, default=str)
                if len(data_str) > max_single:
                    data_str = data_str[:max_single] + f"... (truncated from {len(data_str)} chars)"
                summaries.append(f"### {r.tool_name}\n{data_str}")
            except (TypeError, ValueError):
                summaries.append(f"### {r.tool_name}\n{str(r.data)[:max_single]}")
        elif not r.success:
            summaries.append(f"### {r.tool_name}\nError: {r.error}")

    result = "\n\n".join(summaries)
    if len(result) > max_length:
        result = result[:max_length] + f"\n... (truncated from {len(result)} chars)"
    return result
