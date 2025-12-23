"""Conditional edge routing functions for the agent graph.

These functions determine which node to transition to based on state.
"""

import logging
from typing import Literal

from ...config import settings
from ..state import AgentState

logger = logging.getLogger(__name__)


def should_execute_tools(state: AgentState) -> Literal["execute", "compress"]:
    """Determine whether to execute tools or go directly to compression/synthesis.

    Routes to 'execute' if tools are needed, otherwise to 'compress'.

    Args:
        state: Current agent state.

    Returns:
        The name of the next node.
    """
    query_analysis = state.get("query_analysis")

    # If planning failed or determined no tools needed, skip execution
    if query_analysis is None:
        return "compress"

    if not query_analysis.requires_tools:
        return "compress"

    # If no tools were planned, skip execution
    if not state.get("planned_tools"):
        return "compress"

    return "execute"


def should_recover_or_evaluate(
    state: AgentState,
) -> Literal["recover", "evaluate"]:
    """After execution, check if there are failures that need recovery.

    Routes to 'recover' if any tools failed, otherwise to 'evaluate'.

    Args:
        state: Current agent state with tool_results.

    Returns:
        The name of the next node.
    """
    tool_results = state.get("tool_results", [])

    # Check if any tools failed
    failed_results = [r for r in tool_results if not r.success]

    if failed_results:
        logger.info(f"Found {len(failed_results)} failed tools - routing to recover")
        return "recover"

    return "evaluate"


def should_retry_or_synthesize(
    state: AgentState,
) -> Literal["execute", "compress"]:
    """After recovery, decide whether to retry execution or give up.

    Routes to 'execute' if recovery planned new tools, otherwise to 'compress'.

    Args:
        state: Current agent state.

    Returns:
        The name of the next node.
    """
    planned_tools = state.get("planned_tools", [])

    if planned_tools:
        logger.info(f"Recovery planned {len(planned_tools)} tools - routing to execute")
        return "execute"

    logger.info("No recovery plan - routing to compress")
    return "compress"


def should_retry_quality(
    state: AgentState,
) -> Literal["plan", "compress"]:
    """After evaluation, decide whether to retry with different tools.

    Routes to 'plan' if results were unhelpful and we have attempts left,
    otherwise to 'compress'.

    Args:
        state: Current agent state with quality_evaluation.

    Returns:
        The name of the next node.
    """
    quality_eval = state.get("quality_evaluation")
    attempt_number = state.get("attempt_number", 0)

    # If no evaluation, proceed to compression/synthesis
    if quality_eval is None:
        return "compress"

    # If helpful, compress and synthesize
    if quality_eval.is_helpful:
        logger.info(f"Quality check passed (confidence={quality_eval.confidence})")
        return "compress"

    # Check if we have attempts left
    max_attempts = settings.MAX_QUALITY_ATTEMPTS
    if attempt_number >= max_attempts:
        logger.warning(f"Quality check failed but max attempts ({max_attempts}) reached")
        return "compress"

    # Unhelpful and have attempts left - retry planning
    logger.info(
        f"Quality check failed (attempt {attempt_number}/{max_attempts}): {quality_eval.reason}"
    )
    return "plan"
