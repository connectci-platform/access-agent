"""Parallel RAG + Plan node for combined/dynamic queries.

For combined and dynamic queries, we know at classification time that
MCP tools will be needed. Rather than waiting for UKY RAG to return
before starting tool planning, this node runs both concurrently.

The UKY result is still available for synthesis — it just doesn't
block the planner from starting work.
"""

import asyncio
import logging
from typing import Any

from ...telemetry import get_tracer
from ..state import AgentState
from .plan import plan_node
from .rag_answer import rag_answer_node

logger = logging.getLogger(__name__)


async def rag_and_plan_node(state: AgentState) -> dict[str, Any]:
    """Run RAG retrieval and tool planning concurrently.

    Both rag_answer_node and plan_node read from state but write to
    non-overlapping fields:
    - rag_answer_node writes: rag_matches, rag_used, node_trace
    - plan_node writes: query_analysis, planned_tools, execution_strategy, node_trace

    Their results are merged into a single state update.

    Args:
        state: Current agent state (after classification).

    Returns:
        Merged state update from both nodes.
    """
    tracer = get_tracer("access-agent.nodes")

    with tracer.start_as_current_span(
        "agent.rag_and_plan",
        attributes={
            "agent.node": "rag_and_plan",
        },
    ) as span:
        # Run both concurrently
        gather_results = await asyncio.gather(
            rag_answer_node(state),
            plan_node(state),
            return_exceptions=True,
        )
        _rag_raw = gather_results[0]
        _plan_raw = gather_results[1]

        # Handle exceptions — degrade gracefully
        rag_result: dict[str, Any]
        if isinstance(_rag_raw, BaseException):
            logger.error(f"RAG failed in parallel execution: {_rag_raw}")
            span.set_attribute("rag_and_plan.rag_error", str(_rag_raw)[:200])
            rag_result = {"rag_matches": [], "rag_used": False, "node_trace": []}
        else:
            rag_result = _rag_raw

        plan_result: dict[str, Any]
        if isinstance(_plan_raw, BaseException):
            logger.error(f"Plan failed in parallel execution: {_plan_raw}")
            span.set_attribute("rag_and_plan.plan_error", str(_plan_raw)[:200])
            plan_result = {
                "query_analysis": None,
                "planned_tools": [],
                "execution_strategy": "sequential",
                "node_trace": [],
            }
        else:
            plan_result = _plan_raw

        span.set_attribute("rag_and_plan.rag_used", bool(rag_result.get("rag_used")))
        span.set_attribute(
            "rag_and_plan.tools_planned",
            len(plan_result.get("planned_tools", [])),
        )

        # Merge results — node_trace uses an accumulating reducer (operator.add),
        # so we combine both trace lists
        merged: dict[str, Any] = {}
        merged.update(rag_result)
        merged.update(plan_result)

        # Combine node_trace from both (rag first, then plan)
        rag_trace = rag_result.get("node_trace", [])
        plan_trace = plan_result.get("node_trace", [])
        merged["node_trace"] = rag_trace + plan_trace

        logger.info(
            f"Parallel RAG+Plan complete: "
            f"rag_used={rag_result.get('rag_used', False)}, "
            f"tools_planned={len(plan_result.get('planned_tools', []))}"
        )

        return merged
