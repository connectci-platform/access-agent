"""Call the loop node the way LangGraph does: with a Runtime carrying an AgentContext.

The node reads who the run acts for from ``runtime.context``, not from state,
so a direct call has to supply one.
"""

from typing import Any

from langgraph.runtime import Runtime

from src.agent.context import AgentContext


def context_for(acting_user: str | None = None, *, can_write: bool = True) -> AgentContext:
    """An anonymous context, or a signed-in one that may or may not make changes."""
    return AgentContext.for_eval(acting_user, allow_writes=can_write)


def runtime_for(acting_user: str | None = None, *, can_write: bool = True) -> Runtime[AgentContext]:
    return Runtime(context=context_for(acting_user, can_write=can_write))


async def run_loop(
    state: dict[str, Any], acting_user: str | None = None, *, can_write: bool = True
) -> dict[str, Any]:
    from src.agent.nodes.tool_calling_loop import tool_calling_loop_node

    return await tool_calling_loop_node(state, runtime_for(acting_user, can_write=can_write))
