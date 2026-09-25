"""Recursion-budget regression tests for tool_calling_loop_node (#295).

`create_agent` is built with two middlewares here — `_build_context_editing_middleware()`
(hooks `wrap_model_call`, no separate graph node in the BARE, uninstrumented case) and
`_FlaggingSummarizationMiddleware` (hooks `before_model`/`abefore_model`, which add a
`{name}.before_model` graph node). Reading langchain 1.4.1's graph construction
(`.venv/lib/python3.11/site-packages/langchain/agents/factory.py`, `create_agent`) shows
the resulting BARE per-round node sequence is:

    summarization.before_model -> model -> tools -> (loop back to summarization.before_model)

three graph steps per tool-calling round.

Production is NOT bare. `src/main.py` calls `init_telemetry()` at startup, which calls
`LangchainInstrumentor().instrument()` (opentelemetry-instrumentation-langchain /
OpenLLMetry) whenever `OTEL_ENABLED != "false"` (the default). That instrumentor uses
`wrapt` to patch `AgentMiddleware.before_model` / `after_model` / `abefore_model` /
`aafter_model` (and the `*_agent` variants) IN PLACE on the shared base class
(`langchain.agents.middleware.types.AgentMiddleware`). `create_agent` decides whether to
add a `"{middleware}.before_model"` / `"{middleware}.after_model"` graph node by an
identity check against `AgentMiddleware`'s own (unpatched) method — e.g.
`m.__class__.after_model is not AgentMiddleware.after_model`. Once wrapt has replaced
`AgentMiddleware.after_model` with a `FunctionWrapper`, that check reads True for EVERY
middleware, including ones that never overrode `after_model` (both middlewares built here
fall into this: `ContextEditingMiddleware` and `_FlaggingSummarizationMiddleware` neither
override `after_model`/`aafter_model`), because the instrumented method object is no
longer the one the base class was originally defined with. So under instrumentation,
`create_agent` adds a `before_model` AND an `after_model` node per middleware — doubling
this stack's per-round cost from 3 to 6, verified below (`test_instrumented_steps_per_round`).

Since production always runs instrumented, `tool_calling_loop_node` doesn't hardcode a
measured steps-per-round figure at all — `_steps_per_tool_turn(middleware_count)` derives
the WORST CASE directly from `len(middleware)` (`2 * middleware_count + 2`: a before_model
AND an after_model node per middleware, the instrumented ceiling, plus the model and tools
nodes), and `_recursion_limit_for(middleware_count)` derives `recursion_limit` from that. This
is an upper bound that can never undershoot the real per-round cost and adjusts automatically
if a middleware is added or removed. The measured figures below (25 bare gives 8 rounds, 25
instrumented gives 4) are what the #295 production observation (~3 tool rounds / 4 model
calls before the apology fired) actually matches, and remain as regression guards.

These tests build the REAL agent the node builds (same `create_agent` call, same two
middlewares) against a fake chat model that always emits a tool call, so the loop never
ends on its own, and count how many rounds complete before LangGraph raises
`GraphRecursionError`. `LangchainInstrumentor().instrument()` is a process-global,
irreversible-in-practice monkeypatch (see `tests/test_coverage_handler.py`'s
`TestMainDispatch::test_main_dispatches_to_handler`, which triggers it for the rest of any
pytest process that runs after it — a pre-existing test-isolation gap, out of scope here).
The instrumented measurement therefore runs in a subprocess so it never leaks into this
suite's other tests.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from src.agent.nodes.tool_calling_loop import (
    MAX_TOOL_TURNS,
    _build_context_editing_middleware,
    _FlaggingSummarizationMiddleware,
    _recursion_limit_for,
    _steps_per_tool_turn,
)

# Measured under the OLD recursion_limit of 25 (#295) — documents the empirical
# steps-per-round arithmetic this module's docstring describes.
#   bare:         25 = 3*8 + 1  -> 8 rounds  (bare steps/round is 3, for 2 middlewares)
#   instrumented: 25 = 6*4 + 1  -> 4 rounds  (matches _steps_per_tool_turn(2) == 6)
# The instrumented figure is the worst case _steps_per_tool_turn derives,
# because production is always instrumented (src/main.py calls init_telemetry()
# at startup). See test_bare_steps_per_round and test_instrumented_steps_per_round.
OLD_RECURSION_LIMIT = 25
MEASURED_ROUNDS_UNDER_OLD_LIMIT_BARE = 8
MEASURED_ROUNDS_UNDER_OLD_LIMIT_INSTRUMENTED = 4

# This module's real agent build always uses these two middlewares (context
# editing + flagging summarization) — see _build_real_agent and the
# subprocess script below.
_MIDDLEWARE_COUNT = 2


@tool
def noop_tool(x: str = "") -> str:
    """A trivial tool that returns a string, used only to keep the loop going."""
    return "ok"


class _AlwaysToolCallModel(BaseChatModel):
    """Fake chat model that always emits one tool call — the loop never ends.

    Lets these tests measure exactly how many tool-calling rounds LangGraph
    permits under a given recursion_limit, independent of any real LLM's
    tendency to eventually stop calling tools.
    """

    call_count: int = 0

    @property
    def _llm_type(self) -> str:
        return "always-tool-call-fake"

    def bind_tools(self, tools, **kwargs):  # type: ignore[no-untyped-def]
        # BaseChatModel.bind_tools raises NotImplementedError by default;
        # create_agent calls it to attach the tool schema. A no-op override
        # is enough since this fake ignores the schema and always calls the
        # one tool it knows about.
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:  # type: ignore[no-untyped-def]
        del messages, stop, run_manager, kwargs
        self.call_count += 1
        msg = AIMessage(
            content="",
            tool_calls=[{"id": f"call_{self.call_count}", "name": "noop_tool", "args": {}}],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _build_real_agent(model: BaseChatModel) -> object:
    """Build the agent exactly the way tool_calling_loop_node does.

    Same create_agent call, same two middlewares (context editing +
    flagging summarization) — see tool_calling_loop_node's `agent =
    create_agent(...)` call.
    """
    return create_agent(
        model=model,
        tools=[noop_tool],
        system_prompt="test system prompt",
        middleware=[
            _build_context_editing_middleware(),
            _FlaggingSummarizationMiddleware(
                model=model,
                # Zero token counter: these tests measure graph step budget,
                # not summarization triggering: never let compaction fire.
                token_counter=lambda messages: 0,
                trigger=("tokens", 70_000),
                keep=("tokens", 25_000),
            ),
        ],
    )


async def _rounds_completed_before_recursion_error(recursion_limit: int) -> int:
    """Run the always-tool-call agent under recursion_limit; return completed rounds.

    "Completed round" = the fake model was invoked and returned a tool call
    (model.call_count), which GraphRecursionError always interrupts before a
    final non-tool-call answer, since this model never stops calling tools.
    """
    model = _AlwaysToolCallModel()
    agent = _build_real_agent(model)
    messages = [HumanMessage(content="go")]
    with pytest.raises(GraphRecursionError):
        await agent.ainvoke({"messages": messages}, {"recursion_limit": recursion_limit})
    return model.call_count


# Subprocess script that instruments LangChain exactly as src/main.py does at
# startup (via LangchainInstrumentor, the same class init_telemetry() drives),
# then builds the real agent and reports rounds-completed for each
# recursion_limit given on argv. Runs out-of-process so the instrumentation
# (a process-global, effectively-permanent monkeypatch) never leaks into the
# rest of this test suite — see this module's docstring.
_INSTRUMENTED_SUBPROCESS_SCRIPT = """
import asyncio
import json
import sys

from opentelemetry.instrumentation.langchain import LangchainInstrumentor

LangchainInstrumentor().instrument()

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from src.agent.nodes.tool_calling_loop import (
    _build_context_editing_middleware,
    _FlaggingSummarizationMiddleware,
)


class AlwaysToolCallModel(BaseChatModel):
    call_count: int = 0

    @property
    def _llm_type(self):
        return "always-tool-call-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.call_count += 1
        msg = AIMessage(
            content="",
            tool_calls=[{"id": f"call_{self.call_count}", "name": "noop_tool", "args": {}}],
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


@tool
def noop_tool(x: str = "") -> str:
    'A trivial tool that returns a string.'
    return "ok"


async def rounds_for_limit(limit):
    model = AlwaysToolCallModel()
    agent = create_agent(
        model=model,
        tools=[noop_tool],
        system_prompt="test system prompt",
        middleware=[
            _build_context_editing_middleware(),
            _FlaggingSummarizationMiddleware(
                model=model,
                token_counter=lambda messages: 0,
                trigger=("tokens", 70_000),
                keep=("tokens", 25_000),
            ),
        ],
    )
    messages = [HumanMessage(content="go")]
    try:
        await agent.ainvoke({"messages": messages}, {"recursion_limit": limit})
    except GraphRecursionError:
        pass
    return model.call_count


async def main():
    limits = json.loads(sys.argv[1])
    results = {}
    for limit in limits:
        results[limit] = await rounds_for_limit(limit)
    print(json.dumps(results))


asyncio.run(main())
"""


def _rounds_completed_instrumented(recursion_limits: list[int]) -> dict[int, int]:
    """Measure rounds-completed under LangchainInstrumentor, out-of-process.

    Returns {recursion_limit: rounds_completed}. Runs in a subprocess so the
    process-global instrumentation never leaks into this test process (see
    module docstring).
    """
    proc = subprocess.run(
        [sys.executable, "-c", _INSTRUMENTED_SUBPROCESS_SCRIPT, json.dumps(recursion_limits)],
        capture_output=True,
        text=True,
        cwd=__file__.rsplit("/tests/", 1)[0],
        timeout=60,
        check=True,
    )
    # stdout may carry OTel/langchain log lines before the final JSON line;
    # the script's only print() call emits the JSON, so take the last line.
    last_line = proc.stdout.strip().splitlines()[-1]
    return {int(k): v for k, v in json.loads(last_line).items()}


def test_bare_steps_per_round() -> None:
    """Regression guard for #295, BARE condition (no OTel instrumentation):
    the old recursion_limit=25 allowed at most 8 tool rounds, not the
    "roughly 10" the stale comment claimed — three graph steps per round
    (before_model, model, tools), not two.
    """
    rounds = asyncio.run(_rounds_completed_before_recursion_error(OLD_RECURSION_LIMIT))
    assert rounds <= MEASURED_ROUNDS_UNDER_OLD_LIMIT_BARE


def test_instrumented_steps_per_round() -> None:
    """Regression guard for #295, INSTRUMENTED condition — the production
    condition: src/main.py's init_telemetry() calls LangchainInstrumentor()
    .instrument() at startup. Under the old recursion_limit=25, only 4 tool
    rounds completed, matching the production observation of ~3 tool rounds
    before the apology fired. See this module's docstring for the mechanism
    (wrapt patching AgentMiddleware's base-class hooks defeats create_agent's
    identity check for whether a middleware overrides a hook, doubling the
    per-round node count).
    """
    results = _rounds_completed_instrumented([OLD_RECURSION_LIMIT])
    assert results[OLD_RECURSION_LIMIT] <= MEASURED_ROUNDS_UNDER_OLD_LIMIT_INSTRUMENTED


@pytest.mark.parametrize("condition", ["bare", "instrumented"])
def test_fixed_limit_allows_at_least_max_tool_turns_rounds(condition: str) -> None:
    """With the fixed recursion_limit, an always-tool-call model completes at
    least MAX_TOOL_TURNS rounds before the budget trips — pinning the intent
    that MAX_TOOL_TURNS tool turns are actually available, not merely
    "roughly" available as the old comment claimed. Must hold under BOTH the
    bare condition and the instrumented (production) condition, since
    _recursion_limit_for derives the worst-case (instrumented) cost.
    """
    recursion_limit = _recursion_limit_for(_MIDDLEWARE_COUNT)

    if condition == "bare":
        rounds = asyncio.run(_rounds_completed_before_recursion_error(recursion_limit))
    else:
        rounds = _rounds_completed_instrumented([recursion_limit])[recursion_limit]

    assert rounds >= MAX_TOOL_TURNS


def test_fixed_limit_still_trips_eventually() -> None:
    """The recursion limit still trips for a model that never stops calling
    tools — the apology path (tool_calling_loop_node's GraphRecursionError
    handling) stays exercised.
    """
    recursion_limit = _recursion_limit_for(_MIDDLEWARE_COUNT)
    rounds = asyncio.run(_rounds_completed_before_recursion_error(recursion_limit))
    # An always-tool-call model can never itself terminate the loop, so the
    # recursion budget being finite is what stops it — this rounds count is
    # itself the proof the exception fired (see pytest.raises above).
    assert rounds > 0


@pytest.mark.parametrize(
    ("middleware_count", "expected_steps_per_turn", "expected_recursion_limit"),
    [
        (0, 2, 22),  # no middleware: just model + tools nodes per round
        (1, 4, 42),  # one middleware: before_model + after_model + model + tools
        (2, 6, 62),  # this node's actual stack (context editing + summarization)
    ],
)
def test_steps_per_tool_turn_and_recursion_limit_formula(
    middleware_count: int, expected_steps_per_turn: int, expected_recursion_limit: int
) -> None:
    """Direct unit test of the worst-case formula (#295), independent of any
    real create_agent build: _steps_per_tool_turn(n) = 2*n + 2 (a before_model
    AND an after_model node per middleware — the instrumented ceiling — plus
    the model and tools nodes), and _recursion_limit_for(n) = MAX_TOOL_TURNS *
    _steps_per_tool_turn(n) + 2.
    """
    assert _steps_per_tool_turn(middleware_count) == expected_steps_per_turn
    assert _recursion_limit_for(middleware_count) == expected_recursion_limit
