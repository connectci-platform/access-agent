"""Smoke test for the UKY Qwen3.6 endpoint via our LLM provider wrapper.

Verifies three things end-to-end:
  1. We can reach UKY's LiteLLM/vLLM endpoint with our config.
  2. The `</think>` reasoning trace is stripped from the response.
  3. Optional: `enable_thinking=False` actually disables reasoning at the source.

Usage:
    Set the following in `.env` (or your shell):
        LLM_PROVIDER=vllm
        VLLM_BASE_URL=https://jump-external.ccs.uky.edu/v1
        VLLM_API_KEY=<UKY-supplied key>
        VLLM_MODEL_NAME=ccs/Qwen/Qwen3.6-35B-A3B-FP8

    Then:
        .venv/bin/python qwen_smoke.py

The full agent path (graph + tools + RAG) is exercised by `test_agent.py`;
this script tests only the LLM-layer integration.
"""

import asyncio
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

env_path = _ROOT / ".env"
if env_path.exists():
    with env_path.open() as f:
        for raw_line in f:
            stripped = raw_line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, value = stripped.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

from langchain_core.messages import HumanMessage  # noqa: E402

from src.config import settings  # noqa: E402
from src.llm import get_llm  # noqa: E402
from src.llm.providers import _StrippingChatOpenAI  # noqa: E402


def assert_config() -> None:
    if settings.LLM_PROVIDER != "vllm":
        sys.exit(f"LLM_PROVIDER must be 'vllm' for this smoke test, got {settings.LLM_PROVIDER!r}")
    missing = [
        name
        for name in ("VLLM_BASE_URL", "VLLM_API_KEY", "VLLM_MODEL_NAME")
        if not getattr(settings, name)
    ]
    if missing:
        sys.exit(f"Missing required env: {', '.join(missing)}")
    print(f"  base_url:  {settings.VLLM_BASE_URL}")
    print(f"  model:     {settings.VLLM_MODEL_NAME}")


async def call_with_thinking() -> None:
    print("\n[1/2] Default call (thinking on, expect reasoning to be stripped)")
    llm = get_llm()
    assert isinstance(llm, _StrippingChatOpenAI), (
        f"Expected _StrippingChatOpenAI, got {type(llm).__name__}"
    )
    response = await llm.ainvoke([HumanMessage(content="In one sentence, what is ACCESS-CI?")])
    content = response.content if isinstance(response.content, str) else str(response.content)
    print(f"  response: {content!r}")
    if "</think>" in content:
        sys.exit("FAIL: '</think>' still present in response — strip didn't run")
    if "<think>" in content:
        sys.exit("FAIL: '<think>' present in response — partial strip")
    print("  PASS: no thinking artifacts in response")


async def call_without_thinking() -> None:
    print("\n[2/2] Call with enable_thinking=False (expect short answer, no reasoning trace)")
    llm = get_llm(enable_thinking=False)
    response = await llm.ainvoke([HumanMessage(content="In one sentence, what is ACCESS-CI?")])
    content = response.content if isinstance(response.content, str) else str(response.content)
    print(f"  response: {content!r}")
    if "</think>" in content:
        sys.exit("FAIL: '</think>' present even with enable_thinking=False")
    print("  PASS: clean response with thinking disabled")


async def main() -> None:
    print("Qwen smoke test — UKY endpoint")
    assert_config()
    await call_with_thinking()
    await call_without_thinking()
    print("\nAll checks passed.")


if __name__ == "__main__":
    asyncio.run(main())
