"""Golden-fixture guard: the no-history judge prompt must stay byte-identical
to the pre-conversation_history prompt, so existing runs stay comparable."""

from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "judge_prompt_golden.txt"

GOLDEN_ARGS = {
    "query": "Which ACCESS resources have PyTorch installed?",
    "answer": "Delta and Anvil have PyTorch available via modules.",
    "rag_context": "Score: 0.9\nPyTorch is available on several ACCESS resources.",
    "tool_results": (
        "### Tool call: search_software\n"
        '- arguments: {"query": "pytorch"}\n'
        "- success: True\n- duration_ms: 120\n- result_count: 2\n- empty: False\n"
        "- data:\n[{'resource': 'Delta'}, {'resource': 'Anvil'}]"
    ),
    "node_trace": '[{"node": "tool_calling_loop", "tool_calls": 1}]',
    "required_facts": [
        {"fact_id": "golden-f1", "fact_text": "Answer names Delta among PyTorch resources."},
        "Answer does not invent versions not present in tool results.",
    ],
}


def test_no_history_prompt_matches_pre_change_golden():
    from src.eval.rubric import build_judge_prompt

    prompt = build_judge_prompt(**GOLDEN_ARGS)
    fixture_content = FIXTURE.read_text()
    assert prompt == fixture_content.removesuffix("\n")


def test_history_renders_conversation_section():
    from src.eval.rubric import build_judge_prompt

    prompt = build_judge_prompt(
        **GOLDEN_ARGS,
        conversation_history=[
            ("What GPU resources does ACCESS offer?", "Delta, Anvil, and Bridges-2 offer GPUs."),
            ("Which have A100s?", "(no answer — turn failed)"),
        ],
    )
    assert "## Conversation so far" in prompt
    assert "Turn 1 — User: What GPU resources does ACCESS offer?" in prompt
    assert "Turn 2 — Assistant: (no answer — turn failed)" in prompt
    # Section sits above the current question
    assert prompt.index("## Conversation so far") < prompt.index("## User Question")


def test_empty_history_is_byte_identical_to_none():
    from src.eval.rubric import build_judge_prompt

    assert build_judge_prompt(
        **GOLDEN_ARGS, conversation_history=[]
    ) == FIXTURE.read_text().removesuffix("\n")


def test_history_present_adds_conversation_as_valid_support():
    from src.eval.rubric import build_judge_prompt

    prompt = build_judge_prompt(
        **GOLDEN_ARGS,
        conversation_history=[
            ("What GPU resources does ACCESS offer?", "Delta, Anvil, and Bridges-2 offer GPUs."),
        ],
    )
    assert (
        "supported by the Tool Results, RAG Documents, the prior turns shown in "
        "'Conversation so far', or well-known ACCESS-CI facts" in prompt
    )
    assert (
        "A value the agent correctly restates from an earlier turn (visible in "
        "'Conversation so far') IS supported." in prompt
    )
    assert (
        "supported either by the Tool Results, by the RAG Documents, by the prior turns "
        "shown in 'Conversation so far', or by widely known ACCESS-CI facts" in prompt
    )


def test_no_history_omits_conversation_support_clauses():
    from src.eval.rubric import build_judge_prompt

    prompt = build_judge_prompt(**GOLDEN_ARGS)
    assert "the prior turns shown in 'Conversation so far'" not in prompt
    assert "A value the agent correctly restates from an earlier turn" not in prompt


def test_empty_history_list_also_omits_conversation_support_clauses():
    from src.eval.rubric import build_judge_prompt

    prompt = build_judge_prompt(**GOLDEN_ARGS, conversation_history=[])
    assert "the prior turns shown in 'Conversation so far'" not in prompt
    assert "A value the agent correctly restates from an earlier turn" not in prompt
