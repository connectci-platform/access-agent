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
    assert prompt == fixture_content.rstrip("\n")
