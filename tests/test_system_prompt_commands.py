"""The system prompt must carry the safe-commands rules (D8-3044).

QA caught the assistant suggesting `scancel -u <user>` to cancel pending jobs
(which kills running ones too), a script requesting 16 cores with
OMP_NUM_THREADS=1, and an unsourced claim about how job arrays are counted.
These pin stable phrases for each rule in both prompt branches.
"""

import pytest

from src.agent.prompts.system_prompt import build_system_prompt


@pytest.fixture(params=[None, "jsmith@access-ci.org"])
def prompt(request):
    return build_system_prompt(acting_user=request.param)


def test_destructive_commands_are_scoped_and_explained(prompt):
    assert "Commands and job scripts" in prompt
    assert "exactly what they asked about" in prompt
    assert "say what it will affect" in prompt
    assert "scancel -u <username> -t PENDING" in prompt
    assert "scancel <jobid>" in prompt


def test_thread_counts_match_requested_resources(prompt):
    assert "must match the resources" in prompt
    assert "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" in prompt


def test_scheduler_behavior_needs_a_source(prompt):
    assert "job arrays, priority or charges" in prompt
    assert "unless a tool result or the documentation says so" in prompt
    assert "ask its support" in prompt
