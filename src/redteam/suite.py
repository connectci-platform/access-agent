"""Load + validate the frozen red-team suite (baseline + fetched prompts)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class SuiteVersionMismatch(RuntimeError):
    def __init__(self, baseline_v: str, prompts_v: str) -> None:
        super().__init__(f"suite_version mismatch: baseline={baseline_v!r} prompts={prompts_v!r}")


class SuiteKeyMismatch(RuntimeError):
    pass


@dataclass(frozen=True)
class Baseline:
    suite_version: str
    scorer_version: str
    verdicts: dict[str, str]  # id -> defended|known-jailbreak|soft (exceptions only)


@dataclass(frozen=True)
class PromptEntry:
    id: str
    text: str
    section: str
    wrapper_id: str | None
    probe_id: str | None
    wrapper_category: str | None
    probe_category: str | None
    wrapper_source: str | None


@dataclass(frozen=True)
class Prompts:
    suite_version: str
    pyrit_version: str
    entries: list[PromptEntry]


@dataclass(frozen=True)
class SuiteItem:
    id: str
    text: str
    expected: str  # defended|known-jailbreak|soft
    entry: PromptEntry


def load_baseline(path: Path) -> Baseline:
    data = json.loads(Path(path).read_text())
    verdicts: dict[str, str] = {}
    for i in data.get("known_jailbreak", []):
        verdicts[i] = "known-jailbreak"
    for i in data.get("soft", []):
        verdicts[i] = "soft"
    return Baseline(
        suite_version=data["suite_version"],
        scorer_version=data["scorer_version"],
        verdicts=verdicts,
    )


def load_prompts(path: Path) -> Prompts:
    data = json.loads(Path(path).read_text())
    entries = [
        PromptEntry(
            id=e["id"],
            text=e["text"],
            section=e["section"],
            wrapper_id=e.get("wrapper_id"),
            probe_id=e.get("probe_id"),
            wrapper_category=e.get("wrapper_category"),
            probe_category=e.get("probe_category"),
            wrapper_source=e.get("wrapper_source"),
        )
        for e in data["prompts"]
    ]
    return Prompts(
        suite_version=data["suite_version"],
        pyrit_version=data["pyrit_version"],
        entries=entries,
    )


def assert_versions_match(baseline: Baseline, prompts: Prompts) -> None:
    if baseline.suite_version != prompts.suite_version:
        raise SuiteVersionMismatch(baseline.suite_version, prompts.suite_version)


def join_suite(baseline: Baseline, prompts: Prompts) -> list[SuiteItem]:
    prompt_ids = {e.id for e in prompts.entries}
    # Every explicit (non-defended) baseline verdict must have a prompt.
    orphans = set(baseline.verdicts) - prompt_ids
    if orphans:
        raise SuiteKeyMismatch(f"baseline ids with no prompt: {sorted(orphans)}")
    items: list[SuiteItem] = []
    for e in prompts.entries:
        expected = baseline.verdicts.get(e.id, "defended")
        items.append(SuiteItem(id=e.id, text=e.text, expected=expected, entry=e))
    return items
