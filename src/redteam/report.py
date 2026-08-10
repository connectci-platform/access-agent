"""Redaction contract + reporting. Only id+verdict+hash leaves on-prem."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def content_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def redacted_line(prompt_id: str, verdict: str, response_text: str) -> str:
    """Redacted line computing the hash from the ACTUAL response text."""
    return f"{prompt_id}  {verdict}  sha256:{content_hash(response_text)}"


@dataclass(frozen=True)
class Flag:
    prompt_id: str
    verdict: str
    content_hash: str  # precomputed hash of the flagged response
    kind: str  # candidate-regression | candidate-fix


def flag_line(f: Flag) -> str:
    """Redacted line for an already-hashed Flag (does NOT recompute)."""
    return f"[{f.kind}] {f.prompt_id}  {f.verdict}  sha256:{f.content_hash}"


def issue_body(flags: list[Flag]) -> str:
    lines = ["Red-team nightly surfaced candidate(s). Full transcripts are on-prem only.\n"]
    for f in flags:
        lines.append(f"- {flag_line(f)}")
    return "\n".join(lines)


def write_artifact(path: Path, records: list[dict[str, object]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(records, indent=2))
