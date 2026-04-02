"""Load eval question sets from JSON files."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalQuestion:
    id: str
    question: str
    capability_area: str = "general"
    battery: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_questions(path: str) -> list[EvalQuestion]:
    """Load questions from a JSON file."""
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    with file_path.open() as f:
        data = json.load(f)

    questions = []
    for item in data:
        questions.append(
            EvalQuestion(
                id=item["id"],
                question=item["question"],
                capability_area=item.get("capability_area", "general"),
                battery=item.get("battery", ""),
                metadata={
                    k: v
                    for k, v in item.items()
                    if k not in ("id", "question", "capability_area", "battery")
                },
            )
        )

    logger.info(f"Loaded {len(questions)} questions from {path}")
    return questions
