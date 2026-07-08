"""Load stable-id required facts from the dashboard-owned reporting.question_facts.

The eval keys fact verdicts by a stable ``fact_id`` so a fact's A/B verdicts stay
joinable across runs even when the authored fact list is edited. Facts live in the
dashboard's ``reporting.question_facts`` table, which shares the same Postgres the
eval connects to (DATABASE_URL, as the postgres superuser). This module reads that
table; when the schema is missing (local dev without the dashboard tables) or the
question has no facts, it returns ``None`` so the caller falls back to the YAML
battery's ``required_facts``.

Selection rule (from product): for each ``fact_id``, take the LATEST version
(highest ``version``) and include it unless its status is ``flagged``. Confirmed and
draft are both included; only ``flagged`` is excluded. A fact whose latest version is
flagged is dropped entirely even if an earlier version was confirmed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from sqlalchemy import text as sa_text
from sqlalchemy.exc import SQLAlchemyError

if TYPE_CHECKING:
    from .db import EvalDB

logger = logging.getLogger(__name__)

# DISTINCT ON picks the latest version per fact_id first (ORDER BY version DESC),
# then the outer filter drops any fact whose latest version is flagged. Ordered by
# display_order for stable prompt rendering.
_LATEST_NON_FLAGGED_SQL = """
SELECT fact_id, fact_text
FROM (
    SELECT DISTINCT ON (fact_id)
        fact_id, fact_text, status, display_order
    FROM reporting.question_facts
    WHERE question_id = :question_id
    ORDER BY fact_id, version DESC
) latest
WHERE status <> 'flagged'
ORDER BY display_order, fact_id
"""

# SQLite (tests) has no DISTINCT ON. Emulate latest-per-fact_id via a correlated
# max(version) subquery, then exclude flagged and order by display_order.
_LATEST_NON_FLAGGED_SQL_SQLITE = """
SELECT fact_id, fact_text
FROM reporting.question_facts qf
WHERE question_id = :question_id
  AND version = (
      SELECT MAX(version) FROM reporting.question_facts qf2
      WHERE qf2.fact_id = qf.fact_id AND qf2.question_id = qf.question_id
  )
  AND status <> 'flagged'
ORDER BY display_order, fact_id
"""


def load_question_facts(db: EvalDB, question_id: str) -> list[dict[str, Any]] | None:
    """Load latest-non-flagged facts for a question from reporting.question_facts.

    Returns a list of ``{"fact_id": ..., "fact_text": ...}`` dicts ordered by
    ``display_order``, or ``None`` when the reporting schema/table is absent or the
    question has no facts. Never raises on a missing schema — that is the local-dev
    signal to fall back to the YAML battery.
    """
    dialect = db._engine.dialect.name  # noqa: SLF001  # eval-internal DB access
    sql = _LATEST_NON_FLAGGED_SQL_SQLITE if dialect == "sqlite" else _LATEST_NON_FLAGGED_SQL
    try:
        with db._session_factory() as session:  # noqa: SLF001  # eval-internal DB access
            rows = session.execute(sa_text(sql), {"question_id": question_id}).fetchall()
    except SQLAlchemyError as e:
        # reporting schema/table absent (local dev) or read denied — fall back.
        logger.debug(f"reporting.question_facts unavailable for {question_id!r}: {e}")
        return None

    if not rows:
        return None
    return [{"fact_id": r.fact_id, "fact_text": r.fact_text} for r in rows]


def resolve_required_facts(
    db: EvalDB,
    question_id: str,
    yaml_facts: list[Any] | None,
) -> list[Any] | None:
    """Prefer stable-id facts from reporting.question_facts; fall back to YAML.

    When the reporting table has facts for ``question_id``, those (carrying stable
    ``fact_id``s) are used. Otherwise the YAML battery's ``required_facts`` (the
    current behavior, positional F{n} ids) is returned unchanged.
    """
    db_facts = load_question_facts(db, question_id)
    if db_facts:
        return db_facts
    return yaml_facts
