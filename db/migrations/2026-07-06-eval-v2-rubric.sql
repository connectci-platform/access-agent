-- Eval rubric v2 migration (spec §3.2). Production Postgres only.
-- SQLite test schema is built fresh via models.py + create_all and never runs this.
BEGIN;

-- 1. Drop the retired dimension. No back-fill: old completeness values do NOT
--    map to v2 specificity (different axes) — this is a DROP + ADD, not a RENAME.
ALTER TABLE eval_scores DROP COLUMN completeness;

-- 2. Add the new specificity axis (nullable, no back-fill; pre-v2 rows have no value).
--    Ordinal: 2=Actionable, 1=Mixed, 0=Generic. N/A carried in specificity_na.
ALTER TABLE eval_scores ADD COLUMN specificity SMALLINT;
ALTER TABLE eval_scores ADD COLUMN specificity_na BOOLEAN NOT NULL DEFAULT FALSE;

-- 3. Answerability screen (exclusion filter, never averaged). NULL = pre-v2 rows.
ALTER TABLE eval_scores ADD COLUMN answerable BOOLEAN;

-- 4. Rubric-version segregation so v2 (0-2) rows are never averaged with old (1-5) rows.
ALTER TABLE eval_scores ADD COLUMN rubric_version SMALLINT;

-- 5. Enforce v2 ordinal ranges going forward. NOT VALID because old 1-5 rows would fail;
--    VALIDATE only after old-rubric rows are segregated by rubric_version.
ALTER TABLE eval_scores
  ADD CONSTRAINT ck_eval_scores_v2_ranges CHECK (
    (correctness      IS NULL OR correctness      BETWEEN 0 AND 2) AND
    (relevance        IS NULL OR relevance        BETWEEN 0 AND 2) AND
    (citation_quality IS NULL OR citation_quality BETWEEN 0 AND 2) AND
    (hedging          IS NULL OR hedging          BETWEEN 0 AND 1) AND
    (specificity      IS NULL OR specificity      BETWEEN 0 AND 2)
  ) NOT VALID;

COMMIT;
