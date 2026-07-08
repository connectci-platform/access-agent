-- Eval rubric v2 migration (spec §3.2). Production Postgres only.
-- SQLite test schema is built fresh via models.py + create_all and never runs this.
--
-- APPROACH: archive-and-recreate, NOT alter-in-place.
--
-- Prod eval_scores holds pre-v2 history on the OLD 1-5 rubric (~400 rows / 13 runs
-- as of 2026-07-07), scored on a different scale (correctness/relevance/... 1-5,
-- and a `completeness` dimension v2 dropped). v2 scores on 0-2 ordinals with a new
-- `specificity` axis. Mixing the two in one table would corrupt aggregation: a
-- per-dimension mean over old 1-5 rows and new 0-2 rows is meaningless, and nothing
-- in the code filters by rubric_version. Rather than alter-in-place + rely on a
-- segregation filter that doesn't exist, we RENAME the old tables aside and let the
-- app rebuild fresh v2 tables. This is:
--   - non-destructive / reversible (rename, not drop — no data lost; the old rows
--     stay queryable in *_v1_archive),
--   - free of scale-mixing (the live tables are v2-only, so no rubric_version filter
--     is needed),
--   - clean: create_all in EvalDB.__init__ (src/eval/db.py) builds the fresh v2
--     eval_scores + eval_runs from models.py on the next eval run.
--
-- eval_runs is renamed too even though its schema is unchanged: old runs carry a
-- 1-5-scale composite_score, so archiving both keeps history together and avoids
-- run-level scale-mixing in reports.
--
-- ONE-TIME, applied manually at the v2 merge (like the dashboard reporting-schema
-- bootstrap): connect to the prod DB and run this once, then verify. Not wired into
-- the deploy workflow — it's a deliberate reversible cutover touching prod history.
--
-- AFTER this runs, the fresh v2 tables appear on the next EvalDB init (next eval
-- run), created from models.py. Nothing reads eval_scores between the rename and the
-- first v2 run, so there is no gap where the app breaks.

BEGIN;

ALTER TABLE eval_scores RENAME TO eval_scores_v1_archive;
ALTER TABLE eval_runs   RENAME TO eval_runs_v1_archive;

COMMIT;

-- Rollback (if ever needed): the reverse rename restores the pre-v2 state exactly.
--   BEGIN;
--   DROP TABLE IF EXISTS eval_scores;  -- the fresh v2 table, if create_all already ran
--   DROP TABLE IF EXISTS eval_runs;
--   ALTER TABLE eval_scores_v1_archive RENAME TO eval_scores;
--   ALTER TABLE eval_runs_v1_archive   RENAME TO eval_runs;
--   COMMIT;
