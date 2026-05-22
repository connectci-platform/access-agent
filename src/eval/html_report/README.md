# HTML Comparison Report

Deterministic, code-only generator for the raw-RAG vs. full-agent comparison
report. No AI/Claude is involved at render time — given the same Postgres
state, `python -m src.eval html` produces a byte-identical HTML file every
run.

> **Why this exists:** the first few versions of this report were hand-built
> via the `visual-explainer` skill. That made iterating on the design fast
> but made re-running the report a manual, non-reproducible step. This
> package separates the two: layout + prose live in files you edit; data
> lives in Postgres and flows in by query.

---

## Quickstart

```bash
# Report for the most-recent runs in Postgres (all four batteries)
python -m src.eval html -o /tmp/report.html

# Report for a specific day (UTC)
python -m src.eval html --date 2026-04-17 -o /tmp/report.html

# Only some batteries
python -m src.eval html \
  --date 2026-04-17 \
  --question-sets friendly_battery,combined_battery \
  -o /tmp/partial.html
```

The runner picks, for each `(system, question_set)` pair, the newest
`eval_runs` row (optionally filtered to a single day). If a system or
battery is missing for that day, it is silently skipped and a warning is
logged — you get a partial report rather than an error.

### Publishing to Netlify

The repo at `access-ci/published-reports/` is linked to the
`access-ci-reports.netlify.app` site. To publish:

```bash
cp /tmp/report.html /Users/josephbacal/Projects/sweet-and-fizzy/access-ci/published-reports/production-baseline-comparison-v2.html
cd /Users/josephbacal/Projects/sweet-and-fizzy/access-ci/published-reports
netlify deploy --prod --dir=.
```

The deploy command prints both a permanent `*.netlify.app` URL and an
immutable `<deploy-id>--*.netlify.app` URL.

---

## What you'll want to change, and where

| You want to change… | Edit |
|---|---|
| Battery descriptions ("what's in each battery, why it exists") | `notes.py` → `BATTERY_INFO` |
| The order batteries appear in | `notes.py` → `BATTERY_ORDER` |
| The Observations bullets at the bottom | `notes.py` → `OBSERVATIONS` |
| The subtitle under the main title | `notes.py` → `REPORT_SUBTITLE` |
| Layout, typography, colors | `template.html` (CSS in `<style>`, DOM below) |
| Which questions are a "win" / "tie" / "loss" | `builder.py` → `assemble_bundle` (threshold is ±0.01) |
| Date-range logic (e.g., span a week instead of one day) | `builder.py` → `pick_run_ids` |
| Add a new section (e.g., "top agent wins") | extend `assemble_bundle` to compute the data; add DOM + JS in `template.html` |

You should never need to edit `builder.py` for a cosmetic change, and you
should never need to edit `template.html` for a prose change.

### Ad-hoc title / subtitle / column labels (no code change)

For one-off comparisons that don't match an existing preset — e.g.
agent-vs-agent for a hardening or prompt change, where both runs are
tagged `system=agent_full` and the preset's subtitle is wrong — pass
these flags to `python -m src.eval html`:

| Flag | Overrides | Default |
|---|---|---|
| `--title "..."` | the `<h1>` heading | `"Production Baseline Comparison"` |
| `--subtitle "..."` | the subtitle line | preset's `report_subtitle` |
| `--label-a "..."` | column A (baseline) label | `label_for_system(baseline_id)` |
| `--label-b "..."` | column B (candidate) label | `label_for_system(candidate_id)` |

Example for an agent-vs-agent hardening comparison:

```bash
python -m src.eval html \
  --from-json comparisons/safety_hardening.json \
  --title "Safety Hardening Comparison" \
  --subtitle "ACCESS agent pre-hardening vs post-hardening" \
  --label-a "Pre-hardening" \
  --label-b "Post-hardening" \
  -o /tmp/report.html
```

When to use these vs. adding a preset:
- **Flags** for one-off comparisons whose narrative prose doesn't need to
  be reused. The flags only affect the header/labels; battery
  descriptions and observations still come from the active `--preset`.
- **A new preset in `notes.py`** when the same comparison shape will be
  re-run (different dates, different batteries) and deserves its own
  battery descriptions and observations text. Add a `PresetSpec` entry
  to `PRESETS` and extend the `--preset` `choices=` list in
  `__main__.py`.

---

## File map

```
html_report/
├── __init__.py      — package marker, no logic
├── builder.py       — data layer: SQL queries, aggregation, template stamping
├── notes.py         — human prose (battery descriptions, observations, subtitle)
├── template.html    — layout: HTML/CSS/JS with a single __BUNDLE_JSON__ marker
└── README.md        — this file
```

### Flow

```
pick_run_ids()         resolve which eval_runs rows to include
       ↓               (newest per system × question_set, optional date filter)
fetch_scores()         pull eval_scores for those runs (one SQL round trip)
       ↓               + _fetch_durations() raw-SQL helper for duration_ms
assemble_bundle()      fold into per-question pairs, per-battery averages
       ↓
render_html()          write the bundle into template.html at __BUNDLE_JSON__
       ↓
write output file
```

---

## Conventions

### Battery keys

Internally, batteries are identified by their normalized `question_set`
basename without `.json` — e.g., `friendly_battery`,
`combined_battery`. That's the key used in `BATTERY_INFO`,
`BATTERY_ORDER`, and `--question-sets`.

In the rendered HTML, a short form is used for data-attributes and URLs
(`friendly`, `combined`, etc.) — these are derived from the battery key
by stripping `_battery`.

### Bundle shape

The bundle injected into `template.html` always has these top-level keys:

- `generated_at` — ISO date the report was built
- `subtitle` — from `notes.REPORT_SUBTITLE`
- `battery_order` — short keys in display order
- `battery_labels` — `{short: "Display Name"}`
- `battery_info` — `{short: {name, count, what, why}}`
- `per_battery` — `{short: {n, wins, losses, ties, agent_comp, raw_comp, agent_dur_ms, raw_dur_ms}}`
- `all_pairs` — list of `{qid, battery, question, delta, raw_rag: {...}, agent_full: {...}}`
- `observations` — list of `{text}` from `notes.OBSERVATIONS`
- `run_ids` — per-battery footer strip data

If you add a new key, also add a default in the template JS
(`BUNDLE.new_key || []`) so the template keeps working if someone runs an
older builder.

### `duration_ms`

The eval_scores table has a `duration_ms` column that isn't declared on
the SQLAlchemy model (`src/eval/models.py`). The builder fetches it via
raw SQL in `_fetch_durations`. When the model catches up, delete the
raw-SQL helper and use `EvalScore.duration_ms` directly.

---

## Adding a new section

Two-step process. Example: say you want a "Top 5 biggest agent wins" panel
at the top of the report.

**1. Compute the data in `assemble_bundle`:**

```python
top_wins = sorted(all_pairs, key=lambda e: e["delta"], reverse=True)[:5]
# ...
return {
    ...,
    "top_wins": top_wins,
}
```

**2. Render it in `template.html`:**

```html
<div class="section-label">Top agent wins</div>
<div id="top-wins"></div>
```

```javascript
document.getElementById('top-wins').innerHTML = (BUNDLE.top_wins || [])
  .map(e => `<div>${esc(e.qid)}: +${e.delta.toFixed(2)}</div>`)
  .join('');
```

No builder logic change for the style; no template change for the data.
Separation holds.

---

## Testing

There is no test harness yet. Smoke test: run the command and spot-check
the output against an existing good report. For now:

```bash
python -m src.eval html --date 2026-04-17 -o /tmp/new.html
# compare against the last blessed output
diff <(python -m html.parser /tmp/new.html 2>&1 | head) \
     <(python -m html.parser /path/to/known-good.html 2>&1 | head)
```

Better long-term: a unit test that builds a small fixture DB and asserts
the bundle shape (not byte-equality — `generated_at` moves).

---

## Known caveats

- **`generated_at` changes every run.** Output is not literally
  byte-identical across runs; it's identical modulo the date string. If
  you want byte-equality for caching or diffing, make it the date of the
  newest `eval_runs` row instead of `datetime.now()`.
- **`metadata.system` must be set on `eval_runs`.** The runner filters
  to `system in ('raw_rag', 'agent_full')`. Runs written without that
  metadata key are invisible to this report.
- **Same-day ties go to the newer run.** If you ran the same battery
  against the same system twice on the same day, only the most recent
  one appears.
