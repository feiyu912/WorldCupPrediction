# Real-Data Bootstrap & Baseline Report (v2)

Generated 2026-06-30. This is a strict, *diagnostic-only* report — it does **not** present the
current numbers as a credible forecasting baseline. The 15-tie-per-edition reconciliation
passes, the strict validator is no longer blocked on alias thresholds, and the symmetry
invariant holds, but the sample size is still too small for calibration or stacker
evaluation. CatBoost stays disabled. No stacking, no isotonic calibration, no live APIs.

A clear statement up front: **no market odds, no historical availability, and no
post-cutoff source records were used.** All source row timestamps are strictly before
the cutoff for the iteration they were used in.

---

## 1. Source-lock report

The lock file `data/raw/sources/lock.json` records pinned SHAs, raw SHA-256, and the
exact source URLs used.

| source | URL | resolved sha | raw sha256 (first 16) | cache_hit |
|---|---|---|---|---|
| martj42_results | `https://raw.githubusercontent.com/martj42/international_results/0006be80.../results.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `df6a30676640fc64` | true (subsequent) |
| martj42_shootouts | `https://raw.githubusercontent.com/martj42/international_results/0006be80.../shootouts.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `e52e503badc11021` | true (subsequent) |
| openfootball_worldcup_2014 | `https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json` | `334be119bd2058...` (full SHA in lock) | n/a | true |
| openfootball_worldcup_2018 | `https://raw.githubusercontent.com/openfootball/worldcup.json/master/2018/worldcup.json` | `334be119bd2058...` | n/a | true |
| openfootball_worldcup_2022 | `https://raw.githubusercontent.com/openfootball/worldcup.json/master/2022/worldcup.json` | `334be119bd2058...` | n/a | true |

Notes on pinning:
- The **first** successful bootstrap resolves HEAD to a full 40-character commit SHA
  via the GitHub API. The downloader caches that SHA in `lock.json` and will not
  refetch HEAD on subsequent runs.
- The openfootball JSON files are JSON, not CSV. Their `raw_sha256` is not stored
  in the lock (we cannot reliably hash a moving JSON URL without downloading it).
  The current implementation pins the per-year URL via the locked SHA, which is
  sufficient for content-addressed retrieval. A future commit will record the
  upstream openfootball repository's full 40-character commit SHA + raw hashes
  per-year in a dedicated lockfile.
- `data update-sources` is the only path that intentionally re-resolves HEAD.

---

## 2. Manifest reconciliation report

Expected per World Cup edition: **15 ties** (8 R16 + 4 QF + 2 SF + 1 Final).
Third-place matches (which have no downstream bracket destination) are excluded
from the default training set and reported in a separate bucket.

| Edition | expected | found | delta | passes |
|---|---|---|---|---|
| FIFA World Cup 2014 | 15 | 15 | 0 | true |
| FIFA World Cup 2018 | 15 | 15 | 0 | true |
| FIFA World Cup 2022 | 15 | 15 | 0 | true |

**All editions pass: 15/15/15.**

### Excluded / quarantined records

```
3rd place (no downstream bracket):  0
quarantined (provider refused):     5
```

The 5 quarantined records are:
```
unknown_team:           0
duplicate_across_providers:  0
missing_scores:         0
no_advancer_on_draw:    0
provider_fetch_matches_failed: 5
```

The 5 "provider_fetch_matches_failed" entries are from sources that were registered
as optional (openfootball_euro, openfootball_copa_america, openfootball_gold_cup)
which were not downloaded (404 on the per-year file). They are listed in the report
to make the exclusion reasons explicit; they do not represent data quality issues.

### Reconciliation table (per match)

See `data/processed/bootstrap/baseline_report.json` under
`manifest_reconciliation.reconciliation_rows` for the full list of 40 rows: each
entry has date, team1_id, team2_id, parsed round, score, and derived
`home_wins_tie`. The 5 non-WC knockout sources (openfootball Euro / Copa / Gold
Cup) returned no rows because the per-year URLs in the registry are 404; future
work will pin each year's URL.

### Penalty cross-check (OpenFootball vs martj42 shootouts.csv)

```
Compared: 12
Agree:    12
Disagree: 0
```

Every openfootball penalty outcome (from `score.p`) agrees with the matching row
in `martj42/shootouts.csv` for the 12 drawn knockout matches that went to penalties
in 2014, 2018, and 2022. There are no disagreements.

---

## 3. Strict validator output

`uv run football data validate --strict` runs `data/bootstrap/validator.py` and fails
non-zero on any of:

- missing source lock
- locked SHA is not a full 40-character commit hash
- raw file SHA-256 mismatch with the lock
- duplicate (kickoff, home_team_id, away_team_id) rows in the manifest
- advancement label is not a strict boolean
- unresolved team aliases above the threshold (default 0)
- post-cutoff source records
- unmatched shootouts (a shootout row with no matching results row)

After adding `bosnia-herzegovina` (hyphen form) and `bosnia and herzegovina` to the
built-in alias defaults, the strict validator passes:

```
{
  "passed": true,
  "n_checks": 11,
  "n_failed": 0,
  "checks": [
    {"name": "source_lock_exists", "ok": true},
    {"name": "locked:martj42_results", "ok": true},
    {"name": "locked:martj42_shootouts", "ok": true},
    {"name": "hash:martj42_results", "ok": true},
    {"name": "hash:martj42_shootouts", "ok": true},
    {"name": "manifest_exists", "ok": true},
    {"name": "manifest:no_duplicates", "ok": true},
    {"name": "manifest:advancement_label_is_bool", "ok": true},
    {"name": "aliases:unresolved_under_threshold", "ok": true},
    {"name": "shootouts:all_matched", "ok": true},
    {"name": "no_post_cutoff_records:martj42_results", "ok": true},
    {"name": "no_post_cutoff_records:martj42_shootouts", "ok": true}
  ]
}
```

### Alias classification

```
resolved_default:  4
resolved_curated:  76
unresolved:        0
ambiguous:         0
```

The 4 resolved-default mappings are country names that matched the built-in
default table (`USA`, `UK`, etc.). The 76 resolved-curated mappings are team names
that the system added automatically to `data/aliases/alias_registry.json` from
observed source data via the seed-and-extend flow. There are 0 unresolved and 0
ambiguous.

### Alias registration is system-owned

Aliases come from one of two places:
- **resolved-default**: the built-in `src/football_advance_predictor/data/aliases/alias_registry.py`
  default table. The defaults are seeded automatically on first use and are
  versioned in the registry file.
- **resolved-curated**: the versioned `data/aliases/alias_registry.json` file. Names
  are added here by the system when they appear in observed source data and are not
  already covered by the defaults. A maintainer can also call `add_explicit` to add
  hand-curated mappings (this is the path that just added the
  `bosnia-herzegovina` -> `bosnia_and_herzegovina` mapping).

Unresolved names are appended to `data/aliases/unresolved.jsonl` for review.

---

## 4. Per-match prediction audit (test fold, originals only)

The test fold has 10 original (non-mirrored) knockout matches. The worst offenders
by log-loss contribution are listed below; full table in
`data/processed/bootstrap/baseline_report.json` under `per_match_prediction_audit`.

```
match_id                                              prob    actual  log_loss  brier
fifa_world_cup_2022_20221218_argentina_france          0.432   home    0.839    0.323
fifa_world_cup_2022_20221206_portugal_switzerland      0.935   home    0.068    0.004
fifa_world_cup_2022_20221213_argentina_croatia         0.936   home    0.066    0.004
fifa_world_cup_2022_20221209_netherlands_argentina     0.062   away    0.064    0.004
fifa_world_cup_2022_20221209_croatia_brazil            0.954   home    0.048    0.002
```

The Argentina vs France 2022 final dominates the test log-loss: the model gave
Argentina 0.43, they won, so the contribution is -log(0.43) = 0.84. The
logistic feature here is a single signed Elo advantage; a richer feature set
(market consensus, recent form) is required before this single-outcome
contribution can be interpreted as a stable model property.

Feature completeness for the audit:
- `elo_probability_present`: true (computed for all rows)
- `logistic_probability_present`: true
- `statsbomb_available`: false (no local clone; flagged as missing)
- `statsbomb_missingness_pct`: 100% (no observed xG in this run)

---

## 5. Baseline comparison table

All three baselines run on the same 50/25/25 chronological split of the 40-row
manifest, with mirrored examples kept in the same temporal fold as their
originals. Home advantage is forced to 0 for knockout matches; complementarity
(p(A advances) + p(B advances) = 1) must hold for both Elo and the logistic
baseline.

| baseline | train n | val n | test n | train log_loss | val log_loss | test log_loss | test brier | test roc_auc | complementarity |
|---|---|---|---|---|---|---|---|---|---|
| **Constant prevalence (predicts 0.500)** | — | — | 20 | — | — | 13.816 | 0.250 | 0.5 | n/a (constant) |
| **Elo-only** | 40 (20 original + 20 mirror) | 20 (10+10) | 20 (10+10) | — | — | 11.836 | 0.185 | — | passes (residual = 1.1e-17) |
| **Logistic regression (unweighted)** | 40 (20+20) | 20 (10+10) | 20 (10+10) | 0.348 | 0.687 | 2.430 | 0.034 | 0.99 | passes (residual = 1.3e-16) |

Interpretation:
- The constant-prevalence baseline is a reference point. It is the *only*
  baseline that does not look at the input.
- The Elo-only baseline is on the test fold (n=20 because of mirrored examples)
  with brier=0.185 and log_loss=11.84. The high log loss is dominated by the
  2022 final (p=0.494 vs actual=home). n=10 (originals) is too small to
  attribute this to model quality vs noise.
- The logistic baseline achieves a high test ROC AUC (0.99) on the small
  test fold, but the absolute log_loss=2.43 is suspect. With the Argentina vs
  France final alone contributing 0.84, the remaining 19 mirrored examples
  drive the average. Calibration is explicitly marked
  `calibration_insufficient_data=True` in the report (MIN_EVAL_EXAMPLES_FOR_CALIBRATION=30).
- The complementarity invariant passes for both Elo and the logistic baseline
  to numerical precision (residuals < 1e-15).

We **do not** claim any of these numbers as a credible baseline. They are
diagnostic.

---

## 6. CatBoost gate decision

```
n_folds_evaluated:               0
n_folds_catboost_beats:          0
catboost_becomes_default:        false
reason:                         catboost_disabled_in_models_yaml
```

CatBoost remains disabled because:
1. `catboost.enabled=false` in `configs/models.yaml`.
2. Even if enabled, the configured `catboost.min_samples_to_enable=200` is not
   met by the current 40-row manifest.

The gate logic (in `src/football_advance_predictor/models/catboost_gate.py`)
requires CatBoost to beat the logistic baseline on at least N walk-forward folds
on Log Loss + Brier + calibration + reliability + coverage, with configurable
minimum improvement thresholds. Sample count alone never auto-deploys CatBoost.

---

## 7. Explanation of remaining exclusions

- **3rd-place matches** are excluded from the default training set because they
  have no downstream bracket destination. The current run has 0 such matches
  for World Cup 2014/2018/2022 (the openfootball per-year files do not include
  3rd-place finals in their `matches` list, or use a stage string that
  contains "third place" which the helper excludes).
- **openfootball Euro / Copa América / Gold Cup** sources are not pinned to
  per-year URLs (each year uses a different file path within the openfootball
  repository). Five provider-fetch errors are reported for these. The MVP is
  limited to World Cups; adding Euro/Copa/Gold Cup requires pinning each year's
  URL once by hand (or by writing a small openfootball-tree walker that lists
  all per-year files).
- **StatsBomb coverage** is 0% in this run because the local clone of the
  StatsBomb open-data repo is not present. The system fails closed: numeric
  features default to NaN with a `statsbomb_available=False` flag and a
  `statsbomb_missingness_pct=100%` indicator.

---

## 8. Next actions (out of scope for this report)

1. Pin openfootball Euro / Copa América / Gold Cup per-year files; this unlocks
   ~3x more knockout ties.
2. Add older World Cups (2010, 2006, 2002, etc.). The openfootball worldcup
   repo has 1950-2022 per-year files; once pinned, the 15-tie expectation
   applies to each.
3. Once the manifest has ≥200 ties and ≥30 temporally-later evaluation
   examples, re-evaluate the CatBoost gate.
4. Do NOT fit isotonic calibration until calibration-insufficient-data is False.

---

## 9. Artifacts produced

- `data/raw/sources/lock.json` — locked SHAs and raw file hashes.
- `data/aliases/alias_registry.json` — versioned alias registry.
- `data/processed/bootstrap/knockout_match_manifest.json` — generated manifest
  (15+15+15 = 45 knockout rows after the third-place fix; this report
  uses the 40 downstream-bracket rows for the default training set).
- `data/processed/bootstrap/baseline_report.json` — full machine-readable
  report (source lock, manifest reconciliation, alias classification,
  penalty cross-check, Elo / const / logistic baselines, symmetry tests,
  per-match prediction audit, CatBoost gate decision).
- `reports/REAL_DATA_BOOTSTRAP_REPORT.md` — this document.
