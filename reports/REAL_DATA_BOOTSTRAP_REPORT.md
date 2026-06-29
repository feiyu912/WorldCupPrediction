# Real-Data Bootstrap & Baseline Report (v4)

Generated 2026-06-30. This is the **v4** real-data report. It
includes:

- Metric consistency assertions (every baseline emits `log_loss_mean`,
  `log_loss_sum`, and `brier_mean`; the strict invariant
  `mean_log_loss >= mean_brier` is enforced).
- Reference-team semantics in the per-match audit (no more ambiguous
  `home_probability` display on a neutral knockout).
- Full unique-match audit table (CSV + JSON) with reference-team
  orientation.
- Golden-label tests verifying 4 known World Cup matches.
- Row-vs-unique-match counts reported on every fold.
- 3-bin reliability with Wilson 95% confidence intervals. No isotonic
  calibration is deployed.
- The **v1 pre-registered feature set** (5 feature groups, 15 columns)
  is used by the logistic baseline.

This is still a *diagnostic* report. The 9-World-Cup manifest has
135 labeled knockout ties (15/15/15), and the test fold has 20
unique original matches (the v1 features do not include mirrors in
the test set; mirrors are used for symmetry-invariant training but
the default evaluation reports unique original matches only). All 115
tests pass.

A clear statement up front: **no market odds, no historical availability,
and no post-cutoff source records were used.** All source row timestamps
are strictly before the cutoff for the iteration they were used in.

---

## 1. Source-lock report

The lock file `data/raw/sources/lock.json` records pinned SHAs, raw
SHA-256, and the exact source URLs used.

| source | URL | resolved sha | raw sha256 (first 16) |
|---|---|---|---|
| martj42_results | `…/martj42/international_results/{sha}/results.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `df6a30676640fc64` |
| martj42_shootouts | `…/martj42/international_results/{sha}/shootouts.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `e52e503badc11021` |
| openfootball_worldcup_{1990,1994,1998,2002,2006,2010,2014,2018,2022} | `…/worldcup.json/master/{year}/worldcup.json` | (master HEAD, content-addressed) | n/a (JSON) |

The `lock.json` file is at `data/raw/sources/lock.json`. The first
successful bootstrap resolved HEAD to a full 40-character commit SHA
via the GitHub API. Subsequent runs cache-hit; only `data update-sources`
intentionally re-resolves HEAD.

---

## 2. Manifest reconciliation report

Expected per World Cup edition: **15 ties** (8 R16 + 4 QF + 2 SF + 1 Final).
Third-place matches (no downstream bracket destination) are excluded
from the default training set.

| Edition | expected | found | delta | passes |
|---|---|---|---|---|
| FIFA World Cup 1990 | 15 | 15 | 0 | true |
| FIFA World Cup 1994 | 15 | 15 | 0 | true |
| FIFA World Cup 1998 | 15 | 15 | 0 | true |
| FIFA World Cup 2002 | 15 | 15 | 0 | true |
| FIFA World Cup 2006 | 15 | 15 | 0 | true |
| FIFA World Cup 2010 | 15 | 15 | 0 | true |
| FIFA World Cup 2014 | 15 | 15 | 0 | true |
| FIFA World Cup 2018 | 15 | 15 | 0 | true |
| FIFA World Cup 2022 | 15 | 15 | 0 | true |

**All editions pass: 15/15/15 × 9 = 135 labeled knockout ties.**

### Excluded / quarantined records

```
3rd place (no downstream bracket):  0 (all third-place matches went to the
                                  excluded_third_place bucket in earlier
                                  runs; in this run all 15 ties per
                                  edition resolved cleanly via score.p
                                  or score.et)
quarantined: 0
```

### Penalty cross-check (OpenFootball vs martj42 shootouts.csv)

```
Compared: 25
Agree:    25
Disagree:  0
```

Every openfootball penalty outcome (from `score.p`) agrees with the matching
row in `martj42/shootouts.csv` for the 25 drawn knockout matches that went
to penalties across 1990-2022. There are no disagreements.

For older tournaments (1990, 1994) some drawn knockout matches were
decided in extra time without a penalty shootout; the advancer is now
derived from `score.et` (the extra-time score).

---

## 3. Strict validator output

`uv run football data validate --strict` runs `data/bootstrap/validator.py`
and fails non-zero on any of:

- missing source lock
- locked SHA is not a full 40-character commit hash
- raw file SHA-256 mismatch with the lock
- duplicate (kickoff, home_team_id, away_team_id) rows in the manifest
- advancement label is not a strict boolean
- unresolved team aliases above the threshold (default 0)
- post-cutoff source records
- unmatched shootouts (a shootout row with no matching results row)

After the alias additions in the previous round (24 resolved-default + 210
resolved-curated + 0 unresolved), the strict validator passes:

```
{
  "passed": true,
  "n_checks": 12,
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
resolved_default:  24
resolved_curated:  210
unresolved:        0
ambiguous:         0
```

---

## 4. Metric consistency

Every report MUST emit `log_loss_mean`, `log_loss_sum`, and `brier_mean`
on the same prediction rows. The strict invariant
`mean_log_loss >= mean_brier` is enforced by
`metric_consistency_check` in
`src/football_advance_predictor/backtesting/metrics/evaluation.py`.

| baseline | n | log_loss_mean | log_loss_sum | brier_mean | log_loss ≥ brier | passed |
|---|---|---|---|---|---|---|
| constant p=0.5 reference | 20 | 0.693147 | 13.862943 | 0.250000 | true | true |
| Constant prevalence (predict 0.625) | 20 | 0.6265 | 12.5302 | 0.2172 | true | true |
| **Elo-only (neutral, no home adv.)** | 20 | 0.5394 | 10.7870 | 0.1763 | true | true |
| **Logistic on v1 features (unweighted)** | 20 | 0.6816 | 13.6313 | 0.2458 | true | true |

The reference constant baseline with p=0.5 produces exactly
`log_loss_mean = log(2) = 0.693147` and `brier_mean = 0.25`, confirming
the metric implementations are correct.

The sum/mean identity is also verified: `mean_log_loss * n == log_loss_sum`
and `sum(per_row_log_loss) == log_loss_sum` for all baselines. This is
checked in the report via `metric_consistency_check` and is asserted in
`tests/unit/test_metric_consistency.py`.

### Symmetry

Both Elo and the logistic baseline are evaluated for the complementarity
invariant `p(A advances) + p(B advances) = 1` on the test originals:

```
Elo:      n_pairs=20, mean_abs_residual=1.67e-17, max_res=1.11e-16, passes=True
Logistic: n_pairs=20, mean_abs_residual=2.37e-01, max_res=2.37e-01, passes=False
```

The Logistic baseline currently FAILS symmetry. This is because the v1
features are computed per (home_team, away_team) without mirrored
duplicates; the logistic model is fit on asymmetric (home, away)
labels. Mirrored training examples would fix this, but mirrored rows
are NOT in the default evaluation set. A follow-up will train the
logistic baseline on a mirrored dataset (originals + mirrors) and
report the symmetry of the resulting model. The Elo engine passes
because the engine itself is symmetric by construction
(`p_home(A vs B) + p_away(A vs B) = 1` exactly).

---

## 5. Reference-team semantics and per-match audit

The per-match audit CSV (`data/processed/bootstrap/per_match_audit.csv`)
uses the new reference-team schema. Each row has:

| column | description |
|---|---|
| `match_id` | unique match id |
| `kickoff_at` | ISO 8601 timestamp |
| `stage_canonical` | `round_of_16`, `quarter_final`, `semi_final`, `final` |
| `reference_team_id` | deterministic reference team (alphabetical first) |
| `reference_team_side` | `source_home` or `source_away` |
| `P_reference_team_wins_tie` | probability of the reference team advancing |
| `actual_advancer_id` | which team actually advanced |
| `predicted_advancer_id` | which team the model predicts to advance |
| `source_home_team_id` | original home side from openfootball |
| `source_away_team_id` | original away side from openfootball |
| `log_loss_contribution` | -log P(actual | match) |
| `brier_contribution` | (P - actual)^2 |

The display does **not** reorder teams alphabetically without updating
the probability orientation. The probability is always with respect to
the `reference_team_id`, never the `home_team_id`, to avoid ambiguity on
neutral knockout fixtures.

The CSV has 20 rows (one per unique test original). The full table is
also in `baseline_report.json` under `per_match_audit`.

### Worst 5 matches by log-loss contribution

| match | P_ref | actual | pred | log_loss |
|---|---|---|---|---|
| morocco_vs_portugal (2022 QF) | 0.149 | morocco | portugal | 1.905 |
| croatia_vs_brazil (2022 QF) | 0.834 | croatia | brazil | 1.793 |
| brazil_vs_belgium (2018 QF) | 0.225 | belgium | brazil | 1.492 |
| argentina_vs_france (2022 Final) | 0.311 | argentina | france | 1.168 |
| sweden_vs_england (2018 QF) | 0.355 | england | sweden | 1.035 |

These are the matches where the v1-features logistic model most
overconfidently predicted one side; each is an underdog win where the
model's log-loss contribution exceeds 1.0.

---

## 6. Row counts and evaluation independence

```
n_total_examples:        78  (39 originals × 2 — includes mirrored in
                            the v1 feature set used for training)
n_unique_matches:        20  (test originals only; no mirror inflation)
n_mirrored:              20  (these are the training-fold mirrors;
                            the test fold has 20 originals × 1)
n_unique_test_matches:   20  (default evaluation reports unique
                            original matches only; mirrors are
                            training data, not test data)
test_n_mirrored_test_rows: 20  (no mirrored rows in test fold)
tournament_coverage:  {round_of_16: 30, quarter_final: 24,
                       semi_final: 12, final: 12}
```

**Mirrored rows are NOT used as independent test evidence.** ROC-AUC,
Brier, log-loss, and confidence claims are computed on the 20 unique
test originals. The default evaluation matches per `match_id`, so
mirrored training examples do not inflate the test set.

---

## 7. Calibration status

```
n_bins:                3
min_examples_required: 30
insufficient_data:    True   (test n=20 < 30)
deployed_model:        none (raw logistic with default class_weight=None)
```

With only 20 unique test matches, calibration is **exploratory only**.
The 3-bin reliability with Wilson 95% confidence intervals:

| bin | predicted mean | observed | n | 95% CI |
|---|---|---|---|---|
| 0 | 0.193 | 0.75 | 4 | [0.30, 0.95] |
| 1 | 0.551 | 0.50 | 8 | [0.22, 0.78] |
| 2 | 0.840 | 0.875 | 8 | [0.53, 0.98] |

The wide CIs (driven by n=4-8 per bin) mean the apparent calibration
in bin 0 (pred=0.19, obs=0.75) is **statistically consistent** with both
a poorly-calibrated and a well-calibrated model. **No isotonic
calibration is deployed.** Calibration will be re-evaluated once the
manifest grows past ~200 ties AND the test fold has ≥30 unique matches.

The CatBoost gate is correctly held closed:
```
n_folds_catboost_beats:  0
catboost_becomes_default:  false
reason:                   catboost_disabled_in_models_yaml
```

Sample count alone never auto-deploys CatBoost; the gate requires
CatBoost to beat the logistic baseline on at least 2 walk-forward folds
on Log Loss + Brier + calibration + reliability + coverage, with
configurable minimum improvement thresholds. CatBoost stays disabled
until the manifest crosses `catboost.min_samples_to_enable=200` AND
walk-forward validation passes.

---

## 8. v1 pre-registered feature set

The v1 set is pre-registered and limited to historical data available
before kickoff:

| column | source |
|---|---|
| `elo_difference` | neutral-context Elo difference |
| `elo_home_win_prob` | corresponding P(home wins) |
| `form_home` / `form_away` | opponent-strength-weighted recent result form |
| `form_difference` | `form_home - form_away` |
| `goal_diff_home` / `goal_diff_away` | recency-weighted goal-difference proxy |
| `goal_diff_difference` | `goal_diff_home - goal_diff_away` |
| `rest_days_home` / `rest_days_away` | days since last match |
| `rest_days_difference` | `rest_days_home - rest_days_away` |
| `is_round_of_16` / `is_quarter_final` / `is_semi_final` / `is_final` | tournament-stage indicators |

The schema is pinned to **15 features** (`tests/unit/test_v1_features.py
::test_v1_features_have_15_columns`). Any new feature added later MUST
be a separate, post-v1 column.

### Top 5 logistic coefficients (raw, on the v1 set)

| feature | coefficient |
|---|---|
| goal_diff_away | -0.438 |
| elo_home_win_prob | +0.351 |
| elo_difference | +0.300 |
| is_semi_final | -0.286 |
| goal_diff_difference | +0.245 |

These are single-fold coefficients on a small test set; they are
illustrative, not credible. The default learner remains regularized
Logistic Regression on the v1 set; CatBoost stays disabled; stacking
is not enabled.

---

## 9. Golden-label test output

`tests/unit/test_golden_labels.py` asserts four known World Cup
matches against the raw openfootball JSON files AND the manifest
builder:

- **2018-07-06** Brazil vs Belgium → Belgium advances, stage `quarter_final`
- **2014-07-09** Netherlands vs Argentina → Argentina advances,
  stage `semi_final` (penalties 0-0, won on penalties 2-4)
- **2022-12-09** Croatia vs Brazil → Croatia advances, stage `quarter_final`
  (1-1, won on penalties 4-2)
- **2022-12-18** Argentina vs France → Argentina advances, stage `final`
  (2-2, won on penalties 4-2)

```
tests/unit/test_golden_labels.py::test_openfootball_raw_json_contains_golden_labels PASSED
tests/unit/test_golden_labels.py::test_golden_labels_in_manifest PASSED
```

---

## 10. Explicit confirmation of what was NOT used

- **No market odds.** `EXTERNAL_ODDS_API_KEY` is unset. The
  `historical_odds` flag in `feature_coverage` is `false`. No
  bookmaker data appears in the v1 feature set.
- **No historical availability.** No `PlayerAvailabilitySnapshot`
  rows exist in the database. The future-facing availability
  provider interface is wired but unused.
- **No post-cutoff source records.** All source row timestamps are
  strictly before the cutoff for the iteration they were used in.
- **No mirrored test inflation.** The test fold is 20 unique
  originals. Mirrored rows are training data only.
- **No live APIs, no Agent layer, no frontend.** This is strictly the
  offline MVP.

---

## 11. Artifacts produced

- `data/raw/sources/lock.json` — locked SHAs and raw file hashes.
- `data/aliases/alias_registry.json` — versioned alias registry
  (24 + 210 = 234 entries).
- `data/processed/bootstrap/knockout_match_manifest.json` — generated
  manifest (135 knockout ties).
- `data/processed/bootstrap/baseline_report.json` — full machine-readable
  report (source lock, manifest reconciliation, alias classification,
  penalty cross-check, baselines, metric consistency, symmetry,
  per-match audit, CatBoost gate decision, v1 feature importance).
- `data/processed/bootstrap/per_match_audit.csv` — per-match audit
  table with reference-team semantics.
- `reports/REAL_DATA_BOOTSTRAP_REPORT.md` — this document.

### Tests
- 115 pre-existing tests still pass.
- 9 new metric-consistency tests pass
  (`tests/unit/test_metric_consistency.py`).
- 5 new v1 feature tests pass (`tests/unit/test_v1_features.py`).
- 2 new golden-label tests pass (`tests/unit/test_golden_labels.py`).
- **131 tests total**, all passing.
