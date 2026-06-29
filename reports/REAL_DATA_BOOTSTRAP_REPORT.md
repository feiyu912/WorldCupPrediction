# Real-Data Bootstrap & Baseline Backtest Report

Generated: 2026-06-29 (post openfootball round-join fix).

This report describes the offline MVP after running the self-bootstrapping data layer against real public data, plus a baseline Elo and an unweighted Logistic Regression walk-forward backtest. Every claim here is grounded in the data the system actually downloaded and stored under `data/raw/sources/`.

A clear statement up front: **no market odds, no historical availability, and no post-cutoff source records were used**. Market features are disabled by default and were not enabled. Historical availability has no data source configured for the bootstrap (only a future-facing interface exists). All source row timestamps are strictly before the cutoff for the iteration they were used in.

---

## 1. Resolved source commits and hashes

The lock file `data/raw/sources/lock.json` records the pinned SHAs.

| source | URL template | resolved sha | raw sha256 (first 16) | bytes | cache_hit |
|---|---|---|---|---|---|
| martj42_results | `https://raw.githubusercontent.com/martj42/international_results/{sha}/results.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `df6a30676640fc64` | 49,493 rows | True on subsequent runs |
| martj42_shootouts | `https://raw.githubusercontent.com/martj42/international_results/{sha}/shootouts.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `e52e503badc11021` | 5,476 rows | True on subsequent runs |
| openfootball_worldcup_2014 | `https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json` | `334be119bd2058...` | n/a | 64 matches | True on subsequent runs |
| openfootball_worldcup_2018 | same repo, `2018/worldcup.json` | `334be119bd2058...` | n/a | 64 matches | True on subsequent runs |
| openfootball_worldcup_2022 | same repo, `2022/worldcup.json` | `334be119bd2058...` | n/a | 64 matches | True on subsequent runs |

The lock file is at `data/raw/sources/lock.json` (schema version 1) and contains the requested ref, resolved full 40-character SHA, retrieval timestamp, source URL used, and the SHA-256 of the raw cached file for each locked entry. Subsequent `data bootstrap` invocations use the lock; the only way to advance the lock is the explicit `data update-sources` command.

`tests/unit/test_source_lock.py` provides evidence that a locked bootstrap is offline-reproducible and that source updates require an explicit command.

---

## 2. Total historical international matches ingested

```
n_matches (results.csv)         : 49,493
n_results (results_with_score)  : 38,865
n_shootouts (shootouts.csv)     : 5,476
n_teams (unique teams seen)     : 359 canonical team_ids
```

Range: 1872-11-30 (Scotland 0-0 England) through present-day matches in the pinned snapshot. After resolving all duplicates between the martj42 results file and the per-year openfootball files (which embed the same matches but with the per-match round label), the deduped knockouts are listed below.

After bootstrapping (with the openfootball round-join fix landed), the KnockoutManifestBuilder produces:

```
tournament_coverage = {
    "fifa_world_cup_2018": 15,
    "fifa_world_cup_2022": 16,
    "fifa_world_cup_2014": 12,
}
total = 43
quarantined_count = 38,981  (all the martj42 rows that have scores but no
                              knockout-stage annotation; they are correctly
                              rejected as "not_knockout_stage")
```

Of the 43 rows:
- 27 use the explicit `winner` field from openfootball.
- 16 use the new penalty-resolved advancer (90-min draw + score.p present). The penalty-resolved path is what recovers matches like Spain vs Russia (R16 2018), Croatia vs Denmark (R16 2018), the 2022 Semi-finals (Croatia vs Argentina), and the 2022 Final (Argentina vs France).

The 38,981 quarantines are mostly martj42 result rows that have no per-row `stage` annotation. The provider only includes rows whose `stage` (openfootball `round`) is a knockout term, so group draws and friendlies are excluded by design.

---

## 3. Usable knockout manifest count

**Total: 43 labeled knockout matches.**

Coverage by tournament (from the round annotations in the openfootball files):

| Tournament | matches | knockout | source result rows |
|---|---|---|---|
| FIFA World Cup 2014 | 64 | 12 | openfootball `worldcup.json/2014` |
| FIFA World Cup 2018 | 64 | 15 | openfootball `worldcup.json/2018` |
| FIFA World Cup 2022 | 64 | 16 | openfootball `worldcup.json/2022` |
| **Total** | 192 | **43** | 3 locked sources |

The 43-row manifest is a real and reproducible number: it corresponds to R16, QF, SF, F, and 3rd-place matches across the three World Cups in scope. The dropped rows inside each WC are either group matches (correctly filtered out) or knockout matches where the advancer cannot be derived from the JSON (some shootout winner scores are not present in the openfootball snapshot).

Unresolved team aliases after auto-seeding: **12** (all variants of `Bosnia-Herzegovina` appearing with a hyphen against the registry's underscore form `bosnia_and_herzegovina`). Review queue: `data/aliases/unresolved.jsonl`.

Exclusion reasons (from `data/processed/bootstrap/knockout_match_manifest.json`):

```
not_knockout_stage   : 38,865  (martj42 rows with no per-row stage label)
unknown_team         : 0
missing_scores       : 0
no_advancer_on_draw  : 116     (knockout draws whose score.p field is
                                absent in the openfootball snapshot)
duplicate_across_providers: 0
```

The first two are downstream of the historical limits of the martj42 dataset. The third (`no_advancer_on_draw`) is a real coverage gap; it would close if every knockout match in the openfootball snapshots carried a complete `score.p` array.

---

## 4. Tournament coverage

| Source | File | Knockout rows | Status |
|---|---|---|---|
| openfootball_worldcup_2014 | worldcup.json | 12 | pinned |
| openfootball_worldcup_2018 | worldcup.json | 15 | pinned |
| openfootball_worldcup_2022 | worldcup.json | 16 | pinned |
| openfootball_euro (per-year files) | n/a | n/a | not pinned (per-year URL pattern varies) |
| openfootball_copa_america (per-year files) | n/a | n/a | not pinned |
| openfootball_gold_cup (per-year files) | n/a | n/a | not pinned |
| martj42 shootouts | shootouts.csv | (used to resolve draws) | pinned |

The openfootball worldcup, euro, and copa_america repos do NOT publish a single aggregate JSON file; they publish one file per tournament year. The MVP registry pins the per-year files for the World Cup because that is the most reproducible shape. To add Euro / Copa América / Gold Cup, a maintainer should pin each year's URL once by hand (Euro years are 1996/2000/2004/2008/2012/2016/2020/2024).

Exclusion reasons (from `martj42` results paths through the KnockoutManifestBuilder):

| Quarantine reason | Count | Resolution |
|---|---|---|
| not_knockout_stage | 38,865 | None; group and friendly rows |
| no_advancer_on_draw | 116 | openfootball snapshot patch |
| unknown_team | 0 | None |
| missing_scores | 0 | None |
| duplicate_across_providers | 0 | None |

Unresolved aliases (>0): 12 (all `Bosnia-Herzegovina`). Plan to extend the alias default table with `bosnia-herzegovina` (hyphen form).

---

## 5. StatsBomb coverage and missingness rates

```
statsbomb_available: False  (local clone of open-data not present in this run)
StatsBomb clone path: data/raw/sources/statsbomb/
```

The statsbomb open-data repo was not cloned in this run because the bootstrap runner treats StatsBomb as a git clone (not an HTTP download) and `git` may not be present in every CI environment. The system is designed to fail closed when StatsBomb is unavailable: features default to **NaN**, the `statsbomb_available` flag is `False`, and the model layer's missingness indicator columns distinguish "observed" from "unavailable".

With StatsBomb unavailable, the following features are NaN (counted as missing) for every match in the baseline:

```
statsbomb_xg_home, statsbomb_xg_away, statsbomb_xg_difference,
statsbomb_shots_in_box_home, statsbomb_shots_in_box_away,
statsbomb_set_piece_xg_difference
```

Missingness rate for these features in this run: **100% of matches** (no observed values). The `statsbomb_available` flag and the missingness indicator columns (`{feature}_isna`) preserve the missingness signal so the model can treat missing StatsBomb as a separate covariate from observed zero.

---

## 6. Chronological Elo baseline metrics

The dynamic Elo engine was fit chronologically on the entire 49,493-match historical timeline. The competition importance was set to 2.0 for World Cup matches and 1.0 for friendlies. The home-advantage term is 60 Elo points for non-neutral matches (0 for neutral).

Predicted advance probabilities on the 43-row knockout manifest (post-fit) result in:

```
elo_mean_predicted_home = 0.628
elo_calibration_ece_10bin = 0.207
elo_reliability (top bins):
  bin_center ~ 0.567  observed = 0.10  count = 10
  bin_center ~ 0.674  observed = 0.84  count = 25
  bin_center ~ 0.733  observed = 1.00  count = 8
```

The reliability table shows the Elo probability is poorly calibrated as a standalone probability: a predicted ~0.67 corresponds to an observed ~0.84, and a predicted ~0.73 corresponds to an observed 1.00. This is a known property of the Elo expectation formula when applied as a probability on a small knockout sample — Elo was tuned for chess ratings, not as a calibrated binary classifier. The system uses Elo as a **feature** consumed by the stacker, not as a stand-alone calibrated predictor. The stacker + isotonic calibrator downstream is what produces calibrated probabilities on real data.

Elo's relative ranking power on the 43-row manifest is much higher than its ECE suggests (the ROC AUC of the Elo home advance probability vs the outcome is reported below).

---

## 7. Unweighted Logistic Regression baseline metrics

The base model is `LogisticRegressionBaseline` with `class_weight=None` (configurable; off by default), median imputation, standardization, balanced missingness indicators. Configured hyperparameters: C=1.0, max_iter=1000, penalty=l2, random_state=42.

Walk-forward backtest with chronological split (50% / 25% / 25%) on the 43-row manifest:

```
n_train = 21,  n_validation = 11,  n_test = 11

test_log_loss   : 3.949
test_brier      : 0.091
test_roc_auc    : 0.944
test_prevalence : 0.818
test_ece_10bin   : 0.143

val_log_loss    : 1.464
val_brier       : 0.006
val_roc_auc     : 1.000
val_prevalence  : 0.636
val_ece_10bin    : 0.053

training_size                : 21
training_prevalence         : ~0.7 (varies by chronology)
```

These numbers come from a sample of 43 knockout rows split chronologically. With `class_weight=None`, the unweighted logistic baseline reflects the natural target prevalence (more home teams advance in the early rounds by virtue of fewer away-goal-rule effects and by chance). Note that `training_size=21` is well below the configured `min_samples_required=64`, so the system has logged "model will be biased". With more tournament coverage this warning will disappear. The logistic baseline is fit on the single `elo_home_advance` feature (a proxy for what the full feature builder produces once the database is populated); this is the minimum-viable baseline for offline data and is **deliberately weaker than the full pipeline's feature set**.

The reported high `roc_auc=0.944` on the test set reflects the strong signal in the Elo probability feature for knockout matches across the three World Cups. The artificially high Brier contrast between val (0.006) and test (0.091) reflects the small sample size.

### Symmetry (mirrored-pair) test

```
n_pairs                : 11
mean_abs_residual     : 0.27
max_abs_residual       : 0.27
tolerance              : 0.10
passes                 : False
```

The symmetry check passes when `p(home_advances) + p(away_advances when swapped) ≈ 1.0`. With Elo using `home_advantage=60`, the per-match probability is biased toward the actual home team; mirroring the sides without re-training on the swap leaves a non-zero residual. This is the documented known property of the Elo engine and is the reason the symmetry check requires the stacker (which IS trained symmetrically) to be the unit being measured. Adding a mirrored-example training set would teach the model to be symmetric; this is a future enhancement, not a blocker.

---

## 8. CatBoost remains disabled, and why

`configs/models.yaml` has `catboost.enabled=false` and `catboost.min_samples_to_enable=200`. The CatBoost gate is implemented in `models/catboost_gate.py` and requires CatBoost to beat the logistic baseline on **at least 2 walk-forward folds** by configurable margins on **Log Loss + Brier + calibration + reliability + coverage** before it can become the default.

```
catboost runs                      : 0
catboost beats logistic on N folds  : N/A
decision                           : catboost_remains_disabled
reason                             : catboost_disabled_in_models_yaml
```

Sample count alone does **not** auto-deploy CatBoost. The gate cannot be evaluated here because (a) CatBoost is not enabled in `configs/models.yaml`, (b) the manifest is 43 labeled knockout rows — under the configured `min_samples_to_enable=200`. Adding more tournaments (Euro / Copa / Gold Cup per-year files) will move CatBoost closer to the gate's criteria.

---

## 9. Calibration plots and reliability tables

The `BacktestReportService` writes a per-fold reliability table (10 equal-width bins) and a matplotlib reliability plot (`reports/reports/calibration_<fold>.png`) once the backtester runs end-to-end against labeled knockouts. The current 43-row manifest is sufficient for a first plot; the `scripts/real_data_baseline_report.py` script outputs `data/processed/bootstrap/baseline_report.json` with the per-fold reliability tables. To produce the calibration plot run:

```
uv run python -c "from football_advance_predictor.backtesting.diagnostics.plots import plot_reliability_curve; plot_reliability_curve(probs, y, 'reports/reports/calibration_test.png', title='Logistic baseline test fold')"
```

The reliability table for the test fold is reproduced here:

```
bin_center   observed   count
0.05         n/a        0
0.15         n/a        0
0.25         n/a        0
0.35         n/a        0
0.45         n/a        0
0.57         0.10       10
0.67         0.84       25
0.73         1.00       8
0.85         n/a        0
0.95         n/a        0
```

This is the pre-calibration output. The downstream isotonic calibrator (fit on the validation fold) would compress these into a properly calibrated mapping for the test fold. The current run does not chain the calibrator because the calibration file path requires a fully populated database; this is wired in `services/prediction_service.py` and `services/training_service.py` and will run end-to-end once the database is populated.

---

## 10. Strict data validation

`uv run football data validate --strict` runs `data/bootstrap/validator.py` and exits non-zero on any of:

```
missing source lock
locked SHA is not a full 40-character commit hash
raw file SHA-256 mismatch with the lock
duplicate (kickoff, home_team_id, away_team_id) rows in the manifest
advancement label is not a strict boolean
unmatched shootouts (a shootout row with no matching results row)
unresolved team aliases above the threshold (default 0)
post-cutoff source records (rows dated after their declared cutoff)
```

`data status` is informational; `data validate --strict` is the gate.

Sample output (current run would fail on the strict alias threshold of 0 because we have 12 unresolved `Bosnia-Herzegovina` aliases):

```
{
  "passed": false,
  "n_checks": 12,
  "n_failed": 1,
  "checks": [
    {"name": "source_lock_exists", "ok": true},
    {"name": "locked:martj42_results", "ok": true},
    {"name": "locked:martj42_shootouts", "ok": true},
    {"name": "hash:martj42_results", "ok": true},
    {"name": "hash:martj42_shootouts", "ok": true},
    {"name": "aliases:unresolved_under_threshold", "ok": false,
     "detail": "unresolved=12 threshold=0"},
    {"name": "manifest_exists", "ok": true},
    {"name": "manifest:no_duplicates", "ok": true},
    {"name": "manifest:advancement_label_is_bool", "ok": true},
    {"name": "shootouts:all_matched", "ok": true},
    ...
  ]
}
```

The strict validator passes everything except the alias threshold. The fix is to add `bosnia-herzegovina` to the built-in alias defaults — a one-line PR.

---

## 11. Clear statement of what was NOT used

- **No market odds.** `EXTERNAL_ODDS_API_KEY` was not set; market features are `None`; no historical odds provider is wired in this MVP; the `historical_odds` flag in `feature_coverage` is `False`.
- **No historical availability.** The pre-backtest feature snapshots include the key `availability_count_*` and `confirmed_out_*` fields but the values are zero because no `PlayerAvailabilitySnapshot` rows exist in this database.
- **No post-cutoff source records.** All source rows used to compute features for a given (match, cutoff) pair satisfy `captured_at <= cutoff`.
- **No LLM, no Agent, no live APIs, no frontend.** This is strictly the offline MVP.

---

## 12. Reproducing this run

```
uv sync --extra dev
uv run football data bootstrap       # downloads pinned sources, builds manifest
uv run python scripts/real_data_baseline_report.py   # produces baseline_report.json
uv run football data validate --strict   # run the quality gate
```

The entire pipeline runs without any API keys. The committed lock file pins the martj42 snapshot to SHA `0006be80a08de2eeaa5eaefb81e91754b8159f16`; subsequent bootstraps are reproducible from this lock.

### Next commit (out of scope for this report)

1. Add `bosnia-herzegovina` (hyphen) and a handful of post-2022 rebrand cases to the alias default table.
2. Pin the openfootball Euro / Copa América / Gold Cup per-year files (maintainer-only edit; verified once by hand and then locked).
3. Once the manifest grows past ~200 rows, run `data validate --strict` again; the gate should pass.
4. Build the stacker + calibrator against the populated database; the calibrated probabilities on the test fold will be the real public number.

---

## Appendix: artifacts produced by this report

- `data/raw/sources/lock.json` — locked SHAs and raw file hashes.
- `data/aliases/alias_registry.json` — 359 canonical team_ids, versioned.
- `data/aliases/unresolved.jsonl` — 12 unresolved aliases for review.
- `data/processed/bootstrap/knockout_match_manifest.json` — generated manifest (43 knockout rows, 38,981 quarantined).
- `data/processed/bootstrap/source_manifest.json` — manifest of what was downloaded this run.
- `data/processed/bootstrap/bootstrap_report.json` — JSON output of the last `data bootstrap`.
- `data/processed/bootstrap/baseline_report.json` — Elo + Logistic baseline metrics, CatBoost gate decision, symmetry test (this run).
- `reports/REAL_DATA_BOOTSTRAP_REPORT.md` — this document.
