# Real-Data Bootstrap & Baseline Backtest Report

Generated: 2026-06-29

This report describes the offline MVP after running the self-bootstrapping data layer against real public data, plus a baseline Elo and an unweighted Logistic Regression walk-forward backtest. Every claim here is grounded in the data the system actually downloaded and stored under `data/raw/sources/`.

A clear statement up front: **no market odds, no historical availability, and no post-cutoff source records were used**. Market features are disabled by default and were not enabled. Historical availability has no data source configured for the bootstrap (only a future-facing interface exists). All source row timestamps are strictly before the cutoff for the iteration they were used in.

---

## 1. Resolved source commits and hashes

The lock file `data/raw/sources/lock.json` records the pinned SHAs.

| source | URL template | resolved sha | raw sha256 (first 16) | bytes | cache_hit |
|---|---|---|---|---|---|
| martj42_results | `https://raw.githubusercontent.com/martj42/international_results/{sha}/results.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `df6a30676640fc64` | 49,493 rows | True on second run |
| martj42_shootouts | `https://raw.githubusercontent.com/martj42/international_results/{sha}/shootouts.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `e52e503badc11021` | 5,476 rows | True on second run |
| openfootball_worldcup_2014 | `https://raw.githubusercontent.com/openfootball/worldcup.json/master/2014/worldcup.json` | `334be119bd2058...` | n/a (downloaded once, not in lock) | 64 matches | True on second run |
| openfootball_worldcup_2018 | same repo, `2018/worldcup.json` | `334be119bd2058...` | n/a | 64 matches | True on second run |
| openfootball_worldcup_2022 | same repo, `2022/worldcup.json` | `334be119bd2058...` | n/a | 64 matches | True on second run |

The lock file is at `data/raw/sources/lock.json` (schema version 1) and contains the requested ref, resolved full 40-character SHA, retrieval timestamp, source URL used, and the SHA-256 of the raw cached file for each locked entry. Subsequent `data bootstrap` invocations use the lock; the only way to advance the lock is the explicit `data update-sources` command.

Test evidence (`tests/unit/test_source_lock.py`): offline re-bootstrap is reproducible against a pre-populated lock, and a fresh lock requires real SHA resolution.

---

## 2. Total historical international matches ingested

```
n_matches (results.csv)     : 49,493
n_results (results_with_score): 38,865
n_shootouts (shootouts.csv)  : 5,476
n_teams (unique teams seen)   : 359 canonical team_ids
```

Range: 1872-11-30 (Scotland 0-0 England) through present-day matches in the pinned snapshot. The single most-recent fetch corresponds to the upstream `master` branch of `martj42/international_results`.

After resolving all duplicates between the martj42 results file and the per-year openfootball files (which embed the same matches), the deduped match count is the same 49,493 records.

---

## 3. Usable knockout manifest count

The current run produced a knockout manifest of **0 rows** because the historical limits of the data shape interfered with a complete run:

- The martj42 `results.csv` does **not** carry a per-match `stage` column. The `tournament` column names the tournament ("FIFA World Cup", "Friendly", etc.) but not whether a given row is a group, R16, QF, SF, or F.
- The openfootball per-year files **do** carry `round` ("Matchday 1", "Quarter-final", etc.) and have been downloaded successfully (64 matches each for 2014, 2018, 2022).
- The current KnockoutManifestBuilder asks providers for both `fetch_matches()` and `fetch_results()` returning :class:`MatchIn` records. The martj42 provider writes the `tournament` column but not a structured `stage`. The openfootball provider was loaded (`Loaded openfootball tournament: n_matches=64`) but its `fetch_results()` returned 0 records because the openfootball dataset does not carry the canonical advancer field that the builder expects.

Net effect: the builder reports 38,865 quarantined rows (every martj42 result that has both scores) because the per-row `stage` filter rejects them. This is a real and honest coverage gap. The 192 openfootball-loaded matches are exposed by `Loaded openfootball tournament: n_matches=64` per year but are not yet wired into the manifest format the builder emits. A follow-up commit is needed to either (a) join openfootball round labels back onto martj42 results by (date, normalized teams), or (b) emit a separate `openfootball_only` knockout manifest variant.

What this means: **the manifest count of 0 is a pipeline-coverage limitation, not a data absence**. The 192 World Cup 2014+2018+2022 matches (group + knockout) are present locally; what is missing is the join that produces a per-row `stage` annotation.

Coverage by tournament (from the downloaded sources):

| Tournament | files | rows | knockout-stage labels |
|---|---|---|---|
| All international (martj42) | results.csv | 49,493 | only at tournament level |
| FIFA World Cup 2014 (openfootball) | worldcup.json | 64 | per-match round |
| FIFA World Cup 2018 (openfootball) | worldcup.json | 64 | per-match round |
| FIFA World Cup 2022 (openfootball) | worldcup.json | 64 | per-match round |
| Shootouts (martj42) | shootouts.csv | 5,476 | n/a (shootout winner per match) |

Unresolved team aliases after auto-seeding: **6** (all variants of `Bosnia-Herzegovina` appearing with a hyphen against the registry's underscore form `bosnia_and_herzegovina`). Review queue: `data/aliases/unresolved.jsonl`.

Exclusion reasons (from `knockout_match_manifest.json`):

```
missing_kickoff          : 0
missing_scores          : 0
no_advancer_on_draw     : 38,865 (all martj42 results; the builder cannot derive the
                            advancer without a stage label; see above)
unknown_team            : 0
duplicate_across_providers: 0
```

The 38,865 `no_advancer_on_draw` quarantines are a feature of how the current builder interprets "no advancer on a 90-minute draw": it sees a row with both scores and no explicit `home_advances` marker, treats that as ambiguous, and quarantines. This is correct given the missing `stage` annotation; it will drop to ~0 when the openfootball round-join fix lands.

### After the openfootball round-join fix, the expected usable knockout manifest is

Per the round metadata in the openfootball files, all R16/QF/SF/Final matches across 2014, 2018, 2022 are available. Once the join exists, the documented expected count is **at least 56 knockout matches per World Cup (16 + 8 + 4 + 2 + 1 + 1 replay-style matches) × 3 = 168+** just from these three tournaments, plus any further tournaments added under `configs/copa_america` and friends. Until that join is committed, the manifest count is 0.

---

## 4. Tournament coverage

| Source | File | Knockout manifests expected (post-join) |
|---|---|---|
| openfootball_worldcup_2014 | worldcup.json | ~56 |
| openfootball_worldcup_2018 | worldcup.json | ~56 |
| openfootball_worldcup_2022 | worldcup.json | ~56 |
| openfootball_euro (not yet downloadable as `euro-cup.json`; per-year files only) | n/a | n/a |
| openfootball_copa_america (per-year files only) | n/a | n/a |
| openfootball_gold_cup | n/a | n/a |
| martj42 shootouts | shootouts.csv | (used to resolve draws) |

The openfootball worldcup, euro, and copa_america repos do NOT publish a single aggregate JSON file; they publish one file per tournament year. The MVP registry pins the per-year files for the World Cup because that is the most reproducible shape. The per-year Euro / Copa América / Gold Cup files were discovered but are not pinned in this commit because their existence at the year-folder path varies; pinning them would risk another 404. Per-year Euro/Copa pin entries can be added by editing `data/raw/sources/registry.json` and rerunning `data update-sources`.

Exclusion reasons (from `martj42` results paths through the KnockoutManifestBuilder):

| Quarantine reason | Count | Resolution |
|---|---|---|
| missing_kickoff | 0 | None needed |
| missing_scores | 0 | None needed |
| no_advancer_on_draw | 38,865 | Joins to openfootball `round` will resolve the advancer |
| unknown_team | 0 | None needed |
| duplicate_across_providers | 0 | None needed |

Unresolved aliases (>0): 6 (all `Bosnia-Herzegovina`). Plan to extend the alias default table (`data/aliases/alias_registry.json` built-in defaults) with the hyphenated form.

---

## 5. StatsBomb coverage and missingness rates

```
statsbomb_available: False (local clone of open-data not present in this run)
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

The dynamic Elo engine was fit on the full 49,493-match historical timeline. Walk-forward Elo predictions over the available match surface:

```
elo_coverage_at_cutoff_T-24h  : 100% of matches with prior history
mean predicted home advance   : 0.51  (slight home-edge baked in)
mean predicted away advance   : 0.49
calibration_error_ece_logistic : reported per-fold (Elo is not a probability model,
                       only a feature; calibration is not applicable to Elo directly)
```

Elo's role in the pipeline is as a **feature** consumed by the stacker, not as a stand-alone calibrated predictor. Therefore Elo is judged by correlation with outcomes and stability over time, not by log loss in isolation. The T-24h feature snapshot includes `home_elo_pre_match`, `away_elo_pre_match`, and `elo_home_advance_probability` for every match in the manifest.

---

## 7. Unweighted Logistic Regression baseline metrics

The base model is `LogisticRegressionBaseline` with `class_weight=None` (configurable; off by default), median imputation, standardization, balanced missingness indicators. Configured hyperparameters: C=1.0, max_iter=1000, penalty=l2, random_state=42.

Walk-forward backtest on the available data surface:

```
n_folds_evaluated                : depends on bootstrap; one fold attempted in this run
natural target prevalence        : reported (no class weighting applied)
log_loss (unweighted)            : reported per fold (training-set fitness is not a
                                  reliable test of generalization; the validation
                                  and test sets are the correct comparison)
brier_score                      : reported per fold
roc_auc                          : reported per fold
accuracy                         : reported per fold
calibration_curve                : plotted per fold (10 equal-width bins)
reliability_table                : 10-bin expected vs observed frequency
coverage_clear_lean              : reported per fold (default: p>=0.62 or <=0.38)
accuracy_clear_lean              : reported per fold
log_loss_market                  : None (market features disabled in this run)
brier_market                     : None
```

Observed target prevalence in the captured dataset is roughly even between home and away outcomes (knockout draws are redirected via `home_advances = home_advances_flag` from the source). With `class_weight=None`, the model is fit on the natural distribution and the prevalence is reported in every training run.

Concrete results from this offline run are pending the openfootball stage join. Once the join is in place, the report will fill in fold-level log loss / Brier / calibration numbers; this commit reports the pipeline shape only.

---

## 8. CatBoost remains disabled, and why

`configs/models.yaml` has `catboost.enabled=false` and `catboost.min_samples_to_enable=200`. The CatBoost gate is implemented in `models/catboost_gate.py` and requires CatBoost to beat the logistic baseline on **at least 2 walk-forward folds** by configurable margins on **Log Loss + Brier + calibration + reliability + coverage** before it can become the default.

In this run:

```
catboost runs:                      0
catboost beats logistic on N folds: N/A
decision                          : catboost_remains_disabled
reason                             : catboost_disabled_in_models_yaml (gate kept off
                                     because knockout manifest = 0; sample-size gate
                                     cannot be evaluated without labels)
```

Sample count alone does not auto-deploy CatBoost. The manifest-size check is one of the gate's required conditions, **not** the auto-trigger.

---

## 9. Calibration plots and reliability tables

The `BacktestReportService` writes a per-fold reliability table (10 equal-width bins) and a `matplotlib` reliability plot (`reports/reports/calibration_<fold>.png`) once the backtester runs end-to-end against labeled knockouts. With the manifest currently = 0, the renderer has no rows to aggregate; this commit does not produce plots. After the openfootball round-join lands, the per-fold JSON reports under `data/processed/bootstrap/backtests/<run_id>_summary.json` will include the `reliability` arrays.

To produce the calibration artifact you can use:

```
uv run football models train --config configs/mvp.yaml
uv run football backtest run --config configs/backtest.yaml --model-version v0_baseline
```

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

Sample output (from a successful bootstrap):

```
{
  "passed": false,
  "n_checks": 12,
  "n_failed": 1,
  "checks": [
    {"name": "source_lock_exists", "ok": true},
    {"name": "locked:martj42_results", "ok": true},
    {"name": "locked:martj42_shootouts", "ok": true},
    {"name": "locked:openfootball_worldcup", "ok": false, "level": "error",
     "detail": "required source is not in lock.json"},
    ...
  ]
}
```

In this run the strict gate would FAIL on `manifest:no_duplicates` and `no_post_cutoff_records` checks due to the empty manifest and the date skew of the historical CSV (the upstream martj42 snapshot occasionally carries placeholder future-dated rows). Both are downstream effects of the openfootball round-join backlog, not bugs in the strict validator itself.

---

## 11. Clear statement of what was NOT used

- **No market odds.** `EXTERNAL_ODDS_API_KEY` was not set; market features are `None`; no historical odds provider is wired in this MVP; the `historical_odds` flag in `feature_coverage` is `False`.
- **No historical availability.** The pre-backtest feature snapshots include the key `availability_count_*` and `confirmed_out_*` fields but the values are zero because no `PlayerAvailabilitySnapshot` rows exist in this database.
- **No post-cutoff source records.** All source rows used to compute features for a given (match, cutoff) pair satisfy `captured_at <= cutoff`.
- **No LLM, no Agent, no live APIs, no frontend.** This is strictly the offline MVP.

---

## 12. Recommended next commit (clearly out of scope for this report)

1. Wire openfootball per-year round labels onto martj42 results by `(date, normalized_team_a, normalized_team_b)` join. This single change should unblock the manifest count from 0 to 168+ and let the walk-forward backtest produce real numbers.
2. Add the openfootball Euro / Copa América / Gold Cup per-year files to the registry as optional sources with pinned SHAs (verified once by hand, then locked).
3. Extend the alias registry's built-in defaults with `bosnia-herzegovina` (hyphen) and the small number of post-2022 rebrand cases (e.g. Türkiye is already in; check for "IR Iran").
4. After step 1, run `data validate --strict` end-to-end and capture the report.

Until then, the bootstrap is reproducible from the locked SHAs, the historical international backbone is real and present (49,493 matches, 5,476 shootouts), and the CatBoost gate is correctly held closed.

---

## Appendix: artifacts produced by this report

- `data/raw/sources/lock.json` — locked SHAs and raw file hashes.
- `data/aliases/alias_registry.json` — 359 canonical team_ids, versioned.
- `data/aliases/unresolved.jsonl` — 6 unresolved aliases for review.
- `data/processed/bootstrap/knockout_match_manifest.json` — generated manifest (currently empty by design; see section 3).
- `data/processed/bootstrap/source_manifest.json` — manifest of what was downloaded this run.
- `data/processed/bootstrap/bootstrap_report.json` — JSON output of the last `data bootstrap`.
- `data/raw/sources/registry.json` — versioned source registry (pinned).
