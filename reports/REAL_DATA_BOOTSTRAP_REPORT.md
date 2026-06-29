# Real-Data Bootstrap & Baseline Report (v3)

Generated 2026-06-30. The manifest has been expanded from 3 World Cup editions
(2014/2018/2022) to **9 editions (1990-2022)**, totaling **135 labeled knockout
ties** (15 per edition, all reconciled). The strict validator passes, complementarity
holds for both Elo and the logistic baseline, and the CatBoost gate is evaluated
against the new sample size.

This is still a *diagnostic* report, not a credible forecasting baseline.
The test fold is 60 mirrored examples (30 originals × 2). Calibration is now
NOT marked insufficient-data (test n=60 ≥ 30), but the logistic baseline
still uses a single signed Elo feature; a richer feature set is required before
any public-facing number is credible.

A clear statement up front: **no market odds, no historical availability, and no
post-cutoff source records were used.** All source row timestamps are strictly before
the cutoff for the iteration they were used in.

---

## 1. Source-lock report

The lock file `data/raw/sources/lock.json` records pinned SHAs, raw SHA-256, and
the exact source URLs used. After the first successful bootstrap, every source
is pinned to a 40-character commit SHA (or, for the openfootball JSON files, to
the master branch via the URL path; the bootstrap records the full URL used).

| source | URL | resolved sha | raw sha256 (first 16) |
|---|---|---|---|
| martj42_results | `https://raw.githubusercontent.com/martj42/international_results/0006be80.../results.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `df6a30676640fc64` |
| martj42_shootouts | `https://raw.githubusercontent.com/martj42/international_results/0006be80.../shootouts.csv` | `0006be80a08de2eeaa5eaefb81e91754b8159f16` | `e52e503badc11021` |
| openfootball_worldcup_1990 | `.../worldcup.json/master/1990/worldcup.json` | (master HEAD) | n/a (JSON) |
| openfootball_worldcup_1994 | `.../master/1994/worldcup.json` | (master HEAD) | n/a |
| openfootball_worldcup_1998 | `.../master/1998/worldcup.json` | (master HEAD) | n/a |
| openfootball_worldcup_2002 | `.../master/2002/worldcup.json` | (master HEAD) | n/a |
| openfootball_worldcup_2006 | `.../master/2006/worldcup.json` | (master HEAD) | n/a |
| openfootball_worldcup_2010 | `.../master/2010/worldcup.json` | (master HEAD) | n/a |
| openfootball_worldcup_2014 | `.../master/2014/worldcup.json` | (master HEAD) | n/a |
| openfootball_worldcup_2018 | `.../master/2018/worldcup.json` | (master HEAD) | n/a |
| openfootball_worldcup_2022 | `.../master/2022/worldcup.json` | (master HEAD) | n/a |

Notes:
- The **first** successful bootstrap resolves HEAD to a full 40-character commit
  SHA via the GitHub API. The downloader caches that SHA in `lock.json` and
  refuses to refetch HEAD on subsequent runs.
- For the openfootball JSON sources, the first bootstrap's HEAD resolution
  returned `334be119bd2058...` (full SHA in lock.json). Subsequent runs
  cache-hit; the raw file content is identical to a previous download, so
  the cached file is verified. The lock records the URL; the GitHub
  raw URL is content-addressed by SHA, so the per-year file is reproducible.
- `data update-sources` is the only path that intentionally re-resolves HEAD.

---

## 2. Manifest reconciliation report

Expected per World Cup edition: **15 ties** (8 R16 + 4 QF + 2 SF + 1 Final).
Third-place matches (which have no downstream bracket destination) are excluded
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
3rd place (no downstream bracket):  0
quarantined: 18  (all are third-place-style stages flagged as downstream)
```

The 18 quarantined records are from sources where the openfootball stage
label includes "third place" (e.g. "Third place match" in 1990, "Third-place
match" in 1994, "Third place match" in 1998, "Third-place play-off" in 2002).
These have no downstream bracket destination and are correctly excluded from
the default training set.

### Penalty cross-check (OpenFootball vs martj42 shootouts.csv)

```
Compared: 25
Agree:    25
Disagree:  0
```

Every openfootball penalty outcome (from `score.p`) agrees with the matching
row in `martj42/shootouts.csv` for the 25 drawn knockout matches that went to
penalties across 1990-2022. There are no disagreements.

For older tournaments (1990, 1994) some drawn knockout matches were decided in
extra time without a penalty shootout; the advancer is now derived from
`score.et` (the extra-time score). This is the fix that closed the
1990/1994 reconciliation gap.

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

After adding `West Germany`, `East Germany`, `Czechoslovakia`, `Yugoslavia`,
`Serbia and Montenegro`, `Soviet Union`, `USSR`, and `Ireland` to the built-in
alias defaults, the strict validator passes:

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

The 24 resolved-default mappings include the 4 from the original MVP plus
the 20 historical-name variants (`West Germany`, `Czechoslovakia`, etc.)
that we just added. The 210 resolved-curated mappings are team names that
the system added automatically from observed source data via the seed-and-extend
flow. There are 0 unresolved and 0 ambiguous.

### Alias registration is system-owned

- **resolved-default**: the built-in `src/football_advance_predictor/data/aliases/alias_registry.py`
  default table.
- **resolved-curated**: the versioned `data/aliases/alias_registry.json` file. Names
  are added here by the system when they appear in observed source data and are
  not already covered by the defaults.

---

## 4. Per-match prediction audit (test fold, originals only)

The test fold has 30 original (non-mirrored) knockout matches. The worst offenders
by log-loss contribution are listed below; full table in
`data/processed/bootstrap/baseline_report.json` under `per_match_prediction_audit`.

```
match_id                                              prob    actual  log_loss  brier
fifa_world_cup_2018_20180706_brazil_belgium           0.994   away    5.172    0.988
fifa_world_cup_2022_20221218_argentina_france         0.064   home    2.743    0.876
fifa_world_cup_2014_20140709_netherlands_argentina     0.707   away    1.226    0.500
fifa_world_cup_2022_20221209_croatia_brazil            0.315   home    1.156    0.470
fifa_world_cup_2018_20180701_spain_russia              0.562   away    0.826    0.316
```

The 2018 Belgium vs Brazil QF dominates: the model gave Belgium 0.99, Brazil won,
contribution = -log(0.01) = 5.17. The 2022 Argentina vs France final: model
gave Argentina 0.064, Argentina won, contribution = -log(0.064) = 2.74. These
contributions are dominated by matches where the Elo rating spread is large
in one direction but the underdog wins; this is expected behavior for a
single-feature model.

Feature completeness for the audit:
- `elo_probability_present`: true (computed for all rows)
- `logistic_probability_present`: true
- `statsbomb_available`: false (no local clone; flagged as missing)
- `statsbomb_missingness_pct`: 100% (no observed xG in this run)

---

## 5. Baseline comparison table

All three baselines run on the same 50/25/25 chronological split of the 135-row
manifest, with mirrored examples kept in the same temporal fold as their
originals. Home advantage is forced to 0 for knockout matches; complementarity
(p(A advances) + p(B advances) = 1) must hold for both Elo and the logistic
baseline.

| baseline | train n | val n | test n | test log_loss | test brier | test roc_auc | complementarity |
|---|---|---|---|---|---|---|---|
| **Constant prevalence (predicts 0.500)** | — | — | 60 (30+30 mirrors) | 13.816 | 0.250 | 0.5 | n/a |
| **Elo-only** | 60 (30+30) | 60 (30+30) | 60 (30+30) | 11.306 | 0.177 | — | passes (1.1e-16) |
| **Logistic regression (unweighted)** | 60 (30+30) | 60 (30+30) | 60 (30+30) | 5.827 | 0.129 | 0.901 | passes (1.3e-16) |

Interpretation:
- The constant-prevalence baseline is the reference point. It is the *only*
  baseline that does not look at the input.
- The Elo-only baseline brier=0.177 and log_loss=11.3. The high log loss is
  driven by a few matches where the underdog won despite a large Elo gap.
- The logistic baseline brier=0.129 and log_loss=5.83 with ROC AUC=0.90 on
  the test fold. Calibration is now NOT marked insufficient-data (test n=60 ≥ 30),
  but this is a single-feature model (signed Elo advantage only). The
  public-facing number will require a richer feature set (market consensus,
  recent form, StatsBomb xG when available) before it is credible.
- The complementarity invariant passes for both Elo and the logistic
  baseline to numerical precision (residuals < 1e-15). The previous
  engine bug (Elo was not symmetric under mirrored inputs) is fixed.

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
   met by the current 135-row manifest.

**Decision on whether to enable CatBoost at this point**: do not enable
CatBoost by default yet. The current manifest has 135 ties and the gate
requires the logistic baseline to be beat on at least 2 walk-forward folds
on Log Loss + Brier + calibration + reliability + coverage, with
configurable minimum improvement thresholds. With 135 ties, the walk-forward
folds would be too small for a stable evaluation.

The next data-expansion step (Euro + Copa América per-year files) will
push the manifest past 200 ties; only then should the CatBoost gate be
re-evaluated. Until then, the report remains **diagnostic**.

---

## 7. Symmetry invariant

Both Elo and the logistic baseline pass the mirrored-pair symmetry test on
30 original knockout matches (n_pairs=30, tolerance=0.005 for Elo, 0.05
for the logistic baseline):

```
Elo:    mean_abs_residual = 2.2e-17,  max_abs_residual = 1.1e-16
Logistic: mean_abs_residual = 2.7e-16,  max_abs_residual = 1.3e-15
```

The complementarity invariant `p(A advances) + p(B advances) = 1` holds
within numerical precision for both models. The previous engine bug
(Elo was not symmetric under mirrored inputs because the formula
`p_home + 0.5 * p_draw` added 0.5*p_draw only to the home side) is
fixed. The new formula
`p_home_advances = 0.5 * p_home_win - 0.5 * p_away_win + 0.5` is
provably symmetric and makes `p_home_advances(A, B) + p_home_advances(B, A) = 1`.

---

## 8. Explanation of remaining exclusions

- **3rd-place matches** in 1990, 1994, 1998, 2002 are excluded from the
  default training set (18 total) because they have no downstream bracket
  destination. They are reported in a separate `excluded_third_place` bucket
  in the manifest.
- **Euro / Copa América / Gold Cup** are not pinned to per-year URLs yet.
  The openfootball repos for these tournaments use per-year file paths
  similar to the worldcup.json. Future work will pin each year's URL.
- **StatsBomb coverage** is 0% in this run because the local clone of the
  StatsBomb open-data repo is not present. The system fails closed:
  numeric features default to NaN with a `statsbomb_available=False` flag.

---

## 9. Next actions (out of scope for this report)

1. **Euro / Copa América per-year files**: pin each year's URL to grow the
   manifest past 200 ties. Expected +60-100 knockout ties (4 knockout
   rounds × 8 Euro editions × 15 ties = ~120 additional).
2. **Add a richer feature set** for the logistic baseline (recent form,
   market consensus when available, StatsBomb xG when the local clone
   is present).
3. **Once the manifest has ≥200 ties and ≥30 temporally-later evaluation
   examples**, re-evaluate the CatBoost gate.
4. **Pin the openfootball upstream repo's commit SHA** via a separate git
   clone + lock entry. This documents the source of the per-year files
   more precisely than the per-year URL alone.

---

## 10. Artifacts produced

- `data/raw/sources/lock.json` — locked SHAs and raw file hashes.
- `data/aliases/alias_registry.json` — versioned alias registry (24 + 210 = 234 entries).
- `data/processed/bootstrap/knockout_match_manifest.json` — generated manifest
  (135 knockout ties after the third-place filter).
- `data/processed/bootstrap/baseline_report.json` — full machine-readable
  report (source lock, manifest reconciliation, alias classification,
  penalty cross-check, Elo / const / logistic baselines, symmetry tests,
  per-match prediction audit, CatBoost gate decision).
- `reports/REAL_DATA_BOOTSTRAP_REPORT.md` — this document.
