# Anti-leakage

The single most important engineering requirement of this system is
that no information from the future can leak into a pre-match
feature. This document states the contract and how it is enforced.

## 1. The cutoff-time principle

For a match `M` with kickoff time `T`, every input to a feature
snapshot must satisfy:

```
allowed_information_timestamp <= cutoff_time < T
```

A `cutoff_time` that is not strictly before `T` is rejected by the
feature snapshot service with a `ValueError`.

## 2. Examples of leakage and how we reject it

- **Final score of M.** Not used as a feature. The CatBoost target is
  derived from the post-match result; training-time rows carry
  results, but prediction-time rows do not have access to results.
- **Event data after kickoff.** StatsBomb-like event data is matched
  to matches by match_id; we never include event data generated
  after the cutoff. The StatsBomb local adapter is opt-in and
  computed only from timestamped event files.
- **Lineup published after cutoff.** The lineup-derived features
  (e.g. `lineup_confirmed`, `confirmed_out_*`) only consider
  `PlayerAvailabilitySnapshot.published_at <= cutoff`.
- **Injury news after cutoff.** Same mechanism as lineup. The
  availability ingestion layer stamps every record with
  `published_at`, `observed_at`, `ingested_at`, and `effective_at`.
- **Later ranking updates.** The Elo engine fits chronologically and
  exposes `get_team_rating(team_id, as_of_time)`. Predictions only
  use ratings as of the cutoff, not the present.
- **Current versions of pages with retrospective corrections.** The
  data lineage columns preserve the original payload hash, and we
  store raw payloads unchanged. Re-ingesting with a new hash is the
  only way to update a record.
- **Future matches in form computation.** `weighted_points` and
  `weighted_goal_difference` filter matches with `kickoff_at <
  cutoff` before applying time decay.
- **Random train/test splits.** The system explicitly forbids random
  splits. The `WalkForwardSplitter` only supports time-aware folds.

## 3. Data lineage

Every record carries:

- `source_name`, `source_record_id`, `source_url`
- `observed_at`, `published_at` (if available), `ingested_at`,
  `effective_at`
- `raw_payload_hash`
- `source_version` (if available)

Every feature snapshot carries:

- `match_id`, `cutoff_time`, `feature_version`, `generated_at`
- `features_json`, `source_data_max_timestamp`
- `immutable_hash`

Every prediction carries:

- `match_id`, `cutoff_time`, `model_version`, `feature_snapshot_id`
- `home_advance_probability`, `away_advance_probability`
- `immutable_hash`, `status`, `explanation_payload`
- `created_at`

## 4. Time-aware backtesting

All backtests use the `WalkForwardSplitter`. A fold is defined as:

```
train:      [train_start, train_end]
validation: [validation_start, validation_end]
test:       [test_start, test_end]
```

The validation window is used to fit the stacker and calibrator; the
test window is used to evaluate the final model. The test window is
NEVER used during training or calibration.

## 5. Why random train/test split is invalid

If matches are shuffled, a model can learn from a 2022 match to
predict a 2018 match, even though in production only the past is
visible. Random splits dramatically over-estimate accuracy and
calibration. We do not allow them.

## 6. Automated anti-leakage tests

The integration suite under `tests/integration/test_anti_leakage.py`
intentionally attempts leakage and verifies the pipeline rejects or
filters the bad data:

- `test_elo_does_not_use_future_match_results`
- `test_t24h_snapshot_excludes_lineup_confirmation`
- `test_t75min_snapshot_includes_lineup_confirmation`
- `test_post_kickoff_availability_is_rejected`
- The end-to-end test verifies the prediction ledger is immutable.
