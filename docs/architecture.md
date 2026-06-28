# Architecture

The system is structured as a layered pipeline with clear anti-leakage
boundaries.

## 1. High-level components

| Component | Module | Responsibility |
|---|---|---|
| Ingestion | `src/football_advance_predictor/data/ingestion/` | Validate provider records, write lineage columns, deduplicate by `raw_payload_hash`. |
| Provider adapters | `src/football_advance_predictor/data/adapters/` | Pluggable: local CSV, local JSON, skeleton external APIs, StatsBomb local events. |
| Normalization | `src/football_advance_predictor/data/normalization/` | Team name resolution (alias table, quarantine). |
| Snapshots | `src/football_advance_predictor/data/snapshots/` | Build immutable feature snapshots, hash, deduplicate. |
| Warehouse | `src/football_advance_predictor/data/warehouse/` | DuckDB-backed offline analytical store. |
| Feature builders | `src/football_advance_predictor/features/` | Compose Elo, market, form, lineup, and competition features. |
| Models | `src/football_advance_predictor/models/` | Elo, market, CatBoost, stacker, calibrator, registry. |
| Services | `src/football_advance_predictor/services/` | Training, prediction, agent enrichment interfaces. |
| Backtesting | `src/football_advance_predictor/backtesting/` | Walk-forward splitter, metrics, reliability plot, report. |
| Ledger | `src/football_advance_predictor/ledger/` | Immutable prediction storage, evaluation, exports. |
| API | `src/football_advance_predictor/app/api/` | FastAPI endpoints. |
| CLI | `src/football_advance_predictor/cli/` | Typer commands. |
| DB | `src/football_advance_predictor/db/` | SQLAlchemy 2.x models, session, alembic env. |

## 2. Data flow

1. A provider adapter (e.g. `LocalHistoricalResultsProvider`) returns
   `MatchIn` / `MarketOddsIn` / `AvailabilityIn` records.
2. The ingestion service writes them to the application database with
   explicit lineage columns and a content-hash for deduplication.
3. The feature builder queries only records strictly older than the
   cutoff and computes a feature dict.
4. The feature snapshot service persists the feature dict with an
   immutable hash. Re-running returns the same hash.
5. The training service fits CatBoost, the stacker, and the calibrator
   on a temporal split.
6. The prediction service uses the trained artifacts to produce a
   calibrated home advance probability and writes an immutable
   prediction record.
7. The backtest runner executes the same pipeline across multiple
   time-aware folds and produces a report.

## 3. Configuration

All hyperparameters live in `configs/`. There are no magic numbers
hidden in code paths.

- `base.yaml` — paths, cutoffs, confidence bands, market thresholds.
- `elo.yaml` — Elo K-factor, home advantage, time decay, MOV.
- `features.yaml` — feature toggles and windows.
- `catboost.yaml` — CatBoost hyperparameters.
- `stacking.yaml` — stacker method and weights.
- `calibration.yaml` — calibration method and reliability bins.
- `backtest.yaml` — walk-forward fold definitions.
- `providers.yaml` — provider selection.
- `mvp.yaml` — top-level MVP training configuration.

## 4. Anti-leakage boundaries

Every feature is computed against a single `cutoff_time` per
prediction. The feature snapshot service rejects any cutoff that is
not strictly before the match kickoff. See `docs/anti-leakage.md`.
