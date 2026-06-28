# Data contracts

This document specifies the data contracts between layers.

## 1. Schemas (Pydantic)

All Pydantic schemas live in `src/football_advance_predictor/schemas/`:

- `matches.py` — `MatchIn`, `MatchOut`, `MatchResultIn`, `MatchResultOut`
- `odds.py` — `MarketOddsIn`, `MarketOddsOut`
- `availability.py` — `AvailabilityIn`, `AvailabilityOut`
- `features.py` — `FeatureBuildRequest`, `FeatureSnapshotOut`
- `predictions.py` — `PredictionRequest`, `PredictionOut`, `EvaluationOut`
- `training.py` — `TrainingRequest`, `BacktestRequest`, `ModelRunOut`

Every input schema enforces explicit lineage columns
(`observed_at`, `published_at`, `ingested_at`, `effective_at`,
`raw_payload_hash`, `source_name`, ...).

## 2. SQLAlchemy ORM models

`src/football_advance_predictor/db/models.py` defines the persistent
schema. The naming conventions in `db/base.py` make Alembic
migrations deterministic.

## 3. Provider contracts

`src/football_advance_predictor/data/adapters/base.py` defines the
provider Protocols:

- `MatchDataProvider.fetch_matches / fetch_results / fetch_teams`
- `OddsProvider.fetch_odds`
- `AvailabilityProvider.fetch_availability`

Implementations must be deterministic and must not perform network
calls in tests. Skeleton external providers fail closed when no API
key is present.

## 4. Raw vs normalized data

Raw payloads are stored unchanged in `data/raw/` (or external blob
storage in production). The application database contains only
normalized records with explicit lineage. A re-ingest does not
overwrite the raw file; it adds a new lineage row.

## 5. Timestamp semantics

- `kickoff_at` — the actual kickoff time (UTC).
- `cutoff_time` — strictly before `kickoff_at`.
- `observed_at` — when the source first observed the data.
- `published_at` — when the source published the data.
- `ingested_at` — when our pipeline wrote the record.
- `effective_at` — when the record became effective (used to
  preserve the original effective moment after a re-ingest).
- `captured_at` — for odds, the time the odds were seen (i.e. when
  the snapshot was taken by the bookmaker API).

## 6. Cutoff enforcement

`core.time.assert_cutoff_before` and
`core.time.assert_observed_before` are the only sanctioned ways to
assert a timestamp invariant. Direct comparisons are discouraged.

## 7. Hashing

`core.hashing.stable_hash` produces a stable SHA-256 hex digest of a
JSON-serializable payload. We never use Python's `hash()` because
hash randomization makes it non-reproducible.
