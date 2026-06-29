# football-advance-predictor

Production-quality, reproducible football knockout-match advance-probability prediction system.

> The system predicts only `P(home_advances)` and `P(away_advances)` for a future knockout
> match. It is a forecasting research tool — **not** a betting product. It does not
> recommend stakes, bankroll sizes, or "value bets".

**Self-bootstrapping**: the offline MVP runs without any manual data files or paid API
keys. Run `uv run football data bootstrap` and the system downloads pinned source
revisions, validates schemas, builds a team alias registry, and generates a
`knockout_match_manifest` across World Cups, Euros, Copa América, and CONCACAF Gold
Cup. Only **future-facing live fixtures, availability, and historical odds** are
env-gated and never required.

---

## 1. Project goal

Given a future knockout football match (e.g. `France vs Sweden`), output:

```
France advances: 63%
Sweden advances: 37%
Predicted advancing team: France
Confidence: medium
```

"Advances" includes winning in normal time, extra time, or on penalties.

## 2. Self-bootstrapping data layer

The system downloads and processes its own data with zero manual input. The
pipeline is fully reproducible from a pinned source registry.

```
$ uv run football data bootstrap
{
  "required_sources": [
    {"name": "martj42_results", "cache_hit": false, "schema_valid": true, ...},
    {"name": "martj42_shootouts", "cache_hit": false, "schema_valid": true, ...},
    {"name": "openfootball_worldcup", "cache_hit": false, "schema_valid": true, ...}
  ],
  "alias_registry_size": 412,
  "knockout_manifest": {
    "total": 248,
    "tournament_coverage": {
      "FIFA World Cup": 192, "UEFA Euro": 56, ...
    },
    "quarantined_count": 17
  },
  "statsbomb_available": true,
  "feature_coverage": {
    "statsbomb_events": true, "historical_odds": false, "lineups": false
  }
}

$ uv run football data status     # offline-safe; reports current state
```

`data status` works without any downloads — it just inspects the local cache and
alias registry.

## 3. Architecture (ASCII)

```
+-------------------------+       +--------------------+       +-----------------+
| Pinned source registry  |       | Optional historical |       | StatsBomb local |
| (martj42, openfootball) |  -->  | odds via env key    |       | event data (opt) |
+-------------------------+       +--------------------+       +-----------------+
            |                                |                          |
            v                                v                          v
+--------------------------------------------------------------------------------+
|        System-owned AliasRegistry + KnockoutManifestBuilder                   |
+--------------------------------------------------------------------------------+
            |                                |                          |
            v                                v                          v
+------------------------+        +--------------------------+   +------------+
| Dynamic Elo engine     |        | Market consensus (de-vig)|   | Team form  |
+------------------------+        +--------------------------+   +------------+
            \                                |                          |
             \                               v                          v
              +-------->   Feature snapshot service  <-------------------+
                              (time-frozen, immutable; xG optional)
                                       |
                                       v
+--------------------------------------------------------------------------------+
|         Logistic regression baseline (default) | CatBoost (opt-in flag)        |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|      Out-of-fold logistic-regression stacker (market + Elo + CatBoost)        |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|            Isotonic / Platt calibration (temporal validation only)            |
+--------------------------------------------------------------------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|           Immutable prediction ledger + temporal rolling backtests            |
+--------------------------------------------------------------------------------+
```

## 3. Why this is *not* an LSTM-first project

We do not use LSTM, RNN, Transformer, or reinforcement learning in the MVP because:

1. **Time-aware backtesting is the only honest test.** Most deep sequence models
   degrade quickly when evaluated with strict temporal splits on small, sparse
   knockout datasets.
2. **Market odds are a very strong baseline.** A simple logistic-regression
   stacker combining market, Elo, and a few structured features typically
   matches or beats complex deep models on this problem. Adding complexity
   without backtest evidence is not allowed.
3. **Calibration, not accuracy, is the objective.** A well-calibrated logistic
   or CatBoost model produces trustworthy probabilities; deep models are
   frequently miscalibrated without extra work.
4. **Reproducibility.** Tree-based and linear models are far easier to reproduce
   and audit than deep sequence models.

## 4. Why an Agent is auxiliary, not the predictor

LLM/Agent capabilities are restricted to structured, auditable enrichment
(see `src/football_advance_predictor/services/agent_interfaces.py`):

- extract structured availability from text,
- validate whether a news item is published before a cutoff,
- score source reliability,
- draft post-match error analyses.

Agents **never** alter probabilities, invent player ratings, or bypass the
feature schema. Production deployments must implement these interfaces
deterministically and document their evidence trail.

## 5. Market odds as the strongest baseline

Markets aggregate the beliefs of many bettors with real money at stake. Any
custom model must prove it improves over the market consensus on log loss and
Brier score before being treated as useful.

## 6. Quick start

The system is self-bootstrapping; you do not need any local data files.

```bash
# 1) Install dependencies (uses uv)
uv sync --extra dev

# 2) Bootstrap data: download pinned sources, build the knockout manifest,
#    seed the alias registry. Skip --offline to require a network connection.
uv run football data bootstrap
uv run football data status         # offline-safe; report what was bootstrapped

# 3) Build a feature snapshot for a future match (using the bootstrapped DB)
uv run football features build --match-id <MATCH_ID> --cutoff <ISO_TIMESTAMP>

# 4) Train a model version. CatBoost is opt-in via configs/models.yaml and
#    requires the generated manifest to reach min_samples_to_enable (default 200).
uv run football models train --config configs/mvp.yaml

# 5) Predict
uv run football predict one --match-id <MATCH_ID> --cutoff <ISO_TIMESTAMP> --model-version v0.1.0

# 6) Run a temporal backtest
uv run football backtest run --config configs/backtest.yaml --model-version v0_backtest
```

The only inputs you may provide later are **API keys** for env-gated
optional providers (live fixtures, availability, historical odds). The
offline MVP runs without any.

## 7. Docker

```bash
docker compose up -d
# Wait for the healthcheck, then:
docker compose exec api uv run football ingest matches --file data/fixtures/matches.csv
docker compose exec api uv run football models train --config configs/mvp.yaml
docker compose exec api uv run football predict one --match-id MATCH_KO_001 --cutoff 2026-06-29T00:00:00Z --model-version v0.1.0
```

## 8. API examples

```bash
# Health
curl http://localhost:8000/health

# Ingest a match
curl -X POST http://localhost:8000/ingest/matches \
     -H "Content-Type: application/json" \
     -d '[{"match_id":"X","kickoff_at":"2026-07-01T04:00:00Z","competition_id":"WC","stage":"QF","season_or_year":"2026","home_team_id":"france","away_team_id":"sweden","neutral_venue":true}]'

# Build a feature snapshot
curl -X POST http://localhost:8000/features/build \
     -H "Content-Type: application/json" \
     -d '{"match_id":"MATCH_KO_001","cutoff_time":"2026-06-30T04:00:00Z","feature_version":"v1"}'

# Get a prediction
curl -X POST http://localhost:8000/predictions \
     -H "Content-Type: application/json" \
     -d '{"match_id":"MATCH_KO_001","cutoff_time":"2026-06-30T04:00:00Z","model_version":"v0.1.0"}'
```

## 9. Tests

```bash
uv run pytest                  # run all tests
uv run pytest tests/unit       # unit tests only
uv run pytest tests/integration # integration + anti-leakage
```

The integration suite demonstrates the central anti-leakage guarantees:

- a T-24h snapshot does not see a lineup confirmation published later,
- a T-75min snapshot does,
- a post-kickoff availability record is rejected by the snapshot service,
- the prediction ledger is immutable.

## 10. Model limitations

- This is a forecasting research tool, not a betting product.
- It does NOT claim to beat bookmaker markets.
- The strongest legitimate baseline is the market consensus itself. We
  report log loss and Brier score against this baseline on every backtest.
- Calibrated probabilities can still be wrong; the system never claims
  certainty.
- The default base model is a **regularized logistic regression** with
  median imputation, standardization, balanced class weights, and
  missingness indicators. **CatBoost is opt-in** via
  `configs/models.yaml` and is only enabled when the generated knockout
  manifest reaches `catboost.min_samples_to_enable` (default 200) and
  passes walk-forward validation.
- The model uses **all international matches** for Elo state but
  **only reliably-labeled knockout fixtures** for the advancement
  target. The exact count of usable knockout matches and the per-
  tournament coverage are printed by `uv run football data bootstrap`.
- Historical market odds are **optional** and disabled by default.
  When `EXTERNAL_ODDS_API_KEY` is set, the system uses a single
  reproducible T-24h snapshot per match. No fabrication or backfill.
- Historical availability / lineup data are **future-facing only**.
  v0 backtests run without availability features; the
  `AvailabilityProvider` interface is reserved for live use.

## 11. Responsible use

- Do not use predictions to inform gambling decisions.
- Do not assume that more complex neural networks will improve results
  without rigorous time-aware backtesting.
- Always treat probabilities as forecasts under uncertainty, not as
  guarantees.

## 12. Project layout

See `docs/architecture.md` for the full layout, and
`docs/anti-leakage.md`, `docs/modeling.md`, `docs/data-contracts.md`, and
`docs/evaluation.md` for design details.
