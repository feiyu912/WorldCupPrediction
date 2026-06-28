# football-advance-predictor

Production-quality, reproducible football knockout-match advance-probability prediction system.

> The system predicts only `P(home_advances)` and `P(away_advances)` for a future knockout
> match. It is a forecasting research tool — **not** a betting product. It does not
> recommend stakes, bankroll sizes, or "value bets".

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

## 2. Architecture (ASCII)

```
+-------------------------+       +--------------------+       +-----------------+
| Local CSV / JSON        |       | Skeleton external  |       | StatsBomb local |
| (matches, odds, avail.) |  -->  | providers (env-gated) |   | event data      |
+-------------------------+       +--------------------+       +-----------------+
            |                                |                          |
            v                                v                          v
+--------------------------------------------------------------------------------+
|                     Ingestion service (SQLAlchemy, lineage)                   |
+--------------------------------------------------------------------------------+
            |                                |                          |
            v                                v                          v
+------------------------+        +--------------------------+   +------------+
| Dynamic Elo engine     |        | Market consensus (de-vig)|   | Team form  |
+------------------------+        +--------------------------+   +------------+
            \                                |                          |
             \                               v                          v
              +-------->   Feature snapshot service  <-------------------+
                              (time-frozen, immutable)
                                       |
                                       v
+--------------------------------------------------------------------------------+
|          CatBoost classifier (structured features, native NaNs)               |
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

```bash
# 1) Install dependencies (uses uv)
uv sync --extra dev

# 2) Ingest the bundled local fixtures
uv run football ingest matches --file data/fixtures/matches.csv
uv run football ingest odds --file data/fixtures/odds.csv
uv run football ingest availability --file data/fixtures/availability.json

# 3) Build a feature snapshot for a future match
uv run football features build --match-id MATCH_KO_001 --cutoff 2026-06-29T00:00:00Z

# 4) Train a model version
uv run football models train --config configs/mvp.yaml

# 5) Predict
uv run football predict one --match-id MATCH_KO_001 --cutoff 2026-06-29T00:00:00Z --model-version v0.1.0

# 6) Run a temporal backtest
uv run football backtest run --config configs/backtest.yaml --model-version v0_backtest
```

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
