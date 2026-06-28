# Evaluation

## 1. Metrics

- **Log Loss** — primary metric. Penalizes confident wrong
  predictions.
- **Brier Score** — mean squared error of probabilities vs binary
  labels. Decomposes into calibration + refinement.
- **ROC AUC** — used when meaningful. Reported per fold.
- **Accuracy** — reported but not optimized.
- **Expected Calibration Error (ECE)** — used to assess calibration
  quality.

## 2. Calibration

We always report:

- A reliability table (10 equal-width bins by default).
- A reliability plot (PNG, `matplotlib`, headless).
- ECE.

Calibration is trained on a temporal validation period that is
distinct from the training and test periods. The test period is
NEVER used to fit the calibrator.

## 3. Confidence bands

We assign one of three bands to each prediction:

- `clear_lean` — calibrated `P(home_advances) >= 0.62` or `<= 0.38`
- `slight_lean` — in `[0.55, 0.62)` or `(0.38, 0.45]`
- `near_coin_flip` — otherwise

Thresholds are configurable in `configs/base.yaml`. We report:

- accuracy conditional on the band,
- coverage (fraction of matches in the band).

This prevents cherry-picking easy matches.

## 4. Coverage / abstention

Coverage is the fraction of matches where the model issues a
non-trivial prediction (i.e. anything other than `near_coin_flip`).
A high-coverage model with low accuracy is a model that is too
confident. A low-coverage model with high accuracy is too cautious.
We report both.

## 5. Market baseline comparison

Every backtest reports:

- `log_loss_market`, `brier_market` — market consensus metrics.
- `log_loss_elo`, `brier_elo` — Elo-only metrics.
- Stacked and calibrated model metrics.

If the custom model cannot beat the market on log loss AND Brier
score, the report makes that explicit.

## 6. Coverage of anti-leakage tests

See `docs/anti-leakage.md` and the integration tests under
`tests/integration/`.

## 7. Reproducibility

All training routines accept a fixed random seed (configurable in
`configs/catboost.yaml`). The reproducibility test under
`tests/integration/test_reproducibility.py` verifies that two
training runs with the same seed produce identical predictions.
