# Modeling

This document explains the modeling decisions. The codebase is the
authoritative source for hyperparameters (see `configs/`); this file
focuses on the why.

## 1. Dynamic Elo

We use a standard Elo with the following extensions:

- **Competition weight**: an importance factor in `[0, ∞)` multiplies
  the K-factor per match. World Cup knockouts are weighted more
  heavily than friendlies.
- **Margin-of-victory scaling**: optional. When enabled, the rating
  change is multiplied by `1 + log(1 + |goal_diff|) / log(2)`.
- **Time decay**: optional exponential decay applied to inactive
  teams so ratings regress toward the mean.
- **Neutral venue**: subtracts the home-advantage component from the
  expected-score formula.
- **Tie resolution**: a 90-minute draw in a knockout must be resolved
  via extra time + penalties. The MVP uses a simple heuristic
  (`draw_treated_as_50_50`) that adds 0.5 * P(draw) to the home
  advance probability. The draw probability is a configurable prior.

The formula:

```
E_home = 1 / (1 + 10^((R_away - R_home - H) / 400))
delta_home = K * (S_home - E_home) * MOV
```

with `H = 0` for neutral venue.

## 2. Market consensus

We never use 90-minute moneyline probabilities as a stand-in for
advance probabilities. We prefer a two-way `home_to_advance` /
`away_to_advance` market, de-vigged as:

```
p_home = (1/odds_home) / (1/odds_home + 1/odds_away)
p_away = 1 - p_home
```

The market model returns `None` (not invented values) when no valid
two-way market exists for the match before the cutoff.

## 3. CatBoost

The structured-feature classifier is a CatBoost binary classifier
predicting `home_advances`. Features are listed in
`configs/features.yaml` and built by the
`FeatureBuilder`. The model natively handles missing values; we do
not impute features in the model layer. Categorical features are
declared explicitly via the `cat_features` argument.

We do NOT use random train/test splits. Splits are always temporal
(`WalkForwardSplitter`).

## 4. Stacking

The first layer produces three probability estimates:

- `market` — market consensus home advance probability.
- `elo` — dynamic-Elo home advance probability.
- `catboost` — CatBoost home advance probability.

The second layer is an out-of-fold logistic regression with the
three probabilities as inputs. A simple weighted-blend fallback is
available for comparison.

## 5. Calibration

Calibration is fit on a temporal validation period (distinct from
both the training and test windows). We default to isotonic regression
with `out_of_bounds="clip"`. Platt scaling is available for
comparison.

## 6. Why Log Loss and Brier Score

Accuracy on a 50/50 problem is uninformative. Log loss and Brier score
reward calibrated probabilities; they penalize confident wrong
predictions linearly or quadratically. They are the only metrics
that should drive model selection.

## 7. Why accuracy is insufficient

A 50/50 prediction has 50% accuracy in expectation. A model that
predicts 50/50 for every match will look 50% accurate and is useless.
A model that predicts 50/50 but knows when to be more confident will
have lower log loss. Accuracy, in isolation, hides this.

## 8. Why the final objective is advancement probability

The user-facing question is "who advances?" — not "what is the
score?". Internally we preserve 90-minute score information, but
the public product only exposes two probabilities. This keeps the
model honest and the user interface simple.
