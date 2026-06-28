"""Reproducibility test: same seed must produce the same CatBoost predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from football_advance_predictor.models.catboost_model.catboost_model import (
    CatBoostConfig,
    CatBoostModel,
)


def _toy_dataset(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    n = 200
    X = pd.DataFrame(
        {
            "elo_diff": rng.normal(0, 100, n),
            "form_diff": rng.normal(0, 1, n),
            "rest_diff": rng.normal(0, 2, n),
        }
    )
    y = (X["elo_diff"] + 10 * X["form_diff"] + rng.normal(0, 5, n) > 0).astype(int)
    return X, pd.Series(y)


def test_catboost_predictions_are_reproducible(tmp_path):
    config = CatBoostConfig(iterations=30, depth=3, learning_rate=0.1, random_seed=42, verbose=False)
    X, y = _toy_dataset()

    model_a = CatBoostModel(config).fit(X, y)
    model_b = CatBoostModel(config).fit(X, y)
    preds_a = model_a.predict_proba(X)
    preds_b = model_b.predict_proba(X)
    np.testing.assert_allclose(preds_a, preds_b, atol=1e-6)
