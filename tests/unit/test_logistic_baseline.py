"""Tests for the LogisticRegressionBaseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from football_advance_predictor.models.logistic_baseline.model import (
    LogisticRegressionBaseline,
    LogisticRegressionBaselineConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """200-row synthetic dataset with some signal and some missingness."""
    rng = np.random.default_rng(42)
    n = 200
    home_elo = rng.normal(1500, 200, n)
    away_elo = rng.normal(1500, 200, n)
    elo_diff = home_elo - away_elo
    neutral = rng.integers(0, 2, n)
    feature_a = rng.normal(0, 1, n)

    # Latent logit: positive elo_diff increases P(home win).
    logit = 0.005 * elo_diff - 0.2 * neutral + 0.3 * feature_a
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < p).astype(int)

    X = pd.DataFrame(
        {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "neutral": neutral,
            "feature_a": feature_a,
        }
    )
    return X, pd.Series(y)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_predict_proba_before_fit_raises(synthetic_dataset) -> None:
    X, _ = synthetic_dataset
    model = LogisticRegressionBaseline()
    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.predict_proba(X)


def test_predict_proba_returns_values_in_zero_one(synthetic_dataset) -> None:
    X, y = synthetic_dataset
    model = LogisticRegressionBaseline().fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert float(probs.min()) >= 0.0
    assert float(probs.max()) <= 1.0


def test_feature_importance_sorted_by_abs_coefficient(synthetic_dataset) -> None:
    X, y = synthetic_dataset
    model = LogisticRegressionBaseline().fit(X, y)
    importance = model.feature_importance()
    # All feature columns are present.
    feature_names = [name for name, _ in importance]
    assert set(feature_names) >= {"home_elo", "away_elo", "neutral", "feature_a"}
    # Sorted by absolute coefficient (descending).
    abs_coefs = [abs(coef) for _, coef in importance]
    assert abs_coefs == sorted(abs_coefs, reverse=True)
    # Coefficients are floats.
    for _, coef in importance:
        assert isinstance(coef, float)


def test_save_load_roundtrip(synthetic_dataset, tmp_path: Path) -> None:
    X, y = synthetic_dataset
    model = LogisticRegressionBaseline().fit(X, y)
    original_probs = model.predict_proba(X)

    save_dir = tmp_path / "model"
    model.save(save_dir)
    assert (save_dir / "logistic_baseline.joblib").exists()
    assert (save_dir / "logistic_baseline_manifest.json").exists()

    loaded = LogisticRegressionBaseline.load(save_dir)
    loaded_probs = loaded.predict_proba(X)
    np.testing.assert_allclose(loaded_probs, original_probs, rtol=1e-6)
    # Training log-loss round-trips too.
    assert loaded.training_log_loss is not None
    assert pytest.approx(model.training_log_loss, rel=1e-6) == loaded.training_log_loss


def test_add_missingness_indicators_false(synthetic_dataset) -> None:
    X, y = synthetic_dataset
    # Drop some values to test that the indicator flag actually controls behaviour.
    X_missing = X.copy()
    X_missing.loc[:5, "feature_a"] = np.nan

    cfg = LogisticRegressionBaselineConfig(add_missingness_indicators=False)
    model = LogisticRegressionBaseline(cfg).fit(X_missing, y)
    # No `_isna` columns should be added.
    assert not any(col.endswith("_isna") for col in model.feature_columns)


def test_add_missingness_indicators_true_adds_isna_columns(synthetic_dataset) -> None:
    X, y = synthetic_dataset
    X_missing = X.copy()
    X_missing.loc[:5, "feature_a"] = np.nan

    model = LogisticRegressionBaseline().fit(X_missing, y)
    # feature_a_isna column should be added.
    assert "feature_a_isna" in model.feature_columns
