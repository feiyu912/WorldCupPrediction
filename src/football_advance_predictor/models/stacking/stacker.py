"""Out-of-fold stacking model.

The stacker takes three first-layer predictors:

1. market probability
2. Elo probability
3. CatBoost probability

It trains a logistic regression on out-of-fold predictions to combine
them, and provides a simple weighted-blend fallback.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class StackingConfig:
    method: str = "logistic_regression"  # or "weighted_blend"
    logistic_C: float = 1.0
    logistic_max_iter: int = 1000
    random_state: int = 42
    market_weight: float = 0.5
    elo_weight: float = 0.1
    catboost_weight: float = 0.4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StackingConfig:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        # Nested dicts (e.g. logistic_regression: {C: ...})
        if "logistic_regression" in data and isinstance(data["logistic_regression"], dict):
            lr = data["logistic_regression"]
            valid["logistic_C"] = lr.get("C", valid.get("logistic_C", 1.0))
            valid["logistic_max_iter"] = lr.get("max_iter", valid.get("logistic_max_iter", 1000))
        if "weighted_blend" in data and isinstance(data["weighted_blend"], dict):
            wb = data["weighted_blend"]
            valid["market_weight"] = wb.get("market_weight", valid.get("market_weight", 0.5))
            valid["elo_weight"] = wb.get("elo_weight", valid.get("elo_weight", 0.1))
            valid["catboost_weight"] = wb.get("catboost_weight", valid.get("catboost_weight", 0.4))
        return cls(**valid)


@dataclass
class StackingFold:
    """Indices for a single time-aware fold used during stacker training."""

    name: str
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray | None = None


class StackingModel:
    """Combine three base probabilities into a single probability."""

    def __init__(self, config: StackingConfig | None = None) -> None:
        self.config = config or StackingConfig()
        self.model: LogisticRegression | None = None

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(
        self,
        market: np.ndarray,
        elo: np.ndarray,
        catboost: np.ndarray,
        y: np.ndarray,
    ) -> StackingModel:
        X = self._stack(market, elo, catboost)
        if self.config.method == "logistic_regression":
            self.model = LogisticRegression(
                C=self.config.logistic_C,
                max_iter=self.config.logistic_max_iter,
                random_state=self.config.random_state,
            )
            self.model.fit(X, y)
        elif self.config.method == "weighted_blend":
            self.model = None
        else:
            raise ValueError(f"Unknown stacking method: {self.config.method}")
        return self

    def predict_proba(
        self, market: np.ndarray, elo: np.ndarray, catboost: np.ndarray
    ) -> np.ndarray:
        if self.config.method == "logistic_regression":
            if self.model is None:
                raise RuntimeError("StackingModel not fitted.")
            X = self._stack(market, elo, catboost)
            return self.model.predict_proba(X)[:, 1]
        if self.config.method == "weighted_blend":
            w = self.config
            total = w.market_weight + w.elo_weight + w.catboost_weight
            if total <= 0:
                raise ValueError("Stacking weights must be positive.")
            blended = (
                w.market_weight * np.asarray(market)
                + w.elo_weight * np.asarray(elo)
                + w.catboost_weight * np.asarray(catboost)
            ) / total
            return np.clip(blended, 1e-6, 1.0 - 1e-6)
        raise ValueError(f"Unknown stacking method: {self.config.method}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            import joblib

            joblib.dump(self.model, directory / "logistic.joblib")
        with (directory / "stacker_config.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(self.config), f, indent=2, sort_keys=True)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> StackingModel:
        directory = Path(directory)
        with (directory / "stacker_config.json").open("r", encoding="utf-8") as f:
            cfg = StackingConfig(**json.load(f))
        instance = cls(cfg)
        if (directory / "logistic.joblib").exists():
            import joblib

            instance.model = joblib.load(directory / "logistic.joblib")
        return instance

    @staticmethod
    def _stack(market: np.ndarray, elo: np.ndarray, catboost: np.ndarray) -> np.ndarray:
        m = np.asarray(market, dtype=float)
        e = np.asarray(elo, dtype=float)
        c = np.asarray(catboost, dtype=float)
        # Replace missing market (None) with Elo + CatBoost midpoint.
        if m.size and np.issubdtype(type(m.flat[0]), np.floating) is False:
            m = m.astype(float)
        # Use a safe fill: where market is NaN, fall back to mean(elo, catboost).
        mask = np.isnan(m)
        if mask.any():
            m[mask] = 0.5 * (e[mask] + c[mask])
        return np.column_stack([m, e, c])
