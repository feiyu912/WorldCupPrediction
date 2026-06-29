"""Regularized logistic regression baseline.

This is the default base model for the MVP. CatBoost is opt-in via
the ``catboost.enabled`` config flag and is only enabled after the
generated knockout manifest reaches a documented minimum sample size
and passes walk-forward validation (see ``configs/models.yaml``).

The model is intentionally simple and interpretable. It operates on
the same feature dict the rest of the pipeline uses; missing values
are mean-imputed and a missingness-indicator column is added per
feature to preserve the missingness signal.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from football_advance_predictor.core.hashing import stable_hash
from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LogisticRegressionBaselineConfig:
    """Configuration for the logistic regression baseline.

    Defaults are conservative: no class weighting (we report natural
    target prevalence; balanced is configurable but off by default),
    median imputation, and missingness indicator columns.
    """

    C: float = 1.0
    max_iter: int = 1000
    random_state: int = 42
    penalty: str = "l2"
    class_weight: str | None = None
    solver: str = "lbfgs"
    add_missingness_indicators: bool = True
    min_samples_required: int = 64
    early_stopping_rounds: int = 25

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticRegressionBaselineConfig:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class LogisticRegressionBaseline:
    """Regularized logistic regression baseline."""

    def __init__(self, config: LogisticRegressionBaselineConfig | None = None) -> None:
        self.config = config or LogisticRegressionBaselineConfig()
        self.model: Pipeline | None = None
        self.feature_columns: list[str] = []
        self.training_log_loss: float | None = None
        self.validation_log_loss: float | None = None
        self.training_size: int = 0
        self.training_prevalence: float | None = None
        self.validation_prevalence: float | None = None

    @staticmethod
    def prevalence(y: Any) -> float:
        """Return the mean of ``y`` as a prevalence estimator."""
        import numpy as np

        return float(np.mean(np.asarray(y, dtype=float)))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> LogisticRegressionBaseline:
        """Fit the model with imputation, scaling, and regularization."""
        if len(X_train) < self.config.min_samples_required:
            logger.warning(
                "Training set is smaller than min_samples_required; the model will be biased",
                extra={"n": len(X_train), "min": self.config.min_samples_required},
            )
        self.feature_columns = list(X_train.columns)
        self._ensure_missingness_indicators(X_train, X_val)
        # Re-fetch columns after indicator expansion.
        self.feature_columns = list(X_train.columns)
        self.model = self._build_pipeline()
        self.model.fit(X_train, y_train)
        train_pred = self.model.predict_proba(X_train)[:, 1]
        self.training_log_loss = float(log_loss(y_train, train_pred, labels=[0, 1]))
        self.training_size = len(X_train)
        self.training_prevalence = self.prevalence(y_train)
        if X_val is not None and y_val is not None:
            self._ensure_missingness_indicators(X_val)
            val_pred = self.model.predict_proba(X_val)[:, 1]
            self.validation_log_loss = float(log_loss(y_val, val_pred, labels=[0, 1]))
            self.validation_prevalence = self.prevalence(y_val)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LogisticRegressionBaseline has not been fitted.")
        X = X.copy()
        self._ensure_missingness_indicators(X)
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[self.feature_columns]
        return self.model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted LogisticRegressionBaseline.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(self.model, directory / "logistic_baseline.joblib")
        manifest = {
            "config": asdict(self.config),
            "feature_columns": self.feature_columns,
            "training_log_loss": self.training_log_loss,
            "validation_log_loss": self.validation_log_loss,
            "training_size": self.training_size,
            "feature_hash": stable_hash(self.feature_columns),
        }
        with (directory / "logistic_baseline_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> LogisticRegressionBaseline:
        import joblib

        directory = Path(directory)
        with (directory / "logistic_baseline_manifest.json").open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        instance = cls(LogisticRegressionBaselineConfig(**manifest["config"]))
        instance.feature_columns = list(manifest["feature_columns"])
        instance.training_log_loss = manifest.get("training_log_loss")
        instance.validation_log_loss = manifest.get("validation_log_loss")
        instance.training_size = manifest.get("training_size", 0)
        instance.model = joblib.load(directory / "logistic_baseline.joblib")
        return instance

    # ------------------------------------------------------------------
    # Importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> list[tuple[str, float]]:
        if self.model is None:
            return []
        try:
            coef = self.model.named_steps["classifier"].coef_.ravel()
        except Exception:
            return []
        return sorted(
            zip(self.feature_columns, [float(v) for v in coef]),
            key=lambda x: -abs(x[1]),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> Pipeline:
        clf = LogisticRegression(
            C=self.config.C,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            penalty=self.config.penalty,
            class_weight=self.config.class_weight,
            solver=self.config.solver,
        )
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", clf),
            ]
        )

    def _ensure_missingness_indicators(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
    ) -> None:
        """Augment ``X_train`` (and optionally ``X_val``) with missingness indicators.

        Operates in place. New columns are named ``{col}_isna`` and are
        ONLY added for source feature columns (never for existing
        indicator columns themselves).
        """
        if not self.config.add_missingness_indicators:
            return

        def _augment(frame: pd.DataFrame) -> None:
            for col in list(frame.columns):
                if col.endswith("_isna"):
                    continue  # Don't double-augment an indicator.
                indicator = f"{col}_isna"
                if indicator in frame.columns:
                    continue
                frame[indicator] = frame[col].isna().astype(int)

        _augment(X_train)
        if X_val is not None:
            _augment(X_val)
