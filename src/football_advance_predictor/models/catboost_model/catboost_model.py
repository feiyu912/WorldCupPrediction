"""CatBoost structured-feature classifier.

The wrapper is feature-version aware: the training routine is given a
DataFrame of feature rows and a target vector. Missing values are
handled natively by CatBoost; we do not impute features in the model
layer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss

from football_advance_predictor.core.hashing import stable_hash
from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CatBoostConfig:
    """CatBoost hyperparameters."""

    iterations: int = 500
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 3.0
    loss_function: str = "Logloss"
    eval_metric: str = "Logloss"
    random_seed: int = 42
    thread_count: int = 4
    allow_writing_files: bool = False
    class_weights: list[float] = field(default_factory=list)
    early_stopping_rounds: int = 50
    verbose: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CatBoostConfig:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


class CatBoostModel:
    """Wrapper around ``catboost.CatBoostClassifier``."""

    def __init__(self, config: CatBoostConfig | None = None) -> None:
        self.config = config or CatBoostConfig()
        self.model: CatBoostClassifier | None = None
        self.feature_columns: list[str] = []
        self.categorical_features: list[str] = []
        self.training_log_loss: float | None = None
        self.validation_log_loss: float | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        categorical_features: Iterable[str] | None = None,
    ) -> CatBoostModel:
        """Fit the model.

        Args:
            X_train: Training features.
            y_train: Training labels (0/1).
            X_val: Optional validation features.
            y_val: Optional validation labels.
            categorical_features: Optional categorical column names.
        """
        self.feature_columns = list(X_train.columns)
        self.categorical_features = list(categorical_features or [])
        cat_indices = [
            self.feature_columns.index(c)
            for c in self.categorical_features
            if c in self.feature_columns
        ]
        params = self._build_params()
        self.model = CatBoostClassifier(**params)
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            cat_features=cat_indices or None,
            verbose=int(self.config.verbose),
        )
        # Log training metrics.
        train_pred = self.model.predict_proba(X_train)[:, 1]
        self.training_log_loss = float(log_loss(y_train, train_pred, labels=[0, 1]))
        if eval_set is not None:
            val_pred = self.model.predict_proba(X_val)[:, 1]
            self.validation_log_loss = float(
                log_loss(y_val, val_pred, labels=[0, 1])
            )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted probabilities for the positive class."""
        if self.model is None:
            raise RuntimeError("CatBoostModel has not been fitted.")
        # Reorder columns to match the training order and warn on missing.
        missing = [c for c in self.feature_columns if c not in X.columns]
        for col in missing:
            X = X.copy()
            X[col] = np.nan
        X = X[self.feature_columns]
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Cannot save an unfitted CatBoostModel.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model_path = directory / "catboost.cbm"
        self.model.save_model(str(model_path))
        manifest = {
            "config": asdict(self.config),
            "feature_columns": self.feature_columns,
            "categorical_features": self.categorical_features,
            "training_log_loss": self.training_log_loss,
            "validation_log_loss": self.validation_log_loss,
            "feature_hash": stable_hash(self.feature_columns),
        }
        with (directory / "catboost_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        logger.info("Saved CatBoost artifact", extra={"path": str(directory)})
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> CatBoostModel:
        directory = Path(directory)
        with (directory / "catboost_manifest.json").open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        instance = cls(CatBoostConfig(**manifest["config"]))
        instance.feature_columns = list(manifest["feature_columns"])
        instance.categorical_features = list(manifest["categorical_features"])
        instance.training_log_loss = manifest.get("training_log_loss")
        instance.validation_log_loss = manifest.get("validation_log_loss")
        instance.model = CatBoostClassifier(**instance._build_params())
        instance.model.load_model(str(directory / "catboost.cbm"))
        return instance

    # ------------------------------------------------------------------
    # Importance
    # ------------------------------------------------------------------

    def feature_importance(self) -> list[tuple[str, float]]:
        if self.model is None:
            return []
        importance = self.model.get_feature_importance()
        return sorted(
            zip(self.feature_columns, [float(v) for v in importance]),
            key=lambda x: -x[1],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_params(self) -> dict[str, Any]:
        params = asdict(self.config)
        # ``class_weights`` may be a list; CatBoost accepts None or list.
        if not params.get("class_weights"):
            params["class_weights"] = None
        return params
