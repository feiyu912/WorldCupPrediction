"""Probability calibration.

Implements isotonic and Platt (sigmoid) calibration. The calibrator
must be fit on a temporally separate validation period to avoid
in-sample calibration.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


@dataclass
class CalibrationConfig:
    method: str = "isotonic"  # or "platt"
    isotonic_out_of_bounds: str = "clip"
    platt_C: float = 1.0
    platt_max_iter: int = 1000

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibrationConfig:
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        if "isotonic" in data and isinstance(data["isotonic"], dict):
            iso = data["isotonic"]
            valid["isotonic_out_of_bounds"] = iso.get(
                "out_of_bounds", valid.get("isotonic_out_of_bounds", "clip")
            )
        if "platt" in data and isinstance(data["platt"], dict):
            pl = data["platt"]
            valid["platt_C"] = pl.get("C", valid.get("platt_C", 1.0))
            valid["platt_max_iter"] = pl.get("max_iter", valid.get("platt_max_iter", 1000))
        return cls(**valid)


class Calibrator:
    """Calibrate a probability estimator.

    Args:
        config: :class:`CalibrationConfig`.
    """

    def __init__(self, config: CalibrationConfig | None = None) -> None:
        self.config = config or CalibrationConfig()
        self._isotonic: IsotonicRegression | None = None
        self._platt: LogisticRegression | None = None

    def fit(self, probs: np.ndarray, y: np.ndarray) -> Calibrator:
        probs = np.asarray(probs, dtype=float)
        y = np.asarray(y, dtype=int)
        if self.config.method == "isotonic":
            self._isotonic = IsotonicRegression(
                out_of_bounds=self.config.isotonic_out_of_bounds
            )
            self._isotonic.fit(probs, y)
        elif self.config.method == "platt":
            X = _logit(probs).reshape(-1, 1)
            self._platt = LogisticRegression(
                C=self.config.platt_C, max_iter=self.config.platt_max_iter
            )
            self._platt.fit(X, y)
        else:
            raise ValueError(f"Unknown calibration method: {self.config.method}")
        return self

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=float)
        if self.config.method == "isotonic":
            if self._isotonic is None:
                raise RuntimeError("Calibrator not fitted.")
            out = self._isotonic.predict(probs)
            return np.clip(out, 1e-6, 1.0 - 1e-6)
        if self.config.method == "platt":
            if self._platt is None:
                raise RuntimeError("Calibrator not fitted.")
            X = _logit(probs).reshape(-1, 1)
            out = self._platt.predict_proba(X)[:, 1]
            return np.clip(out, 1e-6, 1.0 - 1e-6)
        raise ValueError(f"Unknown calibration method: {self.config.method}")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "calibrator_config.json").open("w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=2, sort_keys=True)
        if self._isotonic is not None:
            import joblib

            joblib.dump(self._isotonic, directory / "isotonic.joblib")
        if self._platt is not None:
            import joblib

            joblib.dump(self._platt, directory / "platt.joblib")
        return directory

    @classmethod
    def load(cls, directory: str | Path) -> Calibrator:
        directory = Path(directory)
        with (directory / "calibrator_config.json").open("r", encoding="utf-8") as f:
            cfg = CalibrationConfig(**json.load(f))
        instance = cls(cfg)
        if (directory / "isotonic.joblib").exists():
            import joblib

            instance._isotonic = joblib.load(directory / "isotonic.joblib")
        if (directory / "platt.joblib").exists():
            import joblib

            instance._platt = joblib.load(directory / "platt.joblib")
        return instance


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))
