"""Walk-forward / expanding-window splitter.

Each fold defines a strict train / validation / test window. Splits are
never random; they are always time-aware.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from football_advance_predictor.core.time import to_utc


@dataclass
class Fold:
    """A single time-aware fold."""

    name: str
    train_start: datetime
    train_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime


@dataclass
class FoldMetrics:
    """Per-fold metrics (filled by the backtest runner)."""

    fold_name: str
    n_train: int
    n_validation: int
    n_test: int
    log_loss: float | None
    brier_score: float | None
    roc_auc: float | None
    accuracy: float | None
    coverage_clear_lean: float | None
    accuracy_clear_lean: float | None
    log_loss_market: float | None
    brier_market: float | None
    log_loss_elo: float | None
    brier_elo: float | None


@dataclass
class WalkForwardConfig:
    folds: list[Fold]

    @classmethod
    def from_yaml(cls, data: dict) -> WalkForwardConfig:
        folds: list[Fold] = []
        for raw in data.get("folds", []):
            folds.append(
                Fold(
                    name=raw["name"],
                    train_start=to_utc(raw["train_start"]),
                    train_end=to_utc(raw["train_end"]),
                    validation_start=to_utc(raw["validation_start"]),
                    validation_end=to_utc(raw["validation_end"]),
                    test_start=to_utc(raw["test_start"]),
                    test_end=to_utc(raw["test_end"]),
                )
            )
        return cls(folds=folds)


class WalkForwardSplitter:
    """Time-aware splitter. Does NOT do random splits."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def folds(self) -> list[Fold]:
        return list(self.config.folds)
