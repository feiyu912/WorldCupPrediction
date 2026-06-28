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
        config = cls(folds=folds)
        config._validate()
        return config

    def _validate(self) -> None:
        """Validate fold contiguity: train < validation < test.

        Raises:
            ValueError: If any fold violates the chronological order or
                has a non-positive-length window.
        """
        for fold in self.folds:
            if fold.train_start >= fold.train_end:
                raise ValueError(
                    f"Fold {fold.name!r}: train_start must be < train_end "
                    f"(got {fold.train_start.isoformat()} / {fold.train_end.isoformat()})."
                )
            if fold.train_end >= fold.validation_start:
                raise ValueError(
                    f"Fold {fold.name!r}: train_end must be < validation_start "
                    f"(got {fold.train_end.isoformat()} / {fold.validation_start.isoformat()})."
                )
            if fold.validation_start >= fold.validation_end:
                raise ValueError(
                    f"Fold {fold.name!r}: validation_start must be < validation_end."
                )
            if fold.validation_end > fold.test_start:
                raise ValueError(
                    f"Fold {fold.name!r}: validation_end must be <= test_start "
                    f"(got {fold.validation_end.isoformat()} / {fold.test_start.isoformat()})."
                )
            if fold.test_start >= fold.test_end:
                raise ValueError(
                    f"Fold {fold.name!r}: test_start must be < test_end."
                )


class WalkForwardSplitter:
    """Time-aware splitter. Does NOT do random splits."""

    def __init__(self, config: WalkForwardConfig) -> None:
        self.config = config

    def folds(self) -> list[Fold]:
        return list(self.config.folds)
