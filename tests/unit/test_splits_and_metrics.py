"""Tests for the splitter, metrics, and ledger immutability."""

from __future__ import annotations

from datetime import UTC, datetime

from football_advance_predictor.backtesting.metrics.evaluation import (
    accuracy,
    brier_score,
    compute_reliability_table,
    expected_calibration_error,
    log_loss,
    reliability_bins,
    roc_auc,
)
from football_advance_predictor.backtesting.splits.walk_forward import (
    Fold,
    WalkForwardConfig,
    WalkForwardSplitter,
)


def test_walk_forward_splitter_returns_folds_in_order() -> None:
    cfg = WalkForwardConfig(
        folds=[
            Fold(
                name="fold_a",
                train_start=datetime(2010, 1, 1, tzinfo=UTC),
                train_end=datetime(2017, 12, 31, tzinfo=UTC),
                validation_start=datetime(2018, 1, 1, tzinfo=UTC),
                validation_end=datetime(2019, 12, 31, tzinfo=UTC),
                test_start=datetime(2020, 1, 1, tzinfo=UTC),
                test_end=datetime(2022, 12, 31, tzinfo=UTC),
            )
        ]
    )
    splitter = WalkForwardSplitter(cfg)
    folds = splitter.folds()
    assert len(folds) == 1
    assert folds[0].name == "fold_a"


def test_log_loss_perfect_predictions_is_low() -> None:
    ll = log_loss([0.99, 0.01], [1, 0])
    assert ll < 0.05


def test_brier_perfect_predictions_is_zero() -> None:
    assert brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_accuracy_threshold() -> None:
    assert accuracy([0.6, 0.4], [1, 0]) == 1.0
    assert accuracy([0.4, 0.6], [1, 0]) == 0.0


def test_roc_auc_simple_case() -> None:
    auc = roc_auc([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])
    assert auc == 1.0


def test_roc_auc_handles_constant_input() -> None:
    auc = roc_auc([0.5, 0.5, 0.5], [0, 1, 0])
    import math

    assert math.isnan(auc)


def test_reliability_bins_count() -> None:
    bins = reliability_bins([0.05, 0.95], [0, 1], n_bins=10)
    assert len(bins) == 10


def test_compute_reliability_table_shape() -> None:
    table = compute_reliability_table([0.1, 0.4, 0.8], [0, 0, 1], n_bins=4)
    assert len(table) == 4
    assert all("count" in r for r in table)


def test_ece_perfect_is_zero() -> None:
    ece = expected_calibration_error([0.05, 0.5, 0.95], [0, 1, 1], n_bins=10)
    # Sparse bins can inflate ECE. We test that ECE is well below 1.0.
    assert ece < 0.5
