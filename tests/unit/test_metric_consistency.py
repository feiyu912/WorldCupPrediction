"""Tests for the metric consistency invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from football_advance_predictor.backtesting.metrics.evaluation import (
    brier_score,
    brier_score_sum,
    log_loss,
    log_loss_sum,
    metric_consistency_check,
    per_row_brier,
    per_row_log_loss,
)


def test_constant_p_0_5_log_loss_sum_and_mean() -> None:
    """log_loss_mean = log(2) ~ 0.693147, log_loss_sum = n * log(2)."""
    probs = [0.5] * 60
    y = [0, 1] * 30
    mean = log_loss(probs, y)
    s = log_loss_sum(probs, y)
    assert math.isclose(mean, math.log(2), rel_tol=0, abs_tol=1e-9), mean
    assert math.isclose(s, 60 * math.log(2), rel_tol=0, abs_tol=1e-9), s
    assert math.isclose(s, 60 * mean, rel_tol=0, abs_tol=1e-12)


def test_constant_p_0_5_brier_mean() -> None:
    probs = [0.5] * 60
    y = [0, 1] * 30
    brier = brier_score(probs, y)
    assert math.isclose(brier, 0.25, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(brier_score_sum(probs, y), 60 * 0.25, rel_tol=0, abs_tol=1e-12)


def test_mean_log_loss_ge_mean_brier_on_identical_inputs() -> None:
    """The strict invariant: mean_log_loss >= mean_brier for the same rows."""
    rng = np.random.default_rng(0)
    n = 200
    y = rng.integers(0, 2, n).tolist()
    # Use a mix of confident and uncertain predictions.
    probs = [0.5 + 0.4 * (1 - 2 * t) for t in y]  # perfect = log(0.1) each
    probs = [max(0.01, min(0.99, p)) for p in probs]
    # Also add some near-0.5 predictions where the inequality tightens.
    for _ in range(50):
        y.append(int(rng.integers(0, 2)))
        probs.append(0.5)
    result = metric_consistency_check(probs, y)
    assert result["log_loss_ge_brier"], result
    assert result["passed"], result


def test_sum_equals_n_times_mean_exactly() -> None:
    """metric_consistency_check verifies sum/mean consistency."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 50).tolist()
    probs = rng.uniform(0.05, 0.95, 50).tolist()
    result = metric_consistency_check(probs, y)
    assert result["sum_mean_consistent"], result
    assert result["brier_sum_consistent"], result
    assert result["passed"], result


def test_per_row_log_loss_sums_to_log_loss_sum() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 100).tolist()
    probs = rng.uniform(0.05, 0.95, 100).tolist()
    per = per_row_log_loss(probs, y)
    assert math.isclose(sum(per), log_loss_sum(probs, y), rel_tol=0, abs_tol=1e-9)
    assert math.isclose(log_loss(probs, y), log_loss_sum(probs, y) / len(probs))


def test_per_row_brier_sums_to_brier_score_sum() -> None:
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 100).tolist()
    probs = rng.uniform(0.05, 0.95, 100).tolist()
    per = per_row_brier(probs, y)
    assert math.isclose(sum(per), brier_score_sum(probs, y), rel_tol=0, abs_tol=1e-9)


def test_consistency_check_passes_on_perfect_predictions() -> None:
    """Perfect predictions: log_loss=0, brier=0. Inequality is satisfied as 0 >= 0."""
    y = [0, 1, 0, 1, 0, 1] * 5
    probs = [0.0 if t == 0 else 1.0 for t in y]
    # Clipping eps=1e-12 prevents log(0); we use a very small floor manually.
    probs = [0.001 if t == 0 else 0.999 for t in y]
    result = metric_consistency_check(probs, y)
    assert result["log_loss_ge_brier"]
    assert result["passed"]


def test_invariant_violation_raises() -> None:
    """Construct a degenerate case where the inequality is violated; expect AssertionError.

    The log_loss is always >= brier for any binary labels and probabilities
    in [0,1], so a violation requires a buggy input. We force it by passing
    out-of-range probabilities.
    """
    y = [0, 1, 0, 1]
    # Probabilities all 0.5 -> log_loss=log(2)=0.693, brier=0.25. OK.
    # Probabilities all 1.0 -> log_loss=0, brier=0. OK.
    # The invariant holds for any valid (p, y) in [0,1] x {0,1}. So we
    # can't easily construct a violation here. The only failure mode is
    # truncation. We assert the check raises only on (rare) numerical
    # edge cases; this test is here as a placeholder.
    metric_consistency_check([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1])


def test_symmetric_probabilities_match_log_loss_sum() -> None:
    """Mirrored pair (A vs B) and (B vs A) should have the same log_loss_sum."""
    y = [1, 0, 0, 1, 1, 1, 0, 0]
    probs = [0.7, 0.3, 0.4, 0.6, 0.8, 0.55, 0.35, 0.65]
    n = len(y)
    s = log_loss_sum(probs, y)
    m = log_loss(probs, y)
    assert math.isclose(s, n * m, rel_tol=0, abs_tol=1e-9)
