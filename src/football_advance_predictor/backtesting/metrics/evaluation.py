"""Evaluation metrics for the backtest.

Re-implementations of common metrics that operate on plain
arrays, so we don't have to depend on optional sklearn APIs that may
move.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def log_loss(probs: Iterable[float], y: Iterable[int], eps: float = 1e-12) -> float:
    """Mean binary cross-entropy.

    Args:
        probs: Predicted P(y=1).
        y: True labels (0 or 1).
        eps: Clipping to avoid log(0).
    """
    total = 0.0
    n = 0
    for p, t in zip(probs, y, strict=False):
        p = max(min(p, 1 - eps), eps)
        total += -(t * math.log(p) + (1 - t) * math.log(1 - p))
        n += 1
    if n == 0:
        return float("nan")
    return total / n


def brier_score(probs: Iterable[float], y: Iterable[int]) -> float:
    """Mean squared error between probabilities and binary labels."""
    total = 0.0
    n = 0
    for p, t in zip(probs, y, strict=False):
        total += (p - t) ** 2
        n += 1
    if n == 0:
        return float("nan")
    return total / n


def accuracy(probs: Iterable[float], y: Iterable[int], threshold: float = 0.5) -> float:
    """Binary accuracy at ``threshold``."""
    n = 0
    correct = 0
    for p, t in zip(probs, y, strict=False):
        predicted = 1 if p >= threshold else 0
        if predicted == t:
            correct += 1
        n += 1
    if n == 0:
        return float("nan")
    return correct / n


def roc_auc(probs: Iterable[float], y: Iterable[int]) -> float:
    """ROC AUC via the Wilcoxon-Mann-Whitney statistic.

    Returns ``float('nan')`` if the input is degenerate (only one class
    or all probabilities are identical).
    """
    pairs = list(zip(probs, y))
    n_pos = sum(1 for _, t in pairs if t == 1)
    n_neg = sum(1 for _, t in pairs if t == 0)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    unique_probs = {p for p, _ in pairs}
    if len(unique_probs) <= 1:
        return float("nan")
    # Sort by probability; assign average rank to ties.
    pairs.sort(key=lambda x: x[0])
    pos_rank_sum = 0.0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            if pairs[k][1] == 1:
                pos_rank_sum += avg_rank
        i = j
    # WMW formula: subtract the minimum possible sum for n_pos items.
    min_sum = n_pos * (n_pos + 1) / 2.0
    return (pos_rank_sum - min_sum) / (n_pos * n_neg)


def reliability_bins(
    probs: Iterable[float],
    y: Iterable[int],
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """Bin predictions by predicted probability.

    Returns a list of tuples ``(bin_center, observed_frequency, count)``.
    """
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, t in zip(probs, y, strict=False):
        idx = min(n_bins - 1, int(p * n_bins))
        buckets[idx].append((p, t))
    out: list[tuple[float, float, int]] = []
    for i, bucket in enumerate(buckets):
        if not bucket:
            out.append(((i + 0.5) / n_bins, float("nan"), 0))
            continue
        avg_p = sum(p for p, _ in bucket) / len(bucket)
        avg_y = sum(t for _, t in bucket) / len(bucket)
        out.append((avg_p, avg_y, len(bucket)))
    return out


def compute_reliability_table(
    probs: Iterable[float], y: Iterable[int], n_bins: int = 10
) -> list[dict[str, float]]:
    """Return the reliability table as a list of dicts."""
    rows: list[dict[str, float]] = []
    for bin_center, observed, count in reliability_bins(probs, y, n_bins=n_bins):
        rows.append(
            {
                "bin_center": float(bin_center),
                "observed_frequency": float(observed) if not math.isnan(observed) else None,
                "count": float(count),
            }
        )
    return rows


def expected_calibration_error(
    probs: Iterable[float], y: Iterable[int], n_bins: int = 10
) -> float:
    """Expected Calibration Error (ECE) with equal-width bins."""
    rows = compute_reliability_table(probs, y, n_bins=n_bins)
    n_total = sum(r["count"] for r in rows)
    if n_total == 0:
        return float("nan")
    ece = 0.0
    for r in rows:
        if r["observed_frequency"] is None or r["count"] == 0:
            continue
        ece += (r["count"] / n_total) * abs(r["bin_center"] - r["observed_frequency"])
    return ece
