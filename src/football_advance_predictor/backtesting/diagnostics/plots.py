"""Plotting helpers for backtest reports."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt

from football_advance_predictor.backtesting.metrics.evaluation import (
    compute_reliability_table,
)


def plot_reliability_curve(
    probs: Iterable[float],
    y: Iterable[int],
    output: str | Path,
    *,
    n_bins: int = 10,
    title: str = "Reliability",
) -> Path:
    """Render a reliability curve (calibration plot) to ``output``."""
    rows = compute_reliability_table(probs, y, n_bins=n_bins)
    xs = [r["bin_center"] for r in rows if r["count"] > 0 and r["observed_frequency"] is not None]
    ys = [r["observed_frequency"] for r in rows if r["count"] > 0 and r["observed_frequency"] is not None]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="ideal")
    ax.scatter(xs, ys, s=30, label="model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title(title)
    ax.legend()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    return output
