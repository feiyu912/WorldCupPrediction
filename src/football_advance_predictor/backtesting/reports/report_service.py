"""Generate backtest reports.

A report includes:

- per-fold metrics (log loss, brier, accuracy, AUC, coverage, market comparison),
- aggregate metrics across folds,
- reliability plot,
- CSV/Parquet export of per-prediction scores.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from football_advance_predictor.backtesting.metrics.evaluation import (
    accuracy,
    brier_score,
    compute_reliability_table,
    expected_calibration_error,
    log_loss,
    roc_auc,
)
from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestReport:
    run_id: str
    model_version: str
    created_at: datetime
    summary: dict[str, Any]
    per_fold: list[dict[str, Any]]
    per_prediction_csv: str | None
    per_prediction_parquet: str | None
    reliability_plot: str | None


@dataclass
class _Aggregate:
    rows: list[dict[str, Any]] = field(default_factory=list)


class BacktestReportService:
    """Build a backtest report from per-fold prediction records.

    Each input row is a dict with at least:

    - ``fold`` (str)
    - ``home_advance_probability`` (float)
    - ``market_probability`` (float | None)
    - ``elo_probability`` (float | None)
    - ``home_advances`` (int)
    - ``cutoff_time`` (datetime)
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_report(
        self,
        *,
        model_version: str,
        records: list[dict[str, Any]],
        clear_lean_min: float = 0.62,
        reliability_plot: str | Path | None = None,
    ) -> BacktestReport:
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        created_at = datetime.now(tz=UTC)
        df = pd.DataFrame.from_records(records)
        per_fold: list[dict[str, Any]] = []
        for fold_name, group in df.groupby("fold"):
            y = group["home_advances"].astype(int).to_numpy()
            model_p = group["home_advance_probability"].astype(float).to_numpy()
            market_p = group.get("market_probability")
            elo_p = group.get("elo_probability")
            fold_metrics: dict[str, Any] = {
                "fold": fold_name,
                "n_test": len(group),
            }
            if len(y) == 0:
                per_fold.append(fold_metrics)
                continue
            fold_metrics["log_loss"] = log_loss(model_p, y)
            fold_metrics["brier_score"] = brier_score(model_p, y)
            fold_metrics["accuracy"] = accuracy(model_p, y)
            # roc_auc returns NaN for degenerate inputs; no try/except needed.
            fold_metrics["roc_auc"] = roc_auc(model_p, y)
            fold_metrics["ece"] = expected_calibration_error(model_p, y)

            clear_mask = (model_p >= clear_lean_min) | (model_p <= 1.0 - clear_lean_min)
            fold_metrics["coverage_clear_lean"] = float(clear_mask.mean())
            if clear_mask.any():
                fold_metrics["accuracy_clear_lean"] = accuracy(model_p[clear_mask], y[clear_mask])
            else:
                fold_metrics["accuracy_clear_lean"] = None

            if market_p is not None:
                m = market_p.fillna(0.5).to_numpy()
                fold_metrics["log_loss_market"] = log_loss(m, y)
                fold_metrics["brier_market"] = brier_score(m, y)
            if elo_p is not None:
                e = elo_p.fillna(0.5).to_numpy()
                fold_metrics["log_loss_elo"] = log_loss(e, y)
                fold_metrics["brier_elo"] = brier_score(e, y)

            fold_metrics["reliability"] = compute_reliability_table(model_p, y)
            per_fold.append(fold_metrics)

        summary = _aggregate_summary(per_fold)
        csv_path = self.output_dir / f"{run_id}_predictions.csv"
        parquet_path = self.output_dir / f"{run_id}_predictions.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
        json_path = self.output_dir / f"{run_id}_summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run_id,
                    "model_version": model_version,
                    "created_at": created_at.isoformat(),
                    "summary": summary,
                    "per_fold": per_fold,
                },
                f,
                indent=2,
                sort_keys=True,
                default=str,
            )
        reliability_path = str(reliability_plot) if reliability_plot else None
        return BacktestReport(
            run_id=run_id,
            model_version=model_version,
            created_at=created_at,
            summary=summary,
            per_fold=per_fold,
            per_prediction_csv=str(csv_path),
            per_prediction_parquet=str(parquet_path),
            reliability_plot=reliability_path,
        )


def _aggregate_summary(per_fold: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-fold metrics into a summary dict."""
    keys = [
        "log_loss",
        "brier_score",
        "accuracy",
        "roc_auc",
        "coverage_clear_lean",
        "accuracy_clear_lean",
        "log_loss_market",
        "brier_market",
        "log_loss_elo",
        "brier_elo",
        "ece",
    ]
    summary: dict[str, Any] = {"n_folds": len(per_fold)}
    for key in keys:
        values = [f[key] for f in per_fold if f.get(key) is not None and not _is_nan(f.get(key))]
        if not values:
            summary[key] = None
            continue
        summary[key] = sum(values) / len(values)
    return summary


def _is_nan(value: Any) -> bool:
    import math

    return isinstance(value, float) and math.isnan(value)
