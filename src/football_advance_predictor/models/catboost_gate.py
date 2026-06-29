"""Multi-fold CatBoost gate.

CatBoost may be trained, but it becomes the default base learner only
if it beats the Logistic baseline on at least N walk-forward folds on
the primary metrics (Log Loss, Brier Score, calibration curve,
reliability bins, coverage) above configurable minimum practical
improvement thresholds.

Sample count alone never auto-deploys CatBoost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CatBoostGateConfig:
    """Configuration for the CatBoost gate."""

    enabled_in_models_yaml: bool = False
    min_samples_to_enable: int = 200
    min_folds_catboost_beats_logistic: int = 2
    min_log_loss_improvement: float = 0.005
    min_brier_improvement: float = 0.005
    min_coverage_improvement: float = 0.0
    require_calibration_improvement: bool = True
    require_reliability_improvement: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CatBoostGateConfig":
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


@dataclass
class FoldComparison:
    """Per-fold comparison of CatBoost vs Logistic on the primary metrics."""

    fold: str
    log_loss_logistic: float
    log_loss_catboost: float
    brier_logistic: float
    brier_catboost: float
    coverage_logistic: float
    coverage_catboost: float
    calibration_logistic: float | None
    calibration_catboost: float | None
    reliability_logistic: float | None
    reliability_catboost: float | None
    catboost_beats: bool
    meets_practical_threshold: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold": self.fold,
            "log_loss_logistic": self.log_loss_logistic,
            "log_loss_catboost": self.log_loss_catboost,
            "brier_logistic": self.brier_logistic,
            "brier_catboost": self.brier_catboost,
            "coverage_logistic": self.coverage_logistic,
            "coverage_catboost": self.coverage_catboost,
            "calibration_logistic": self.calibration_logistic,
            "calibration_catboost": self.calibration_catboost,
            "reliability_logistic": self.reliability_logistic,
            "reliability_catboost": self.reliability_catboost,
            "catboost_beats": self.catboost_beats,
            "meets_practical_threshold": self.meets_practical_threshold,
        }


@dataclass
class CatBoostGateDecision:
    """The final decision after evaluating the gate."""

    config: CatBoostGateConfig
    fold_comparisons: list[FoldComparison] = field(default_factory=list)
    n_folds_evaluated: int = 0
    n_folds_catboost_beats: int = 0
    catboost_becomes_default: bool = False
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "enabled_in_models_yaml": self.config.enabled_in_models_yaml,
                "min_samples_to_enable": self.config.min_samples_to_enable,
                "min_folds_catboost_beats_logistic": self.config.min_folds_catboost_beats_logistic,
                "min_log_loss_improvement": self.config.min_log_loss_improvement,
                "min_brier_improvement": self.config.min_brier_improvement,
                "min_coverage_improvement": self.config.min_coverage_improvement,
                "require_calibration_improvement": self.config.require_calibration_improvement,
                "require_reliability_improvement": self.config.require_reliability_improvement,
            },
            "fold_comparisons": [c.to_dict() for c in self.fold_comparisons],
            "n_folds_evaluated": self.n_folds_evaluated,
            "n_folds_catboost_beats": self.n_folds_catboost_beats,
            "catboost_becomes_default": self.catboost_becomes_default,
            "reason": self.reason,
        }


def evaluate_gate(
    fold_metrics: list[dict[str, Any]],
    config: CatBoostGateConfig,
) -> CatBoostGateDecision:
    """Apply the multi-fold gate.

    Args:
        fold_metrics: List of per-fold dicts. Each dict must include:

            - ``fold`` (str)
            - ``log_loss_logistic``, ``log_loss_catboost`` (float)
            - ``brier_logistic``, ``brier_catboost`` (float)
            - ``coverage_logistic``, ``coverage_catboost`` (float)
            - ``calibration_logistic``, ``calibration_catboost`` (float|None)
            - ``reliability_logistic``, ``reliability_catboost`` (float|None)

        config: :class:`CatBoostGateConfig`.
    """
    decision = CatBoostGateDecision(config=config)
    for raw in fold_metrics:
        ll_l = float(raw["log_loss_logistic"])
        ll_c = float(raw["log_loss_catboost"])
        b_l = float(raw["brier_logistic"])
        b_c = float(raw["brier_catboost"])
        cov_l = float(raw["coverage_logistic"])
        cov_c = float(raw["coverage_catboost"])
        cal_l = raw.get("calibration_logistic")
        cal_c = raw.get("calibration_catboost")
        rel_l = raw.get("reliability_logistic")
        rel_c = raw.get("reliability_catboost")
        cat_wins = (ll_c + config.min_log_loss_improvement < ll_l) and (
            b_c + config.min_brier_improvement < b_l
        )
        meets_threshold = cat_wins and (
            cov_c + config.min_coverage_improvement >= cov_l
        )
        if config.require_calibration_improvement and cal_l is not None and cal_c is not None:
            meets_threshold = meets_threshold and cal_c <= cal_l
        if config.require_reliability_improvement and rel_l is not None and rel_c is not None:
            meets_threshold = meets_threshold and rel_c <= rel_l
        decision.fold_comparisons.append(
            FoldComparison(
                fold=str(raw.get("fold", "?")),
                log_loss_logistic=ll_l,
                log_loss_catboost=ll_c,
                brier_logistic=b_l,
                brier_catboost=b_c,
                coverage_logistic=cov_l,
                coverage_catboost=cov_c,
                calibration_logistic=cal_l,
                calibration_catboost=cal_c,
                reliability_logistic=rel_l,
                reliability_catboost=rel_c,
                catboost_beats=cat_wins,
                meets_practical_threshold=meets_threshold,
            )
        )

    decision.n_folds_evaluated = len(decision.fold_comparisons)
    decision.n_folds_catboost_beats = sum(
        1 for c in decision.fold_comparisons if c.meets_practical_threshold
    )
    if not config.enabled_in_models_yaml:
        decision.catboost_becomes_default = False
        decision.reason = "catboost_disabled_in_models_yaml"
    elif decision.n_folds_catboost_beats < config.min_folds_catboost_beats_logistic:
        decision.catboost_becomes_default = False
        decision.reason = (
            f"only {decision.n_folds_catboost_beats} of "
            f"{decision.n_folds_evaluated} folds beat logistic by the "
            f"required margins (need {config.min_folds_catboost_beats_logistic})"
        )
    else:
        decision.catboost_becomes_default = True
        decision.reason = (
            f"catboost beats logistic on "
            f"{decision.n_folds_catboost_beats} of {decision.n_folds_evaluated} folds"
        )
    logger.info(
        "CatBoost gate decision",
        extra={
            "becomes_default": decision.catboost_becomes_default,
            "reason": decision.reason,
        },
    )
    return decision
