"""Model layer.

The default base model is the regularized logistic regression
baseline. CatBoost is opt-in via the ``catboost.enabled`` config flag
and is only enabled when the generated knockout manifest reaches the
documented minimum sample size (see ``configs/models.yaml``).
"""

from football_advance_predictor.models.calibration.calibrator import Calibrator
from football_advance_predictor.models.catboost_model.catboost_model import (
    CatBoostConfig,
    CatBoostModel,
)
from football_advance_predictor.models.elo_model.elo_model import EloModel
from football_advance_predictor.models.logistic_baseline import (
    LogisticRegressionBaseline,
    LogisticRegressionBaselineConfig,
)
from football_advance_predictor.models.market_model.market_model import MarketModel
from football_advance_predictor.models.registry.registry import ModelArtifact, ModelRegistry
from football_advance_predictor.models.stacking.stacker import (
    StackingConfig,
    StackingModel,
)

__all__ = [
    "Calibrator",
    "CatBoostConfig",
    "CatBoostModel",
    "EloModel",
    "LogisticRegressionBaseline",
    "LogisticRegressionBaselineConfig",
    "MarketModel",
    "ModelArtifact",
    "ModelRegistry",
    "StackingConfig",
    "StackingModel",
]
