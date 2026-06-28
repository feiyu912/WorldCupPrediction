"""Model layer: Elo, market, CatBoost, stacking, calibration, registry."""

from football_advance_predictor.models.calibration.calibrator import Calibrator
from football_advance_predictor.models.catboost_model.catboost_model import CatBoostModel
from football_advance_predictor.models.elo_model.elo_model import EloModel
from football_advance_predictor.models.market_model.market_model import MarketModel
from football_advance_predictor.models.registry.registry import ModelArtifact, ModelRegistry
from football_advance_predictor.models.stacking.stacker import StackingModel

__all__ = [
    "Calibrator",
    "CatBoostModel",
    "EloModel",
    "MarketModel",
    "ModelArtifact",
    "ModelRegistry",
    "StackingModel",
]
