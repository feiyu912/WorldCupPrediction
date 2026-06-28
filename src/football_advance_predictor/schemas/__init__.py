"""Pydantic schemas (request/response and internal data contracts)."""

from football_advance_predictor.schemas.availability import (
    AvailabilityIn,
    AvailabilityOut,
)
from football_advance_predictor.schemas.features import (
    FeatureBuildRequest,
    FeatureSnapshotOut,
)
from football_advance_predictor.schemas.matches import (
    MatchIn,
    MatchOut,
    MatchResultIn,
    MatchResultOut,
)
from football_advance_predictor.schemas.odds import (
    MarketOddsIn,
    MarketOddsOut,
)
from football_advance_predictor.schemas.predictions import (
    ConfidenceBand,
    EvaluationOut,
    PredictionOut,
    PredictionRequest,
    assign_confidence_band,
)
from football_advance_predictor.schemas.training import (
    BacktestRequest,
    ModelRunOut,
    TrainingRequest,
)

__all__ = [
    "AvailabilityIn",
    "AvailabilityOut",
    "BacktestRequest",
    "ConfidenceBand",
    "EvaluationOut",
    "FeatureBuildRequest",
    "FeatureSnapshotOut",
    "MarketOddsIn",
    "MarketOddsOut",
    "MatchIn",
    "MatchOut",
    "MatchResultIn",
    "MatchResultOut",
    "ModelRunOut",
    "PredictionOut",
    "PredictionRequest",
    "TrainingRequest",
    "assign_confidence_band",
]
