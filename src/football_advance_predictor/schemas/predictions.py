"""Pydantic schemas for predictions and evaluations."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ConfidenceBand(str, Enum):
    """Confidence band categories for predictions."""

    CLEAR_LEAN = "clear_lean"
    SLIGHT_LEAN = "slight_lean"
    NEAR_COIN_FLIP = "near_coin_flip"


def assign_confidence_band(
    home_advance_prob: float,
    *,
    clear_lean_min: float = 0.62,
    slight_lean_min: float = 0.55,
) -> ConfidenceBand:
    """Assign a confidence band based on the calibrated home advance probability.

    Args:
        home_advance_prob: Calibrated P(home_advances) in [0, 1].
        clear_lean_min: Lower bound for ``clear_lean``.
        slight_lean_min: Lower bound for ``slight_lean``.

    Returns:
        The ``ConfidenceBand`` for the prediction.

    Raises:
        ValueError: If the probability is outside ``[0, 1]``.
    """
    if not 0.0 <= home_advance_prob <= 1.0:
        raise ValueError("home_advance_prob must be in [0, 1]")

    if home_advance_prob >= clear_lean_min or home_advance_prob <= 1.0 - clear_lean_min:
        return ConfidenceBand.CLEAR_LEAN
    if home_advance_prob >= slight_lean_min or home_advance_prob <= 1.0 - slight_lean_min:
        return ConfidenceBand.SLIGHT_LEAN
    return ConfidenceBand.NEAR_COIN_FLIP


class PredictionRequest(BaseModel):
    """Request to generate a prediction for a specific match and cutoff."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    cutoff_time: datetime
    model_version: str


class PredictionOut(BaseModel):
    """Output schema for a stored prediction."""

    model_config = ConfigDict(from_attributes=True)

    prediction_id: str
    match_id: str
    cutoff_time: datetime
    model_version: str
    feature_snapshot_id: str | None = None

    market_probability: float | None = None
    elo_probability: float | None = None
    catboost_probability: float | None = None
    stacked_probability: float | None = None
    calibrated_probability: float

    home_advance_probability: float
    away_advance_probability: float
    predicted_advancer_id: str
    confidence_band: str
    status: str
    explanation_payload: dict[str, Any] = Field(default_factory=dict)
    immutable_hash: str
    created_at: datetime


class EvaluationOut(BaseModel):
    """Output schema for a stored evaluation record."""

    model_config = ConfigDict(from_attributes=True)

    evaluation_id: str
    prediction_id: str
    actual_home_advances: bool
    log_loss: float
    brier_score: float
    correct_classification: bool
    evaluated_at: datetime
