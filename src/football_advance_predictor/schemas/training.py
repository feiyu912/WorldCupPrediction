"""Pydantic schemas for training and backtest requests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TrainingRequest(BaseModel):
    """Request to train a model version."""

    model_config = ConfigDict(extra="forbid")

    model_config_name: str
    training_window: dict[str, str] = Field(
        ..., description="Mapping with 'start' and 'end' ISO dates."
    )
    validation_window: dict[str, str] = Field(
        ..., description="Mapping with 'start' and 'end' ISO dates."
    )
    test_window: dict[str, str] = Field(
        ..., description="Mapping with 'start' and 'end' ISO dates."
    )


class BacktestRequest(TrainingRequest):
    """Request to run a backtest across multiple folds."""

    folds: list[dict[str, Any]] = Field(default_factory=list)


class ModelRunOut(BaseModel):
    """Output schema for a stored model run."""

    model_config = ConfigDict(from_attributes=True)

    model_run_id: str
    model_type: str
    model_version: str
    training_start: datetime
    training_end: datetime
    validation_start: datetime
    validation_end: datetime
    test_start: datetime
    test_end: datetime
    feature_version: str
    hyperparameters_json: dict[str, Any]
    metrics_json: dict[str, Any]
    artifact_path: str
