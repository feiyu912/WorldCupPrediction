"""Pydantic schemas for feature snapshots and feature build requests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict


class FeatureBuildRequest(BaseModel):
    """Request to build a time-frozen feature snapshot."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    cutoff_time: datetime
    feature_version: str = "v1"


class FeatureSnapshotOut(BaseModel):
    """Output schema for a feature snapshot."""

    model_config = ConfigDict(from_attributes=True)

    feature_snapshot_id: str
    match_id: str
    cutoff_time: datetime
    feature_version: str
    features_json: dict[str, Any]
    source_data_max_timestamp: datetime
    immutable_hash: str
