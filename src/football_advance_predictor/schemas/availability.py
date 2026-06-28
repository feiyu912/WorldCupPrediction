"""Pydantic schemas for player availability snapshots."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class AvailabilityIn(BaseModel):
    """Input schema for a player availability record."""

    model_config = ConfigDict(extra="forbid")

    availability_id: str | None = None
    match_id: str
    team_id: str
    player_id: str | None = None
    role: str = Field(default="unknown")
    availability_status: str = Field(
        ...,
        description=(
            "One of: available, questionable, doubtful, suspended, "
            "confirmed_out, lineup_confirmed, unknown"
        ),
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    published_at: datetime
    cutoff_eligible: bool = True
    source: str = "local"
    raw_text: str | None = None

    # Lineage
    source_name: str = "local"
    source_record_id: str | None = None
    source_url: str | None = None
    observed_at: datetime
    ingested_at: datetime
    effective_at: datetime
    raw_payload_hash: str
    source_version: str | None = None


class AvailabilityOut(AvailabilityIn):
    """Output schema for an availability record."""

    model_config = ConfigDict(from_attributes=True)
    availability_id: str
