"""Pydantic schemas for matches and results."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MatchIn(BaseModel):
    """Input schema for ingesting a match record."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    kickoff_at: datetime
    competition_id: str
    stage: str = "unknown"
    season_or_year: str
    home_team_id: str
    away_team_id: str
    home_goals: int | None = None
    away_goals: int | None = None
    winner_team_id: str | None = None
    advancing_team_id: str | None = None
    neutral_venue: bool = False
    venue_name: str | None = None
    city: str | None = None
    country: str | None = None
    source: str = "local"


class MatchOut(MatchIn):
    """Output schema for a match record."""

    model_config = ConfigDict(from_attributes=True)


class MatchResultIn(BaseModel):
    """Input schema for a verified knockout result."""

    model_config = ConfigDict(extra="forbid")

    match_id: str
    final_status: str = Field(default="final", description="e.g. final, awarded, void")
    home_goals_90: int | None = None
    away_goals_90: int | None = None
    home_goals_et: int | None = None
    away_goals_et: int | None = None
    penalties_home: int | None = None
    penalties_away: int | None = None
    home_advances: bool
    result_verified_at: datetime


class MatchResultOut(MatchResultIn):
    """Output schema for a knockout result."""

    model_config = ConfigDict(from_attributes=True)
