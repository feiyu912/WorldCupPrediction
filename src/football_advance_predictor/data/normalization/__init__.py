"""Normalization utilities: team-name resolution, alias mapping."""

from football_advance_predictor.data.normalization.team_resolver import (
    TeamNameResolver,
    TeamResolutionError,
)

__all__ = ["TeamNameResolver", "TeamResolutionError"]
