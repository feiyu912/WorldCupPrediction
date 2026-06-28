"""Data ingestion, normalization, and warehouse layer."""

from football_advance_predictor.data.adapters.base import (
    AvailabilityProvider,
    MatchDataProvider,
    OddsProvider,
)

__all__ = ["AvailabilityProvider", "MatchDataProvider", "OddsProvider"]
