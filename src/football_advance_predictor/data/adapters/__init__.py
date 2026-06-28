"""Provider adapters for matches, odds, and availability data."""

from football_advance_predictor.data.adapters.availability import LocalAvailabilityProvider
from football_advance_predictor.data.adapters.base import (
    AvailabilityProvider,
    MatchDataProvider,
    OddsProvider,
)
from football_advance_predictor.data.adapters.local_matches import LocalHistoricalResultsProvider
from football_advance_predictor.data.adapters.local_odds import LocalOddsProvider
from football_advance_predictor.data.adapters.skeletons import (
    SkeletonExternalOddsProvider,
    SkeletonFootballDataProvider,
)
from football_advance_predictor.data.adapters.statsbomb_local import (
    StatsBombLocalProvider,
)

__all__ = [
    "AvailabilityProvider",
    "LocalAvailabilityProvider",
    "LocalHistoricalResultsProvider",
    "LocalOddsProvider",
    "MatchDataProvider",
    "OddsProvider",
    "SkeletonExternalOddsProvider",
    "SkeletonFootballDataProvider",
    "StatsBombLocalProvider",
]
