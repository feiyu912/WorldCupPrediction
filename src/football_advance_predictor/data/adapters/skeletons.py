"""Skeleton external provider adapters.

These are intentionally inert by default. They document the contract
and refuse to perform network requests without explicit configuration
and an environment-provided API key.
"""

from __future__ import annotations

import os
from typing import Any

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.data.adapters.base import (
    AvailabilityProvider,
    MatchDataProvider,
    OddsProvider,
)
from football_advance_predictor.schemas.availability import AvailabilityIn
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn
from football_advance_predictor.schemas.odds import MarketOddsIn

logger = get_logger(__name__)


class SkeletonExternalOddsProvider(OddsProvider):
    """Skeleton external odds provider. Returns no records by default.

    To activate, set ``EXTERNAL_ODDS_API_KEY`` and implement the HTTP
    client. The adapter deliberately fails closed to avoid surprising
    network calls in tests.
    """

    name = "skeleton_external_odds"

    def __init__(self) -> None:
        self.api_key = os.environ.get("EXTERNAL_ODDS_API_KEY", "")

    def fetch_odds(self, **kwargs: Any) -> list[MarketOddsIn]:
        if not self.api_key:
            logger.info("SkeletonExternalOddsProvider: no API key set; returning empty list.")
            return []
        # TODO: implement external odds fetching. The data contract MUST:
        # - map response to MarketOddsIn,
        # - include captured_at, published_at, ingested_at, effective_at,
        # - include raw_payload_hash for immutability.
        raise NotImplementedError(
            "External odds provider not implemented. See TODO in skeleton."
        )


class SkeletonFootballDataProvider(MatchDataProvider, AvailabilityProvider):
    """Skeleton football-data.org style provider. Returns no records by default."""

    name = "skeleton_football_data"

    def __init__(self) -> None:
        self.token = os.environ.get("EXTERNAL_FOOTBALLDATA_TOKEN", "")

    # --- MatchDataProvider ---
    def fetch_matches(self, **kwargs: Any) -> list[MatchIn]:
        if not self.token:
            logger.info("SkeletonFootballDataProvider: no token; returning empty list.")
            return []
        raise NotImplementedError("External match fetch not implemented.")

    def fetch_results(self, **kwargs: Any) -> list[MatchResultIn]:
        if not self.token:
            return []
        raise NotImplementedError("External results fetch not implemented.")

    def fetch_teams(self, **kwargs: Any) -> list[dict[str, Any]]:
        if not self.token:
            return []
        raise NotImplementedError("External teams fetch not implemented.")

    # --- AvailabilityProvider ---
    def fetch_availability(self, **kwargs: Any) -> list[AvailabilityIn]:
        if not self.token:
            return []
        raise NotImplementedError("External availability fetch not implemented.")
