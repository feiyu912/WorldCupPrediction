"""Provider interfaces (Protocols) for the data ingestion layer.

Adapters implementing these protocols are pluggable: local CSV/JSON
implementations for the demo, skeleton implementations for external
paid providers, and an optional StatsBomb-style event provider.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from football_advance_predictor.schemas.availability import AvailabilityIn
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn
from football_advance_predictor.schemas.odds import MarketOddsIn


@runtime_checkable
class MatchDataProvider(Protocol):
    """Provider interface for historical match records and results."""

    name: str

    def fetch_matches(self, **kwargs: Any) -> list[MatchIn]:
        """Return all matches known to the provider."""
        ...

    def fetch_results(self, **kwargs: Any) -> list[MatchResultIn]:
        """Return verified match results (post-match labels)."""
        ...

    def fetch_teams(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return team metadata (name, FIFA code, aliases, etc.)."""
        ...


@runtime_checkable
class OddsProvider(Protocol):
    """Provider interface for market odds snapshots."""

    name: str

    def fetch_odds(self, **kwargs: Any) -> list[MarketOddsIn]:
        """Return all market odds snapshots known to the provider."""
        ...


@runtime_checkable
class AvailabilityProvider(Protocol):
    """Provider interface for player availability / lineup records."""

    name: str

    def fetch_availability(self, **kwargs: Any) -> list[AvailabilityIn]:
        """Return all availability records known to the provider."""
        ...
