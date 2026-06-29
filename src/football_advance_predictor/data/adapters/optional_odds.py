"""Optional historical-odds provider interface.

Historical odds are an OPTIONAL enrichment. They are disabled by
default and only active when an environment variable is set. The
default MVP never requires historical odds: market features are
marked as missing and the market branch of the stacker is skipped.

When a key IS set, the provider should be configured to retrieve a
single T-24h snapshot per match (the most reproducible cutoff).
Multi-snapshot odds-movement features are deferred to a later phase
and never fabricated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.schemas.odds import MarketOddsIn

logger = get_logger(__name__)


@dataclass(frozen=True)
class HistoricalOddsProviderConfig:
    """Configuration for the optional historical-odds provider."""

    api_key: str
    base_url: str
    default_cutoff_hours_before_kickoff: int = 24
    min_bookmakers: int = 1
    timeout_seconds: int = 30
    source_name: str = "optional_historical_odds"


class HistoricalOddsProvider(Protocol):
    """Optional historical-odds provider protocol.

    The provider fetches a single reproducible cutoff snapshot per
    match (default T-24h). Multi-snapshot movement features are
    intentionally NOT part of this protocol.
    """

    name: str

    def fetch_odds(self, **kwargs: Any) -> list[MarketOddsIn]: ...


class NoOpHistoricalOddsProvider:
    """Default no-op provider when no API key is configured.

    Always returns an empty list. The system is supposed to detect
    this and skip the market branch of the stacker.
    """

    name = "noop_historical_odds"

    def __init__(self) -> None:
        logger.info("Historical odds provider disabled (no API key set).")

    def fetch_odds(self, **kwargs: Any) -> list[MarketOddsIn]:
        return []


def build_default_optional_provider(
    env_var: str = "EXTERNAL_ODDS_API_KEY",
) -> HistoricalOddsProvider:
    """Build the default optional provider based on environment.

    When ``env_var`` is set, this function returns a skeleton provider
    that returns an empty list. Real implementation is a future-facing
    task; this returns an explicit "not implemented" stub that
    fails closed (does not call the network).
    """
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        return NoOpHistoricalOddsProvider()
    logger.warning(
        "Historical odds API key set but no default provider is wired up; "
        "failing closed with a no-op provider. Implement a concrete adapter."
    )
    return _SkeletonAuthedNoFetchProvider(api_key)


class _SkeletonAuthedNoFetchProvider:
    """Skeleton provider that authenticates but does not fetch.

    Returning no records (rather than calling the network) is the
    safe default until a concrete adapter is implemented.
    """

    name = "skeleton_optional_odds"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def fetch_odds(self, **kwargs: Any) -> list[MarketOddsIn]:
        logger.warning(
            "Skeleton provider; returning empty list. "
            "Implement the actual HTTP call in a concrete adapter."
        )
        return []


__all__ = [
    "HistoricalOddsProvider",
    "HistoricalOddsProviderConfig",
    "NoOpHistoricalOddsProvider",
    "build_default_optional_provider",
]
