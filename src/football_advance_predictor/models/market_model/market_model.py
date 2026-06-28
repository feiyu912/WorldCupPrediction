"""Market model wrapper that exposes a unified ``predict_proba`` API.

Returns ``None`` when no valid two-way market is available, and never
fabricates a probability. The wrapper is fit-free: it queries the
ingested odds snapshots at prediction time.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_advance_predictor.core.time import to_utc
from football_advance_predictor.db.models import MarketOddsSnapshot
from football_advance_predictor.features.market.consensus import (
    MarketAdvanceProbabilityModel,
    MarketConsensus,
)


class MarketModel:
    """A market consensus predictor.

    Args:
        session: SQLAlchemy session used to query odds at predict time.
        min_bookmakers: Minimum number of contributing bookmakers.
    """

    def __init__(self, session: Session, *, min_bookmakers: int = 2) -> None:
        """Market consensus predictor.

        Args:
            session: SQLAlchemy session used to query odds at predict time.
            min_bookmakers: Minimum number of contributing bookmakers.
                Default 2; pass 1 only when your data has a single
                bookmaker.
        """
        self.session = session
        self.min_bookmakers = max(1, int(min_bookmakers))

    def predict_proba(
        self, *, match_id: str, as_of_time: datetime
    ) -> float | None:
        """Return P(home advances) from market consensus at ``as_of_time``.

        Returns ``None`` if no valid two-way market exists.
        """
        consensus = self.consensus_at(match_id=match_id, as_of_time=as_of_time)
        if consensus is None:
            return None
        return consensus.home_advance_probability

    def consensus_at(
        self, *, match_id: str, as_of_time: datetime
    ) -> MarketConsensus | None:
        cutoff_utc = to_utc(as_of_time)
        stmt = (
            select(MarketOddsSnapshot)
            .where(MarketOddsSnapshot.match_id == match_id)
            .where(MarketOddsSnapshot.captured_at <= cutoff_utc)
        )
        snapshots: Iterable[MarketOddsSnapshot] = list(self.session.scalars(stmt))
        model = MarketAdvanceProbabilityModel(
            snapshots, min_bookmakers=self.min_bookmakers
        )
        return model.consensus_at(cutoff_utc)
