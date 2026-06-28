"""Market consensus model.

For a two-way ``to advance`` market:

    raw_home = 1 / home_decimal_odds
    raw_away = 1 / away_decimal_odds
    normalized_home = raw_home / (raw_home + raw_away)

The overround is ``raw_home + raw_away - 1``. We report it as a
diagnostic; we do NOT remove vig from three-way 90-minute odds
because the mapping from 90-minute win/draw/loss to advancement is
non-trivial and we prefer to use real ``to advance`` markets.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from football_advance_predictor.core.time import to_utc
from football_advance_predictor.db.models import MarketOddsSnapshot


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability.

    Args:
        decimal_odds: Decimal odds (> 1.0).

    Returns:
        Implied probability in [0, 1].
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def de_vig_two_way(home_odds: float, away_odds: float) -> tuple[float, float]:
    """Normalize two implied probabilities to remove overround.

    Returns:
        A tuple (home_normalized, away_normalized) summing to 1.0.
    """
    p_home = implied_probability(home_odds)
    p_away = implied_probability(away_odds)
    total = p_home + p_away
    if total <= 0:
        raise ValueError("Implied probabilities must be positive.")
    return p_home / total, p_away / total


@dataclass
class MarketConsensusDiagnostics:
    """Diagnostics about a market consensus calculation."""

    bookmaker_count: int
    overround: float
    raw_home: float
    raw_away: float
    dispersion: float


@dataclass
class MarketConsensus:
    """The result of a market consensus query."""

    home_advance_probability: float
    away_advance_probability: float
    diagnostics: MarketConsensusDiagnostics
    most_recent_captured_at: datetime


class MarketAdvanceProbabilityModel:
    """Compute consensus home/away advance probability from odds snapshots.

    The model intentionally returns ``None`` when no valid two-way
    ``to advance`` market exists. It never fabricates probabilities.
    """

    def __init__(
        self,
        snapshots: Iterable[MarketOddsSnapshot],
        *,
        min_bookmakers: int = 1,
        prefer_advance_market: bool = True,
    ) -> None:
        self.snapshots = list(snapshots)
        self.min_bookmakers = max(1, int(min_bookmakers))
        self.prefer_advance_market = prefer_advance_market

    def consensus_at(self, cutoff: datetime) -> MarketConsensus | None:
        """Return market consensus as of ``cutoff``.

        Only snapshots with ``captured_at <= cutoff`` are considered.
        The most recent snapshot per (bookmaker, market_type, selection)
        wins.
        """
        cutoff_utc = to_utc(cutoff)
        latest: dict[tuple[str, str, str], MarketOddsSnapshot] = {}
        for snap in sorted(self.snapshots, key=lambda s: s.captured_at):
            if to_utc(snap.captured_at) > cutoff_utc:
                continue
            key = (snap.bookmaker, snap.market_type, snap.selection)
            latest[key] = snap
        if not latest:
            return None

        # Decide which market to use.
        market_types = sorted({s.market_type for s in latest.values()})
        chosen_market = self._choose_market(market_types)
        if chosen_market is None:
            return None

        bookmakers_for_market = sorted(
            {s.bookmaker for s in latest.values() if s.market_type == chosen_market}
        )
        if len(bookmakers_for_market) < self.min_bookmakers:
            return None

        home_by_book: dict[str, float] = {}
        away_by_book: dict[str, float] = {}
        for s in latest.values():
            if s.market_type != chosen_market:
                continue
            sel = s.selection.lower()
            if sel in {"home", "home_to_advance"}:
                home_by_book[s.bookmaker] = s.decimal_odds
            elif sel in {"away", "away_to_advance"}:
                away_by_book[s.bookmaker] = s.decimal_odds

        bookmakers = sorted(set(home_by_book) & set(away_by_book))
        if len(bookmakers) < self.min_bookmakers:
            return None

        home_probs: list[float] = []
        away_probs: list[float] = []
        raw_home_total = 0.0
        raw_away_total = 0.0
        most_recent = max(s.captured_at for s in latest.values())
        for b in bookmakers:
            p_h, p_a = de_vig_two_way(home_by_book[b], away_by_book[b])
            home_probs.append(p_h)
            away_probs.append(p_a)
            raw_home_total += implied_probability(home_by_book[b])
            raw_away_total += implied_probability(away_by_book[b])

        n = len(home_probs)
        avg_home = sum(home_probs) / n
        avg_away = sum(away_probs) / n
        # Renormalize to handle small numerical drift.
        s = avg_home + avg_away
        if s > 0:
            avg_home /= s
            avg_away /= s
        dispersion = statistics.pstdev(home_probs) if n > 1 else 0.0
        overround = (raw_home_total / n) + (raw_away_total / n) - 1.0

        return MarketConsensus(
            home_advance_probability=avg_home,
            away_advance_probability=avg_away,
            diagnostics=MarketConsensusDiagnostics(
                bookmaker_count=n,
                overround=float(overround),
                raw_home=float(raw_home_total / n),
                raw_away=float(raw_away_total / n),
                dispersion=float(dispersion),
            ),
            most_recent_captured_at=most_recent,
        )

    def _choose_market(self, market_types: list[str]) -> str | None:
        if not market_types:
            return None
        advance_types = [m for m in market_types if "advance" in m]
        if self.prefer_advance_market and advance_types:
            return advance_types[0]
        if "moneyline_90" in market_types:
            return "moneyline_90"
        return market_types[0]
