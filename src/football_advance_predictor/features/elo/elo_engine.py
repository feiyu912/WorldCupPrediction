"""Dynamic Elo engine.

Implements a chronological, configurable Elo model:

    expected_home = 1 / (1 + 10^((R_away - R_home - H) / 400))
    R_new = R_old + K * (score - expected) * mov_mult

where ``score`` is 1.0 for a home win in regulation, 0.5 for a draw
(advance or loss in normal time both map to 0/1 with separate handling
for knockout), and 0.0 for a loss. ``K`` is the per-match update
factor, possibly modulated by margin of victory and time decay.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc

logger = get_logger(__name__)


@dataclass
class EloConfig:
    """Elo configuration parameters.

    See ``configs/elo.yaml`` and ``docs/modeling.md`` for details.
    """

    initial_rating: float = 1500.0
    base_k_factor: float = 20.0
    k_floor: float = 5.0
    k_ceiling: float = 40.0
    home_advantage: float = 60.0
    time_decay_per_day: float = 0.0
    mov_multiplier: float = 0.0
    mov_scale: float = 2.0
    tie_resolution: str = "draw_treated_as_50_50"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EloConfig:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class EloRating:
    """Elo rating state for a single team."""

    rating: float
    last_match_at: datetime | None = None
    history: list[tuple[datetime, float]] = field(default_factory=list)


class DynamicEloEngine:
    """Configurable dynamic Elo engine.

    The engine is fit on a chronologically sorted list of matches. It
    exposes ``get_team_rating`` and ``predict_home_advance_probability``
    as-of any cutoff timestamp.
    """

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self._ratings: dict[str, EloRating] = {}
        self._training_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, matches: Iterable[dict[str, Any]]) -> DynamicEloEngine:
        """Compute ratings by iterating matches in chronological order.

        Args:
            matches: Iterable of dicts with keys:
                - ``kickoff_at`` (datetime, UTC)
                - ``home_team_id`` (str)
                - ``away_team_id`` (str)
                - ``home_goals`` (int | None)
                - ``away_goals`` (int | None)
                - ``competition_importance`` (float, optional)
                - ``neutral_venue`` (bool, optional)
                - ``home_advances`` (bool | None, optional)

        Returns:
            self
        """
        for match in sorted(matches, key=lambda m: to_utc(m["kickoff_at"])):
            self._update(match)
        return self

    def get_team_rating(self, team_id: str, as_of_time: datetime) -> float:
        """Return the team's rating as of ``as_of_time``.

        Uses the rating history to return the rating as it was at
        ``as_of_time``. Includes time decay for days since the last
        update when ``time_decay_per_day > 0``.
        """
        as_of = to_utc(as_of_time)
        rating = self._ratings.get(team_id)
        if rating is None:
            return self.config.initial_rating
        if not rating.history:
            return rating.rating
        # Find the latest history entry at or before ``as_of``.
        latest_ts: datetime | None = None
        latest_value: float = self.config.initial_rating
        for ts, value in rating.history:
            if to_utc(ts) <= as_of and (latest_ts is None or to_utc(ts) > latest_ts):
                latest_ts = to_utc(ts)
                latest_value = value
        if latest_ts is None:
            # No history entry at or before as_of; return initial rating.
            return self.config.initial_rating
        if self.config.time_decay_per_day <= 0.0:
            return latest_value
        days = (as_of - latest_ts).total_seconds() / 86400.0
        if days <= 0:
            return latest_value
        return latest_value * math.exp(-self.config.time_decay_per_day * days)

    def predict_home_win_probability(
        self,
        home_team_id: str,
        away_team_id: str,
        as_of_time: datetime,
        neutral_venue: bool = False,
    ) -> float:
        """Return P(home team wins in regulation) as of ``as_of_time``.

        Knockout draws are not handled here; the caller converts to
        P(home_advances) via ``predict_home_advance_probability``.
        """
        r_home = self.get_team_rating(home_team_id, as_of_time)
        r_away = self.get_team_rating(away_team_id, as_of_time)
        h = 0.0 if neutral_venue else self.config.home_advantage
        exponent = (r_away - r_home - h) / 400.0
        return 1.0 / (1.0 + 10.0**exponent)

    def predict_home_advance_probability(
        self,
        home_team_id: str,
        away_team_id: str,
        as_of_time: datetime,
        neutral_venue: bool = False,
    ) -> float:
        """Return P(home team advances) under the configured tie model.

        The advance probabilities for the two sides MUST sum to 1 within
        numerical tolerance (complementarity invariant).

        Under ``draw_treated_as_50_50`` a 90-minute draw is treated as
        50/50 advancement. Using the symmetric identity::

            p_draw = 1 - p_home_win - p_away_win
            p_home_advances = 0.5 * p_home_win - 0.5 * p_away_win + 0.5

        which makes p_home_advances(B, A) = 1 - p_home_advances(A, B) for
        any rating gap. The earlier formula
        ``p_home_win + 0.5 * p_draw`` was NOT symmetric and broke the
        complementarity invariant.
        """
        p_home_win = self.predict_home_win_probability(
            home_team_id, away_team_id, as_of_time, neutral_venue=neutral_venue
        )
        p_away_win = 1.0 - p_home_win
        if self.config.tie_resolution == "draw_treated_as_50_50":
            return 0.5 * p_home_win - 0.5 * p_away_win + 0.5
        if self.config.tie_resolution == "no_draw_assume_extra_time":
            return max(p_home_win, 0.5)
        if self.config.tie_resolution == "no_draw_penalties_50_50":
            return 0.5
        # Default: ignore draws.
        return p_home_win

    def predict_away_advance_probability(
        self,
        home_team_id: str,
        away_team_id: str,
        as_of_time: datetime,
        neutral_venue: bool = False,
    ) -> float:
        """Return P(away team advances) under the configured tie model.

        The complement of :meth:`predict_home_advance_probability`. Exposed
        as a separate method so the symmetry invariant can be checked
        directly.
        """
        return 1.0 - self.predict_home_advance_probability(
            home_team_id, away_team_id, as_of_time, neutral_venue=neutral_venue
        )

    def generate_rating_history(self) -> list[dict[str, Any]]:
        """Return the rating history as a list of dicts."""
        out: list[dict[str, Any]] = []
        for team_id, rating in self._ratings.items():
            for ts, value in rating.history:
                out.append({"team_id": team_id, "timestamp": ts, "rating": value})
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _update(self, match: dict[str, Any]) -> None:
        home_id = match["home_team_id"]
        away_id = match["away_team_id"]
        kickoff = to_utc(match["kickoff_at"])
        neutral_venue = bool(match.get("neutral_venue", False))
        home_goals = match.get("home_goals")
        away_goals = match.get("away_goals")
        importance = float(match.get("competition_importance", 1.0))
        home_advances = match.get("home_advances")

        r_home_pre = self.get_team_rating(home_id, kickoff)
        r_away_pre = self.get_team_rating(away_id, kickoff)
        h = 0.0 if neutral_venue else self.config.home_advantage
        expected_home = 1.0 / (1.0 + 10.0 ** ((r_away_pre - r_home_pre - h) / 400.0))
        expected_away = 1.0 - expected_home

        if home_advances is not None:
            # Knockout perspective: score 1 if home advances, 0.5 if draw goes to home,
            # 0 if home does not advance. We don't have goal diff for penalty outcomes.
            if home_advances:
                score_home = 1.0
                score_away = 0.0
            else:
                score_home = 0.0
                score_away = 1.0
        else:
            if home_goals is None or away_goals is None:
                return
            if home_goals > away_goals:
                score_home = 1.0
                score_away = 0.0
            elif home_goals < away_goals:
                score_home = 0.0
                score_away = 1.0
            else:
                score_home = 0.5
                score_away = 0.5

        k = self._k_factor(importance)
        mov = self._mov_multiplier(home_goals, away_goals) if self.config.mov_multiplier > 0 else 1.0
        delta_home = k * (score_home - expected_home) * mov
        delta_away = k * (score_away - expected_away) * mov

        self._record(home_id, kickoff, r_home_pre + delta_home)
        self._record(away_id, kickoff, r_away_pre + delta_away)

    def _k_factor(self, importance: float) -> float:
        k = self.config.base_k_factor * importance
        return max(self.config.k_floor, min(self.config.k_ceiling, k))

    @staticmethod
    def _mov_multiplier(home_goals: int | None, away_goals: int | None) -> float:
        if home_goals is None or away_goals is None:
            return 1.0
        diff = abs(home_goals - away_goals)
        return 1.0 + math.log1p(diff) / math.log(2.0)  # ~1 + 0.5*log2(1+diff) style

    def _record(self, team_id: str, ts: datetime, new_rating: float) -> None:
        rating = self._ratings.get(team_id)
        if rating is None:
            rating = EloRating(rating=self.config.initial_rating)
            self._ratings[team_id] = rating
        rating.rating = new_rating
        rating.last_match_at = ts
        rating.history.append((ts, new_rating))


# ---------------------------------------------------------------------------
# Simple helpers
# ---------------------------------------------------------------------------


def expected_score(rating_a: float, rating_b: float) -> float:
    """Standard Elo expected score formula."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def aggregate_competition_weights(competition_weights: dict[str, float]) -> dict[str, float]:
    """Validate and normalize competition weights to a dict."""
    if not competition_weights:
        return {}
    return {k: float(v) for k, v in competition_weights.items() if v > 0}
