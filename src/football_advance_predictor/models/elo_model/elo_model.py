"""Thin wrapper around ``DynamicEloEngine`` for the prediction service.

The wrapper is intentionally minimal. It exposes
``predict_proba(features)`` for compatibility with the rest of the
model layer, but the actual computation is delegated to the engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from football_advance_predictor.features.elo.elo_engine import DynamicEloEngine, EloConfig


class EloModel:
    """Elo-based home advance probability predictor.

    Args:
        config: Optional :class:`EloConfig`. Defaults to a sensible
            international-football configuration.
    """

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self.engine = DynamicEloEngine(self.config)
        self.is_fitted: bool = False
        self._fitted_teams: set[str] = set()

    def fit(self, matches: list[dict[str, Any]]) -> EloModel:
        """Fit the engine on a chronological list of matches."""
        self.engine.fit(matches)
        self.is_fitted = True
        self._fitted_teams.update(m.get("home_team_id") for m in matches if m.get("home_team_id"))
        self._fitted_teams.update(m.get("away_team_id") for m in matches if m.get("away_team_id"))
        return self

    def predict_proba(
        self,
        *,
        home_team_id: str,
        away_team_id: str,
        as_of_time: datetime,
        neutral_venue: bool = False,
    ) -> float:
        """Return P(home team advances) as of ``as_of_time``."""
        if not self.is_fitted:
            # Cold start: still works via initial rating.
            self.is_fitted = True
        return self.engine.predict_home_advance_probability(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            as_of_time=as_of_time,
            neutral_venue=neutral_venue,
        )

    def get_team_rating(self, team_id: str, as_of_time: datetime) -> float:
        return self.engine.get_team_rating(team_id, as_of_time)
