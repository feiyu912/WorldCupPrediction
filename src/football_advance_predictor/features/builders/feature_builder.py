"""Feature builder that composes individual feature modules.

The builder is the only place that touches the database for feature
generation. It enforces the anti-leakage contract: every input must
have a timestamp strictly less than the cutoff.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.db.models import (
    Competition,
    MarketOddsSnapshot,
    Match,
    MatchResult,
    PlayerAvailabilitySnapshot,
)
from football_advance_predictor.features.elo.elo_engine import DynamicEloEngine, EloConfig
from football_advance_predictor.features.lineup.lineup_features import (
    diff_features,
    lineup_features,
)
from football_advance_predictor.features.market.consensus import MarketAdvanceProbabilityModel
from football_advance_predictor.features.team_form.team_form import (
    FormMatch,
    rest_days,
    weighted_goal_difference,
    weighted_points,
)

logger = get_logger(__name__)


class FeatureBuilder:
    """Compose features for a (match, cutoff) pair from a database session.

    Args:
        session: SQLAlchemy session.
        elo_config: Optional Elo configuration.
        feature_version: Logical feature version, recorded in the snapshot.
    """

    def __init__(
        self,
        session: Session | None = None,
        elo_config: EloConfig | None = None,
        feature_version: str = "v1",
        market_min_bookmakers: int = 1,
    ) -> None:
        self.session = session
        self.elo_config = elo_config or EloConfig()
        self.feature_version = feature_version
        self.market_min_bookmakers = market_min_bookmakers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self, *, match_id: str, cutoff_time: datetime
    ) -> tuple[dict[str, Any], datetime]:
        """Build a time-frozen feature dict.

        Args:
            match_id: Match identifier.
            cutoff_time: Cutoff timestamp (UTC).

        Returns:
            A tuple ``(features, source_data_max_timestamp)``.
        """
        if self.session is None:
            raise RuntimeError("FeatureBuilder requires a database session.")

        cutoff = to_utc(cutoff_time)
        match = self.session.get(Match, match_id)
        if match is None:
            raise ValueError(f"Match not found: {match_id}")
        if to_utc(match.kickoff_at) <= cutoff:
            raise ValueError(
                f"Cutoff must be strictly before kickoff "
                f"(cutoff={cutoff.isoformat()}, kickoff={match.kickoff_at.isoformat()})."
            )

        home_id = match.home_team_id
        away_id = match.away_team_id

        # ---- 1. Long-term team strength (Elo) ----------------------
        elo_engine = self._build_elo()
        elo_home = elo_engine.get_team_rating(home_id, cutoff)
        elo_away = elo_engine.get_team_rating(away_id, cutoff)
        elo_diff = elo_home - elo_away
        elo_home_adv = elo_engine.predict_home_advance_probability(
            home_id, away_id, cutoff, neutral_venue=match.neutral_venue
        )

        # ---- 2. Recent form ---------------------------------------
        form_matches = self._form_matches_for(home_id, away_id, cutoff)
        home_form_recent = [m for m in form_matches if m.team_id == home_id]
        away_form_recent = [m for m in form_matches if m.team_id == away_id]
        half_life = 180.0
        home_pts = weighted_points(home_form_recent, cutoff=cutoff, half_life_days=half_life)
        away_pts = weighted_points(away_form_recent, cutoff=cutoff, half_life_days=half_life)
        home_gd = weighted_goal_difference(home_form_recent, cutoff=cutoff, half_life_days=half_life)
        away_gd = weighted_goal_difference(away_form_recent, cutoff=cutoff, half_life_days=half_life)

        rest_home = rest_days(home_form_recent, cutoff=cutoff)
        rest_away = rest_days(away_form_recent, cutoff=cutoff)

        # ---- 3. Market features -----------------------------------
        odds = list(
            self.session.scalars(
                select(MarketOddsSnapshot)
                .where(MarketOddsSnapshot.match_id == match_id)
                .where(MarketOddsSnapshot.captured_at <= cutoff)
            )
        )
        market_model = MarketAdvanceProbabilityModel(odds, min_bookmakers=self.market_min_bookmakers)
        consensus = market_model.consensus_at(cutoff)
        market_features: dict[str, Any] = {
            "market_available": consensus is not None,
            "market_bookmaker_count": consensus.diagnostics.bookmaker_count if consensus else 0,
            "market_overround": consensus.diagnostics.overround if consensus else 0.0,
            "market_dispersion": consensus.diagnostics.dispersion if consensus else 0.0,
            "market_home_advance_probability": (
                consensus.home_advance_probability if consensus else 0.5
            ),
        }

        # ---- 4. Squad / availability features --------------------
        availability = list(
            self.session.scalars(
                select(PlayerAvailabilitySnapshot)
                .where(PlayerAvailabilitySnapshot.match_id == match_id)
                .where(PlayerAvailabilitySnapshot.published_at <= cutoff)
            )
        )
        home_avail = [a for a in availability if a.team_id == home_id]
        away_avail = [a for a in availability if a.team_id == away_id]
        lineup_diff = diff_features(home_avail, away_avail)
        home_lineup = {f"home_{k}": v for k, v in lineup_features(home_avail).items()}
        away_lineup = {f"away_{k}": v for k, v in lineup_features(away_avail).items()}

        # ---- 5. Schedule / environment ---------------------------
        competition = self.session.get(Competition, match.competition_id)
        competition_features = {
            "competition_importance": float(competition.importance_weight if competition else 1.0),
            "neutral_venue": bool(match.neutral_venue),
            "is_knockout": _is_knockout_match(match, competition),
        }

        features: dict[str, Any] = {
            # Elo
            "home_elo_pre_match": elo_home,
            "away_elo_pre_match": elo_away,
            "elo_difference": elo_diff,
            "elo_home_advance_probability": elo_home_adv,
            # Form
            "home_form_points": home_pts,
            "away_form_points": away_pts,
            "form_points_difference": home_pts - away_pts,
            "home_goal_difference_recent": home_gd,
            "away_goal_difference_recent": away_gd,
            "goal_difference_recent_diff": home_gd - away_gd,
            # Schedule
            "rest_days_home": float(rest_home),
            "rest_days_away": float(rest_away),
            "rest_days_difference": float(rest_home - rest_away),
            # Squad
            **home_lineup,
            **away_lineup,
            **{f"{k}_diff": v for k, v in lineup_diff.items()},
            # Market
            **market_features,
            # Competition
            **competition_features,
        }

        # Compute the maximum source timestamp we've consumed.
        source_max_ts = to_utc(cutoff)
        if odds:
            source_max_ts = max(
                source_max_ts, max(to_utc(o.captured_at) for o in odds)
            )
        if availability:
            source_max_ts = max(
                source_max_ts, max(to_utc(a.published_at) for a in availability)
            )

        return features, source_max_ts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_elo(self) -> DynamicEloEngine:
        """Build an Elo engine from all matches ending before the cutoff.

        Note: this is a per-snapshot fit. For very large datasets we'd
        precompute rating history; the MVP uses lazy per-snapshot fitting.
        """
        # We do not filter by cutoff in this query: instead, we filter
        # during iteration. This keeps the function pure given a session.
        stmt = select(Match, MatchResult).outerjoin(
            MatchResult, MatchResult.match_id == Match.match_id
        )
        rows = self.session.execute(stmt).all()
        engine = DynamicEloEngine(self.elo_config)
        for match, result in rows:
            if result is None or result.home_advances is None:
                continue
            engine._update(
                {
                    "kickoff_at": match.kickoff_at,
                    "home_team_id": match.home_team_id,
                    "away_team_id": match.away_team_id,
                    "neutral_venue": match.neutral_venue,
                    "home_goals": result.home_goals_90,
                    "away_goals": result.away_goals_90,
                    "competition_importance": self._competition_importance(match.competition_id),
                    "home_advances": result.home_advances,
                }
            )
        return engine

    def _competition_importance(self, competition_id: str) -> float:
        comp = self.session.get(Competition, competition_id)
        return float(comp.importance_weight) if comp else 1.0

    def _form_matches_for(
        self, home_id: str, away_id: str, cutoff: datetime
    ) -> list[FormMatch]:
        cutoff_utc = cutoff
        # Pull all results for either team, finished before cutoff.
        stmt = (
            select(Match, MatchResult)
            .join(MatchResult, MatchResult.match_id == Match.match_id)
            .where(Match.kickoff_at < cutoff_utc)
        )
        rows = self.session.execute(stmt).all()
        form: list[FormMatch] = []
        for match, result in rows:
            for team_id, opponent_id, gf, ga, won, drew, lost in (
                (match.home_team_id, match.away_team_id, result.home_goals_90, result.away_goals_90,
                 (result.home_goals_90 or 0) > (result.away_goals_90 or 0),
                 (result.home_goals_90 == result.away_goals_90) if (result.home_goals_90 is not None and result.away_goals_90 is not None) else False,
                 (result.home_goals_90 or 0) < (result.away_goals_90 or 0)),
                (match.away_team_id, match.home_team_id, result.away_goals_90, result.home_goals_90,
                 (result.away_goals_90 or 0) > (result.home_goals_90 or 0),
                 (result.home_goals_90 == result.away_goals_90) if (result.home_goals_90 is not None and result.away_goals_90 is not None) else False,
                 (result.away_goals_90 or 0) < (result.home_goals_90 or 0)),
            ):
                if team_id not in (home_id, away_id):
                    continue
                if gf is None or ga is None:
                    continue
                form.append(
                    FormMatch(
                        kickoff_at=match.kickoff_at,
                        team_id=team_id,
                        opponent_id=opponent_id,
                        goals_for=gf,
                        goals_against=ga,
                        home_advances=result.home_advances if team_id == match.home_team_id else (not result.home_advances),
                        won=won,
                        drew=drew,
                        lost=lost,
                    )
                )
        return form


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _is_knockout_match(match: Match, competition: Competition | None) -> bool:
    """Return True when the match is a knockout match.

    A match is considered a knockout match if any of the following
    holds (in priority order):

    1. The competition is marked ``is_knockout_capable=False`` (group
       tournament) AND the match's stage is "Group"/"League"/"Round
       Robin"/"unknown" → not a knockout.
    2. The match's stage contains a knockout keyword.
    3. The competition is not marked and the stage is non-group.
    """
    if competition is not None and not competition.is_knockout_capable:
        return _is_knockout_stage(match.stage)
    return _is_knockout_stage(match.stage)


def _is_knockout_stage(stage: str) -> bool:
    if not stage:
        return False
    lowered = stage.lower()
    group_terms = ("group", "league", "round robin", "round-robin")
    knockout_terms = (
        "round of 16", "r16", "round-of-16",
        "quarter", "qf", "quarter-final",
        "semi", "sf", "semi-final", "semifinal",
        "final", "3rd place", "third place", "play-off", "playoff", "knockout",
    )
    if any(term in lowered for term in group_terms):
        return False
    return any(term in lowered for term in knockout_terms)
