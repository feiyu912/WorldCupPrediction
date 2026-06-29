"""Pre-registered v1 feature set.

Limited to historical data available before kickoff:

- neutral-context Elo difference
- opponent-strength-weighted recent result form
- recency-weighted goal-difference proxy
- rest-day difference
- tournament-stage indicators where known before kickoff

The set is intentionally small (5 features). A richer feature set
(StatsBomb xG, market consensus, lineup data) is out of scope for the
v1 pre-registration and will be added in a later release.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc

logger = get_logger(__name__)


@dataclass(frozen=True)
class V1FeatureRow:
    """The v1 feature set for a single (match, cutoff) pair.

    All features are computed strictly before the match kickoff. The
    reference team orientation is the home team as recorded in the
    manifest (the report script applies ``reference_team_for_match``
    when it needs to display a probability against a deterministic
    reference side).
    """

    match_id: str
    home_team_id: str
    away_team_id: str
    kickoff_at: datetime
    stage_canonical: str

    # 1) neutral-context Elo difference. Uses home_advantage=0 because
    # all knockout matches are treated as neutral venues. Positive
    # values favor the home team.
    elo_difference: float
    elo_home_win_prob: float

    # 2) opponent-strength-weighted recent result form. Each team's
    # last 5 results weighted by the opponent's rating at the time.
    # Positive values favor the home team.
    form_home: float
    form_away: float
    form_difference: float

    # 3) recency-weighted goal-difference proxy. Last 8 matches per
    # team with exponential time decay. Positive favors the home team.
    goal_diff_home: float
    goal_diff_away: float
    goal_diff_difference: float

    # 4) rest-day difference. Days between each team's last match
    # and the current kickoff.
    rest_days_home: float
    rest_days_away: float
    rest_days_difference: float

    # 5) tournament-stage indicators. Encoded as one-hot.
    is_round_of_16: int
    is_quarter_final: int
    is_semi_final: int
    is_final: int

    def feature_dict(self) -> dict[str, float]:
        return {
            "elo_difference": self.elo_difference,
            "elo_home_win_prob": self.elo_home_win_prob,
            "form_difference": self.form_difference,
            "form_home": self.form_home,
            "form_away": self.form_away,
            "goal_diff_difference": self.goal_diff_difference,
            "goal_diff_home": self.goal_diff_home,
            "goal_diff_away": self.goal_diff_away,
            "rest_days_difference": self.rest_days_difference,
            "rest_days_home": self.rest_days_home,
            "rest_days_away": self.rest_days_away,
            "is_round_of_16": float(self.is_round_of_16),
            "is_quarter_final": float(self.is_quarter_final),
            "is_semi_final": float(self.is_semi_final),
            "is_final": float(self.is_final),
        }


def _exp_decay_weight(age_days: float, half_life_days: float = 365.0) -> float:
    if age_days < 0:
        return 0.0
    return math.exp(-math.log(2) * age_days / half_life_days)


def compute_v1_features(
    *,
    home_team_id: str,
    away_team_id: str,
    kickoff_at: datetime,
    stage_canonical: str,
    cutoff: datetime,
    home_elo_at_cutoff: float,
    away_elo_at_cutoff: float,
    home_recent_results: Iterable[tuple[datetime, str, int]],
    away_recent_results: Iterable[tuple[datetime, str, int]],
    home_recent_goal_diff: Iterable[tuple[datetime, int]],
    away_recent_goal_diff: Iterable[tuple[datetime, int]],
    home_last_match_at: datetime | None,
    away_last_match_at: datetime | None,
) -> V1FeatureRow:
    """Build a single :class:`V1FeatureRow`.

    Args:
        home_team_id / away_team_id: canonical team ids.
        kickoff_at: scheduled kickoff (UTC).
        stage_canonical: canonical stage name from the manifest.
        cutoff: prediction cutoff (UTC, must be < kickoff_at).
        home_elo_at_cutoff / away_elo_at_cutoff: Elo ratings as of ``cutoff``.
        home_recent_results / away_recent_results: each item is
            ``(match_at, opponent_id, points)`` with ``points in {0, 1, 3}``.
        home_recent_goal_diff / away_recent_goal_diff: each item is
            ``(match_at, goals_for - goals_against)``.
        home_last_match_at / away_last_match_at: most recent match
            timestamp for each team (or None if unknown).
    """
    if cutoff >= kickoff_at:
        raise ValueError("cutoff must be strictly before kickoff_at")

    # 1) neutral-context Elo difference
    elo_difference = home_elo_at_cutoff - away_elo_at_cutoff
    exponent = (away_elo_at_cutoff - home_elo_at_cutoff) / 400.0
    elo_home_win_prob = 1.0 / (1.0 + 10.0**exponent)

    # 2) opponent-strength-weighted recent result form. Weight each
    # result by the opponent's rating at the cutoff (use 1500 as a
    # neutral fallback for unknown opponents).
    def _weighted_form(
        rows: Iterable[tuple[datetime, str, int]],
        ratings: dict[str, float],
    ) -> float:
        total_w = 0.0
        total_wp = 0.0
        for match_at, opponent_id, points in rows:
            age = (to_utc(cutoff) - to_utc(match_at)).total_seconds() / 86400.0
            decay = _exp_decay_weight(age)
            opponent_rating = ratings.get(opponent_id, 1500.0)
            w = decay * opponent_rating
            total_w += w
            total_wp += w * float(points)
        if total_w == 0:
            return 0.0
        return total_wp / total_w

    form_home = _weighted_form(
        home_recent_results, {away_team_id: away_elo_at_cutoff}
    )
    form_away = _weighted_form(
        away_recent_results, {home_team_id: home_elo_at_cutoff}
    )
    form_difference = form_home - form_away

    # 3) recency-weighted goal-difference proxy.
    def _weighted_goal_diff(rows: Iterable[tuple[datetime, int]]) -> float:
        total_w = 0.0
        total_wd = 0.0
        for match_at, gd in rows:
            age = (to_utc(cutoff) - to_utc(match_at)).total_seconds() / 86400.0
            decay = _exp_decay_weight(age)
            total_w += decay
            total_wd += decay * float(gd)
        if total_w == 0:
            return 0.0
        return total_wd / total_w

    goal_diff_home = _weighted_goal_diff(home_recent_goal_diff)
    goal_diff_away = _weighted_goal_diff(away_recent_goal_diff)
    goal_diff_difference = goal_diff_home - goal_diff_away

    # 4) rest-day difference. Days since each team's last match.
    cutoff_utc = to_utc(cutoff)
    kickoff_utc = to_utc(kickoff_at)
    rest_home = (
        (kickoff_utc - to_utc(home_last_match_at)).total_seconds() / 86400.0
        if home_last_match_at is not None
        else 7.0  # default 7-day prior
    )
    rest_away = (
        (kickoff_utc - to_utc(away_last_match_at)).total_seconds() / 86400.0
        if away_last_match_at is not None
        else 7.0
    )
    rest_days_home = float(rest_home)
    rest_days_away = float(rest_away)
    rest_days_difference = rest_days_home - rest_days_away

    # 5) tournament-stage indicators (one-hot).
    is_round_of_16 = int(stage_canonical == "round_of_16")
    is_quarter_final = int(stage_canonical == "quarter_final")
    is_semi_final = int(stage_canonical == "semi_final")
    is_final = int(stage_canonical == "final")

    return V1FeatureRow(
        match_id=f"{home_team_id}_vs_{away_team_id}_{int(kickoff_utc.timestamp())}",
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        kickoff_at=kickoff_at,
        stage_canonical=stage_canonical,
        elo_difference=float(elo_difference),
        elo_home_win_prob=float(elo_home_win_prob),
        form_home=float(form_home),
        form_away=float(form_away),
        form_difference=float(form_difference),
        goal_diff_home=float(goal_diff_home),
        goal_diff_away=float(goal_diff_away),
        goal_diff_difference=float(goal_diff_difference),
        rest_days_home=rest_days_home,
        rest_days_away=rest_days_away,
        rest_days_difference=rest_days_difference,
        is_round_of_16=is_round_of_16,
        is_quarter_final=is_quarter_final,
        is_semi_final=is_semi_final,
        is_final=is_final,
    )


def v1_features_to_dataframe(rows: list[V1FeatureRow]) -> tuple[Any, Any]:
    """Convert v1 feature rows into (X, y) DataFrames for sklearn.

    The label is the home team's win_tie indicator. Mirrored rows
    are added by the caller (kept in the same fold as their
    originals) to enforce the symmetry invariant at training time.
    """
    import pandas as pd

    X = pd.DataFrame([r.feature_dict() for r in rows])
    X["match_id"] = [r.match_id for r in rows]
    X["home_team_id"] = [r.home_team_id for r in rows]
    X["away_team_id"] = [r.away_team_id for r in rows]
    X["kickoff_at"] = [r.kickoff_at.isoformat() for r in rows]
    X = X.set_index("match_id")
    return X, X
