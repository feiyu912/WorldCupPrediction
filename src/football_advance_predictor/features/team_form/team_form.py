"""Recent form features with exponential time decay.

A team's recent form is summarized by the weighted average of match
points (3 win, 1 draw, 0 loss) over a configurable set of windows. The
weights decay exponentially with the age of the match relative to the
cutoff timestamp.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from football_advance_predictor.core.time import to_utc


@dataclass(frozen=True)
class FormMatch:
    """A historical match relevant for form computation."""

    kickoff_at: datetime
    team_id: str
    opponent_id: str
    goals_for: int
    goals_against: int
    home_advances: bool | None  # knockout; None for non-knockout
    won: bool
    drew: bool
    lost: bool


def _decay(age_days: float, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    return math.exp(-math.log(2.0) * age_days / half_life_days)


def weighted_points(
    matches: Iterable[FormMatch],
    *,
    cutoff: datetime,
    half_life_days: float = 180.0,
) -> float:
    """Return time-decayed weighted points total (3W, 1D, 0L)."""
    cutoff_utc = to_utc(cutoff)
    total = 0.0
    weight_sum = 0.0
    for m in matches:
        if m.team_id is None:
            continue
        age_days = (cutoff_utc - to_utc(m.kickoff_at)).total_seconds() / 86400.0
        if age_days < 0:
            continue
        w = _decay(age_days, half_life_days)
        points = 3 if m.won else 1 if m.drew else 0
        total += w * points
        weight_sum += w
    if weight_sum <= 0:
        return 0.0
    return total / weight_sum


def weighted_goal_difference(
    matches: Iterable[FormMatch],
    *,
    cutoff: datetime,
    half_life_days: float = 180.0,
) -> float:
    cutoff_utc = to_utc(cutoff)
    total = 0.0
    weight_sum = 0.0
    for m in matches:
        age_days = (cutoff_utc - to_utc(m.kickoff_at)).total_seconds() / 86400.0
        if age_days < 0:
            continue
        w = _decay(age_days, half_life_days)
        total += w * (m.goals_for - m.goals_against)
        weight_sum += w
    if weight_sum <= 0:
        return 0.0
    return total / weight_sum


def rest_days(
    matches: Iterable[FormMatch],
    *,
    cutoff: datetime,
    default: int = 7,
) -> int:
    """Days between the most recent match and ``cutoff``."""
    cutoff_utc = to_utc(cutoff)
    latest: datetime | None = None
    for m in matches:
        kickoff = to_utc(m.kickoff_at)
        if kickoff >= cutoff_utc:
            continue
        if latest is None or kickoff > latest:
            latest = kickoff
    if latest is None:
        return default
    return max(0, int((cutoff_utc - latest).total_seconds() // 86400))
