"""Lineup / availability derived features.

Operates on a list of ``PlayerAvailabilitySnapshot`` rows filtered to
``published_at <= cutoff``. For each team, we count confirmed-outs and
suspensions by role and return a small feature dict.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from football_advance_predictor.db.models import PlayerAvailabilitySnapshot

# Roles used for bucketing.
ROLE_BUCKETS: tuple[str, ...] = ("goalkeeper", "defender", "midfielder", "attacker")


def lineup_features(
    records: Iterable[PlayerAvailabilitySnapshot],
) -> dict[str, float]:
    """Compute lineup-derived features for one team."""
    feats: dict[str, float] = {}
    counts_status: Counter[str] = Counter()
    counts_role_out: Counter[str] = Counter()
    for r in records:
        counts_status[r.availability_status] += 1
        if r.availability_status in {"confirmed_out", "suspended"}:
            role = (r.role or "unknown").lower()
            for bucket in ROLE_BUCKETS:
                if bucket in role:
                    counts_role_out[bucket] += 1
                    break
    for status, count in counts_status.items():
        feats[f"availability_count_{status}"] = float(count)
    for bucket in ROLE_BUCKETS:
        feats[f"confirmed_out_{bucket}s"] = float(counts_role_out.get(bucket, 0))
    feats["lineup_confirmed"] = float(counts_status.get("lineup_confirmed", 0) > 0)
    return feats


def diff_features(
    home_records: Iterable[PlayerAvailabilitySnapshot],
    away_records: Iterable[PlayerAvailabilitySnapshot],
) -> dict[str, float]:
    """Return ``_diff`` features for the two team feature sets."""
    home = lineup_features(home_records)
    away = lineup_features(away_records)
    diff: dict[str, float] = {}
    for key in set(home) | set(away):
        diff[f"{key}_diff"] = home.get(key, 0.0) - away.get(key, 0.0)
    return diff
