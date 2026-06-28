"""Matchup features (placeholder implementation).

The implementation here is a no-op stub. It exists to make the feature
builder's surface stable when ``features.matchup.enabled`` is False.
"""

from __future__ import annotations

from typing import Any


def matchup_features(*args: Any, **kwargs: Any) -> dict[str, float]:
    """Return an empty matchup feature dict.

    Real implementations would compute high-press vulnerability,
    set-piece offense vs defense, transition-risk proxy, etc., from
    StatsBomb-style event data. They must only include features whose
    computation is fully traceable to timestamped event data.
    """
    return {}
