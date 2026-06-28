"""Minimal StatsBomb-style local event data provider.

The provider only reads a directory of pre-exported StatsBomb-like
JSON files. It does NOT make network requests. Event data is only used
for optional advanced features (xG, set-piece xG, etc.) and is
explicitly opt-in via the ``features.matchup.enabled`` flag.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


class StatsBombLocalProvider:
    """Read StatsBomb-like events from a local directory of JSON files."""

    name = "statsbomb_local"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"StatsBomb directory not found: {self.path}")

    def iter_matches(self) -> Iterable[dict[str, Any]]:
        for file in sorted(self.path.glob("*.json")):
            try:
                with file.open("r", encoding="utf-8") as f:
                    yield json.load(f)
            except json.JSONDecodeError as exc:
                logger.warning("Invalid event JSON", extra={"file": str(file), "error": str(exc)})

    def aggregate_team_match_features(self, match_id: str) -> dict[str, float]:
        """Aggregate per-team match features for a single match.

        Returns an empty dict when the match is not present.
        """
        for record in self.iter_matches():
            if record.get("match_id") == match_id:
                return self._summarize(record)
        return {}

    @staticmethod
    def _summarize(record: dict[str, Any]) -> dict[str, float]:
        events = record.get("events", []) or []
        shots = [e for e in events if e.get("type", {}).get("name") == "Shot"]
        xg_total = sum(float(s.get("shot", {}).get("statsbomb_xg") or 0.0) for s in shots)
        set_piece_xg = sum(
            float(s.get("shot", {}).get("statsbomb_xg") or 0.0)
            for s in shots
            if s.get("play_pattern", {}).get("name") in {"From Free Kick", "From Corner"}
        )
        return {
            "events_count": float(len(events)),
            "shots_count": float(len(shots)),
            "xg_total": xg_total,
            "set_piece_xg": set_piece_xg,
        }
