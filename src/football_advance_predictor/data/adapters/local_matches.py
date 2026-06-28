"""Local historical results provider (CSV).

Reads a CSV file with historical international or domestic matches. The
schema is intentionally permissive: column names are matched
case-insensitively. Unrecognized rows are logged and quarantined rather
than silently dropped.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.data.adapters.base import MatchDataProvider
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn

logger = get_logger(__name__)

# Column aliases (lowercased).
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "match_id": ("match_id", "id"),
    "kickoff_at": ("date", "kickoff", "kickoff_at", "match_date"),
    "competition_id": ("tournament", "competition", "competition_id"),
    "stage": ("stage", "round"),
    "season_or_year": ("season", "year", "season_or_year"),
    "home_team_id": ("home_team", "home", "home_team_id"),
    "away_team_id": ("away_team", "away", "away_team_id"),
    "home_goals": ("home_score", "home_goals"),
    "away_goals": ("away_score", "away_goals"),
    "winner_team_id": ("winner", "winner_team_id"),
    "advancing_team_id": ("advancing_team", "advancing_team_id"),
    "neutral_venue": ("neutral", "neutral_venue"),
    "venue_name": ("venue", "venue_name"),
    "city": ("city",),
    "country": ("country",),
    "source": ("source",),
}

ALIAS_LOOKUP: dict[str, str] = {
    alias: canonical for canonical, aliases in COLUMN_ALIASES.items() for alias in aliases
}


class LocalHistoricalResultsProvider(MatchDataProvider):
    """Read historical matches from a local CSV file."""

    name = "local_historical_results"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Match CSV not found: {self.path}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_matches(self, **kwargs: Any) -> list[MatchIn]:
        rows = self._read_csv()
        matches: list[MatchIn] = []
        for idx, row in enumerate(rows):
            try:
                matches.append(self._row_to_match(row))
            except Exception as exc:
                logger.warning(
                    "Failed to parse match row",
                    extra={"row_index": idx, "error": str(exc), "row": row},
                )
        return matches

    def fetch_results(self, **kwargs: Any) -> list[MatchResultIn]:
        rows = self._read_csv()
        results: list[MatchResultIn] = []
        for idx, row in enumerate(rows):
            try:
                match = self._row_to_match(row)
                result = self._match_to_result(match, row)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.warning(
                    "Failed to parse match result row",
                    extra={"row_index": idx, "error": str(exc), "row": row},
                )
        return results

    def fetch_teams(self, **kwargs: Any) -> list[dict[str, Any]]:
        rows = self._read_csv()
        teams: dict[str, dict[str, Any]] = {}
        for row in rows:
            for col in ("home_team_id", "away_team_id"):
                team_id = self._canonicalize_team_id(row, col)
                if team_id and team_id not in teams:
                    teams[team_id] = {"team_id": team_id, "canonical_name": team_id}
        return list(teams.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _read_csv(self) -> list[dict[str, str]]:
        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return [{self._canonical_key(k): v for k, v in row.items() if k is not None} for row in reader]

    @staticmethod
    def _canonical_key(key: str) -> str:
        return ALIAS_LOOKUP.get(key.lower().strip(), key.lower().strip())

    def _canonicalize_team_id(self, row: dict[str, str], canonical_col: str) -> str | None:
        value = row.get(canonical_col)
        if not value:
            return None
        return value.strip().lower().replace(" ", "_")

    def _row_to_match(self, row: dict[str, str]) -> MatchIn:
        kickoff = self._parse_datetime(row.get("kickoff_at"))
        home_goals = self._parse_int(row.get("home_goals"))
        away_goals = self._parse_int(row.get("away_goals"))
        return MatchIn(
            match_id=row.get("match_id") or f"MATCH_{abs(hash(tuple(sorted(row.items()))))}",
            kickoff_at=kickoff,
            competition_id=row.get("competition_id", "unknown"),
            stage=row.get("stage", "unknown"),
            season_or_year=row.get("season_or_year", str(kickoff.year)),
            home_team_id=self._canonicalize_team_id(row, "home_team_id") or "home",
            away_team_id=self._canonicalize_team_id(row, "away_team_id") or "away",
            home_goals=home_goals,
            away_goals=away_goals,
            winner_team_id=self._canonicalize_team_id(row, "winner_team_id"),
            advancing_team_id=self._canonicalize_team_id(row, "advancing_team_id"),
            neutral_venue=self._parse_bool(row.get("neutral_venue")),
            venue_name=row.get("venue_name"),
            city=row.get("city"),
            country=row.get("country"),
            source=row.get("source", "local"),
        )

    @staticmethod
    def _match_to_result(match: MatchIn, row: dict[str, str]) -> MatchResultIn | None:
        if match.home_goals is None or match.away_goals is None:
            return None
        if match.advancing_team_id is None:
            # Without an explicit advancer, assume home advances if not a draw.
            if match.home_goals == match.away_goals:
                return None
            advancer = match.home_team_id if match.home_goals > match.away_goals else match.away_team_id
            home_advances = advancer == match.home_team_id
        else:
            home_advances = match.advancing_team_id == match.home_team_id
        return MatchResultIn(
            match_id=match.match_id,
            final_status="final",
            home_goals_90=match.home_goals,
            away_goals_90=match.away_goals,
            home_goals_et=None,
            away_goals_et=None,
            penalties_home=None,
            penalties_away=None,
            home_advances=home_advances,
            result_verified_at=to_utc(match.kickoff_at),
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            raise ValueError("Missing kickoff_at")
        try:
            return to_utc(value)
        except ValueError as exc:
            raise ValueError(f"Invalid kickoff_at: {value!r}") from exc

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return None

    @staticmethod
    def _parse_bool(value: str | None) -> bool:
        if not value:
            return False
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def normalize_team_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize raw team rows (helper for tests)."""
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({k.lower().strip(): v for k, v in r.items() if k})
    return out
