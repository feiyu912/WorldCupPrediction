"""Adapter for the martj42/international-results dataset.

Reads the two CSV files produced by the bootstrap downloader:

- ``results.csv`` columns: ``date, home_team, away_team, home_score,
  away_score, tournament, city, country, neutral``
  (a few historical versions also carry ``winner``; we ignore it).
- ``shootouts.csv`` columns: ``date, home_team, away_team, winner``
  (``winner`` is the team that won the penalty shootout).

Both files are joined on ``(date, home_team, away_team)`` to determine
the advancer for drawn knockout matches. All team names flow through
the :class:`AliasRegistry` for canonicalization.
"""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.data.adapters.base import MatchDataProvider
from football_advance_predictor.data.aliases.alias_registry import (
    AliasRegistry,
    canonical_key,
)
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn

logger = get_logger(__name__)

_RESULTS_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date",),
    "home_team": ("home_team", "home", "home_team_name"),
    "away_team": ("away_team", "away", "away_team_name"),
    "home_score": ("home_score", "home_goals"),
    "away_score": ("away_score", "away_goals"),
    "tournament": ("tournament", "competition"),
    "city": ("city",),
    "country": ("country",),
    "neutral": ("neutral", "neutral_venue"),
}

_SHOOTOUTS_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date",),
    "home_team": ("home_team", "home", "home_team_name"),
    "away_team": ("away_team", "away", "away_team_name"),
    "winner": ("winner", "winning_team"),
}

_RESULTS_LOOKUP: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _RESULTS_COLUMN_ALIASES.items()
    for alias in aliases
}
_SHOOTOUTS_LOOKUP: dict[str, str] = {
    alias: canonical
    for canonical, aliases in _SHOOTOUTS_COLUMN_ALIASES.items()
    for alias in aliases
}

SOURCE_NAME = "martj42"


class MartJ42ResultsProvider(MatchDataProvider):
    """Read international results from the martj42 dataset.

    Args:
        results_path: Path to ``results.csv``.
        shootouts_path: Path to ``shootouts.csv``. If the file does not
            exist (or is empty), shootout lookups silently return
            ``None`` and drawn matches are skipped.
        alias_registry: Optional :class:`AliasRegistry` used for
            team-name canonicalization. If not provided, an in-memory
            default registry is used.
    """

    name = "martj42_international_results"

    def __init__(
        self,
        results_path: str | Path,
        shootouts_path: str | Path,
        alias_registry: AliasRegistry | None = None,
    ) -> None:
        self.results_path = Path(results_path)
        self.shootouts_path = Path(shootouts_path)
        self._registry = alias_registry or AliasRegistry.open(
            self.results_path.parent / "_aliases_martj42"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_matches(self, **kwargs: Any) -> list[MatchIn]:
        """Return all rows from ``results.csv`` as ``MatchIn`` records."""
        raw_rows = self._read_csv(self.results_path, _RESULTS_LOOKUP)
        matches: list[MatchIn] = []
        for idx, row in enumerate(raw_rows):
            try:
                matches.append(self._row_to_match(row))
            except Exception as exc:
                logger.warning(
                    "Failed to parse martj42 results row",
                    extra={
                        "row_index": idx,
                        "error": str(exc),
                        "row": row,
                    },
                )
        logger.info(
            "Loaded martj42 results",
            extra={"n_matches": len(matches), "path": str(self.results_path)},
        )
        return matches

    def fetch_results(self, **kwargs: Any) -> list[MatchResultIn]:
        """Return :class:`MatchResultIn` records for matches with both scores.

        Drawn matches are only emitted if a shootout record exists (so
        we can attribute the advancer); otherwise the result is skipped
        (treated as a non-knockout draw).
        """
        raw_rows = self._read_csv(self.results_path, _RESULTS_LOOKUP)
        shootout_index = self._load_shootout_index()
        results: list[MatchResultIn] = []
        for idx, row in enumerate(raw_rows):
            try:
                match = self._row_to_match(row)
                result = self._match_to_result(match, row, shootout_index)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.warning(
                    "Failed to parse martj42 result row",
                    extra={
                        "row_index": idx,
                        "error": str(exc),
                        "row": row,
                    },
                )
        logger.info(
            "Loaded martj42 results records",
            extra={
                "n_results": len(results),
                "path": str(self.results_path),
                "n_shootouts": sum(len(v) for v in shootout_index.values()),
            },
        )
        return results

    def fetch_teams(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return observed team metadata from both files."""
        raw_rows = self._read_csv(self.results_path, _RESULTS_LOOKUP)
        names: set[str] = set()
        for row in raw_rows:
            for col in ("home_team", "away_team"):
                val = row.get(col)
                if val:
                    names.add(val)
        return [
            {
                "team_id": self._registry.resolve(name, source=SOURCE_NAME),
                "canonical_name": name,
            }
            for name in sorted(names)
        ]

    # ------------------------------------------------------------------
    # CSV I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _read_csv(path: Path, lookup: dict[str, str]) -> list[dict[str, str]]:
        if not path.exists():
            raise FileNotFoundError(f"martj42 CSV not found: {path}")
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            out: list[dict[str, str]] = []
            for raw_row in reader:
                normalised: dict[str, str] = {}
                for key, value in raw_row.items():
                    if key is None:
                        continue
                    canonical = lookup.get(key.lower().strip(), key.lower().strip())
                    if value is not None:
                        normalised[canonical] = value
                out.append(normalised)
            return out

    def _load_shootout_index(self) -> dict[tuple[str, str, str], str]:
        """Index shootouts by (date_iso, home_key, away_key) -> winning team raw name.

        The keys are case-insensitive and accent-normalized so we can
        look them up without re-running the alias registry on every row.
        """
        if not self.shootouts_path.exists():
            logger.debug(
                "martj42 shootouts file missing; drawn matches will be skipped",
                extra={"path": str(self.shootouts_path)},
            )
            return {}
        rows = self._read_csv(self.shootouts_path, _SHOOTOUTS_LOOKUP)
        index: dict[tuple[str, str, str], str] = {}
        for row in rows:
            date = (row.get("date") or "").strip()
            home = (row.get("home_team") or "").strip()
            away = (row.get("away_team") or "").strip()
            winner = (row.get("winner") or "").strip()
            if not (date and home and away and winner):
                continue
            key = (
                date,
                canonical_key(home),
                canonical_key(away),
            )
            index[key] = winner
        return index

    # ------------------------------------------------------------------
    # Row -> schema mapping
    # ------------------------------------------------------------------

    def _row_to_match(self, row: dict[str, str]) -> MatchIn:
        raw_date = row.get("date")
        if not raw_date:
            raise ValueError("Missing date")
        kickoff = self._parse_date(raw_date)
        home_raw = (row.get("home_team") or "").strip()
        away_raw = (row.get("away_team") or "").strip()
        if not home_raw or not away_raw:
            raise ValueError("Missing home_team or away_team")
        home_team_id = self._registry.resolve(home_raw, source=SOURCE_NAME)
        away_team_id = self._registry.resolve(away_raw, source=SOURCE_NAME)
        match_id = self._build_match_id(raw_date, home_raw, away_raw)
        tournament = (row.get("tournament") or "unknown").strip() or "unknown"
        competition_id = self._slugify(tournament)
        home_goals = self._parse_int(row.get("home_score"))
        away_goals = self._parse_int(row.get("away_score"))
        winner_team_id: str | None = None
        if home_goals is not None and away_goals is not None and home_goals != away_goals:
            winner_team_id = home_team_id if home_goals > away_goals else away_team_id
        return MatchIn(
            match_id=match_id,
            kickoff_at=kickoff,
            competition_id=competition_id,
            stage="unknown",
            season_or_year=str(kickoff.year),
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=home_goals,
            away_goals=away_goals,
            winner_team_id=winner_team_id,
            advancing_team_id=None,
            neutral_venue=self._parse_bool(row.get("neutral")),
            venue_name=None,
            city=(row.get("city") or None),
            country=(row.get("country") or None),
            source=SOURCE_NAME,
        )

    def _match_to_result(
        self,
        match: MatchIn,
        row: dict[str, str],
        shootout_index: dict[tuple[str, str, str], str],
    ) -> MatchResultIn | None:
        """Build a :class:`MatchResultIn` or return ``None`` if not derivable.

        Drawn matches require a shootout record to determine the
        advancer; otherwise we conservatively skip (likely a group
        stage draw).
        """
        if match.home_goals is None or match.away_goals is None:
            logger.debug(
                "Skipping martj42 result: missing goals",
                extra={"match_id": match.match_id},
            )
            return None

        home_advances: bool
        if match.home_goals == match.away_goals:
            shootout_key = (
                row.get("date", ""),
                canonical_key(row.get("home_team", "")),
                canonical_key(row.get("away_team", "")),
            )
            winner_raw = shootout_index.get(shootout_key)
            if winner_raw is None:
                logger.debug(
                    "Skipping martj42 result: draw with no shootout record",
                    extra={"match_id": match.match_id},
                )
                return None
            winner_id = self._registry.resolve(winner_raw, source=SOURCE_NAME)
            home_advances = winner_id == match.home_team_id
        else:
            home_advances = match.home_goals > match.away_goals

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value: str) -> datetime:
        """Parse a ``YYYY-MM-DD`` (or ISO 8601) date into UTC midnight."""
        text = value.strip()
        # The dataset uses bare dates; fromisoformat handles them.
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"Invalid date: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt

    @staticmethod
    def _parse_int(value: str | None) -> int | None:
        if value is None:
            return None
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            try:
                return int(float(text))
            except ValueError:
                return None

    @staticmethod
    def _parse_bool(value: str | None) -> bool:
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "t", "yes", "y"}

    @staticmethod
    def _slugify(value: str) -> str:
        """Slugify a tournament name (mirrors :func:`canonical_key`)."""
        return canonical_key(value).replace(" ", "_") or "unknown"

    @staticmethod
    def _build_match_id(date: str, home: str, away: str) -> str:
        """Stable match id from (date, home, away) without alias rewriting."""
        key = canonical_key(home) + "_" + canonical_key(away) + "_" + date.strip()
        # Replace any remaining characters that would be awkward in a column.
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in key)
        return f"MJ42_{safe}"


def collect_observed_aliases(
    results_path: str | Path,
    shootouts_path: str | Path,
) -> dict[str, set[str]]:
    """Return the set of raw names observed in both files (helper for tests).

    Returned as ``{"results": {home1, away1, ...}, "shootouts": {...}}``.
    """
    results_rows = MartJ42ResultsProvider._read_csv(Path(results_path), _RESULTS_LOOKUP)
    shootout_rows = (
        MartJ42ResultsProvider._read_csv(Path(shootouts_path), _SHOOTOUTS_LOOKUP)
        if Path(shootouts_path).exists()
        else []
    )
    results_names: set[str] = set()
    for row in results_rows:
        for col in ("home_team", "away_team"):
            val = (row.get(col) or "").strip()
            if val:
                results_names.add(val)
    shootout_names: set[str] = set()
    for row in shootout_rows:
        for col in ("home_team", "away_team", "winner"):
            val = (row.get(col) or "").strip()
            if val:
                shootout_names.add(val)
    return {"results": results_names, "shootouts": shootout_names}
