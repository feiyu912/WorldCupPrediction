"""Adapter for the Open Football JSON tournament format.

Reads JSON files such as ``world-cup.json`` / ``euro-cup.json`` published
under the openfootball project. The schema is a top-level dict with a
``name`` and a list of ``rounds``, each containing ``matches`` with rich
score information (``score1``, ``score1et``, ``score1pen``) and an
explicit ``winner`` field (``1`` or ``2``).

The Open Football "home" team is ``team1``; international tournaments
use neutral venues, so we treat all matches as neutral.
"""

from __future__ import annotations

import json
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

SOURCE_NAME = "openfootball"


class OpenFootballTournamentProvider(MatchDataProvider):
    """Read a single Open Football tournament JSON file.

    Args:
        path: Path to the JSON file (``world-cup.json``,
            ``euro-cup.json``, etc.).
        alias_registry: Optional :class:`AliasRegistry` used for
            team-name canonicalization. If not provided, an in-memory
            default registry is used.
        tournament_name: Optional override for the tournament name.
            Defaults to the top-level ``name`` field in the JSON file.
    """

    name = "openfootball_tournament"

    def __init__(
        self,
        path: str | Path,
        alias_registry: AliasRegistry | None = None,
        tournament_name: str | None = None,
    ) -> None:
        self.path = Path(path)
        self._tournament_name_override = tournament_name
        self._registry = alias_registry or AliasRegistry.open(
            self.path.parent / "_aliases_openfootball"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_matches(self, **kwargs: Any) -> list[MatchIn]:
        """Return all matches in the tournament as ``MatchIn`` records."""
        document = self._load_document()
        tournament_name = self._resolve_tournament_name(document)
        competition_id = canonical_key(tournament_name).replace(" ", "_") or "unknown"
        raw_matches = self._iter_matches(document)
        matches: list[MatchIn] = []
        for idx, (round_name, raw) in enumerate(raw_matches):
            try:
                matches.append(self._match_to_in(raw, round_name, competition_id))
            except Exception as exc:
                logger.warning(
                    "Failed to parse openfootball match",
                    extra={
                        "row_index": idx,
                        "round": round_name,
                        "error": str(exc),
                        "match_key": raw.get("key"),
                    },
                )
        logger.info(
            "Loaded openfootball tournament",
            extra={
                "tournament": tournament_name,
                "n_matches": len(matches),
                "path": str(self.path),
            },
        )
        return matches

    def fetch_results(self, **kwargs: Any) -> list[MatchResultIn]:
        """Return :class:`MatchResultIn` records for matches with valid scores."""
        document = self._load_document()
        tournament_name = self._resolve_tournament_name(document)
        competition_id = canonical_key(tournament_name).replace(" ", "_") or "unknown"
        raw_matches = self._iter_matches(document)
        results: list[MatchResultIn] = []
        for idx, (round_name, raw) in enumerate(raw_matches):
            try:
                match = self._match_to_in(raw, round_name, competition_id)
                result = self._match_to_result(match, raw)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.warning(
                    "Failed to parse openfootball result",
                    extra={
                        "row_index": idx,
                        "round": round_name,
                        "error": str(exc),
                        "match_key": raw.get("key"),
                    },
                )
        logger.info(
            "Loaded openfootball results",
            extra={
                "tournament": tournament_name,
                "n_results": len(results),
                "path": str(self.path),
            },
        )
        return results

    def fetch_teams(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Return observed team metadata from the tournament."""
        document = self._load_document()
        names: set[str] = set()
        for _round_name, raw in self._iter_matches(document):
            for col in ("team1", "team2"):
                team_obj = raw.get(col) or {}
                name = (team_obj.get("name") or "").strip() if isinstance(team_obj, dict) else ""
                if name:
                    names.add(name)
        return [
            {
                "team_id": self._registry.resolve(name, source=SOURCE_NAME),
                "canonical_name": name,
            }
            for name in sorted(names)
        ]

    # ------------------------------------------------------------------
    # Document I/O
    # ------------------------------------------------------------------

    def _load_document(self) -> dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"openfootball JSON not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected top-level dict in {self.path}, got {type(data).__name__}"
            )
        # Two valid layouts:
        # 1) {"name": ..., "rounds": [{"name": ..., "matches": [...]}, ...]}
        # 2) {"name": ..., "matches": [{"round": ..., ...}, ...]}
        has_rounds = isinstance(data.get("rounds"), list)
        has_matches = isinstance(data.get("matches"), list)
        if not has_rounds and not has_matches:
            raise ValueError(
                f"Missing or invalid 'rounds'/'matches' in {self.path}"
            )
        return data

    def _resolve_tournament_name(self, document: dict[str, Any]) -> str:
        if self._tournament_name_override:
            return self._tournament_name_override
        name = document.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Missing top-level 'name' in {self.path}; pass tournament_name explicitly"
            )
        return name.strip()

    def _iter_matches(self, document: dict[str, Any]):
        """Yield ``(round_name, match_dict)`` for every match in the document.

        Supports both layouts:
        - rounds: each round is {"name": str, "matches": [...]}
        - matches: each match has its own "round" field
        """
        for round_obj in document.get("rounds", []) or []:
            round_name = (
                (round_obj.get("name") or "unknown")
                if isinstance(round_obj, dict)
                else "unknown"
            )
            for match in round_obj.get("matches", []) or []:
                if not isinstance(match, dict):
                    continue
                yield round_name, match
        for match in document.get("matches", []) or []:
            if not isinstance(match, dict):
                continue
            round_name = match.get("round") or "unknown"
            yield round_name, match

    # ------------------------------------------------------------------
    # Match -> schema mapping
    # ------------------------------------------------------------------

    def _match_to_in(
        self,
        raw: dict[str, Any],
        round_name: str,
        competition_id: str,
    ) -> MatchIn:
        kickoff = self._parse_datetime(raw.get("date"))
        home_team_id, home_raw = self._resolve_team(raw.get("team1"))
        away_team_id, away_raw = self._resolve_team(raw.get("team2"))
        home_goals = self._parse_int(raw.get("score1"))
        away_goals = self._parse_int(raw.get("score2"))
        match_id = self._build_match_id(raw, kickoff, home_raw, away_raw)
        winner_team_id: str | None = None
        if home_goals is not None and away_goals is not None and home_goals != away_goals:
            winner_team_id = home_team_id if home_goals > away_goals else away_team_id
        return MatchIn(
            match_id=match_id,
            kickoff_at=kickoff,
            competition_id=competition_id,
            stage=round_name,
            season_or_year=str(kickoff.year),
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            home_goals=home_goals,
            away_goals=away_goals,
            winner_team_id=winner_team_id,
            advancing_team_id=None,
            neutral_venue=True,
            venue_name=None,
            city=None,
            country=None,
            source=SOURCE_NAME,
        )

    def _match_to_result(self, match: MatchIn, raw: dict[str, Any]) -> MatchResultIn | None:
        """Build a :class:`MatchResultIn` or return ``None``.

        Requires valid 90-minute scores (``score1`` / ``score2``). The
        advancer is derived from the explicit ``winner`` field (``1`` /
        ``2``); falls back to score comparison when the field is absent
        or the match is a draw.
        """
        if match.home_goals is None or match.away_goals is None:
            logger.debug(
                "Skipping openfootball result: missing scores",
                extra={"match_id": match.match_id},
            )
            return None

        home_goals_et = self._parse_int(raw.get("score1et"))
        away_goals_et = self._parse_int(raw.get("score2et"))
        penalties_home = self._parse_int(raw.get("score1pen"))
        penalties_away = self._parse_int(raw.get("score2pen"))

        winner_field = raw.get("winner")
        home_advances: bool
        if isinstance(winner_field, int) and winner_field in (1, 2):
            home_advances = winner_field == 1
        elif isinstance(winner_field, str) and winner_field.strip() in ("1", "2"):
            home_advances = winner_field.strip() == "1"
        else:
            match_type = (raw.get("type") or "").strip().lower()
            if match.home_goals == match.away_goals and match_type == "draw":
                logger.debug(
                    "Skipping openfootball result: drawn group match",
                    extra={"match_id": match.match_id, "stage": match.stage},
                )
                return None
            if match.home_goals == match.away_goals:
                # No explicit winner for a drawn knockout match; cannot
                # derive safely.
                logger.debug(
                    "Skipping openfootball result: drawn knockout with no winner",
                    extra={"match_id": match.match_id, "stage": match.stage},
                )
                return None
            home_advances = match.home_goals > match.away_goals

        return MatchResultIn(
            match_id=match.match_id,
            final_status="final",
            home_goals_90=match.home_goals,
            away_goals_90=match.away_goals,
            home_goals_et=home_goals_et,
            away_goals_et=away_goals_et,
            penalties_home=penalties_home,
            penalties_away=penalties_away,
            home_advances=home_advances,
            result_verified_at=to_utc(match.kickoff_at),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_team(self, team_obj: Any) -> tuple[str, str]:
        """Return ``(canonical_team_id, raw_name)`` for a team dict/string."""
        if isinstance(team_obj, dict):
            name = (team_obj.get("name") or "").strip()
        elif isinstance(team_obj, str):
            name = team_obj.strip()
        else:
            name = ""
        if not name:
            raise ValueError("Missing team name")
        return self._registry.resolve(name, source=SOURCE_NAME), name

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if value is None:
            raise ValueError("Missing date")
        text = str(value).strip()
        if not text:
            raise ValueError("Missing date")
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid date: {value!r}") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        else:
            dt = dt.astimezone(UTC)
        return dt

    @staticmethod
    def _parse_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            # bool is a subclass of int; treat True/False as missing.
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value != value:  # NaN
                return None
            return int(value)
        text = str(value).strip()
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
    def _build_match_id(
        raw: dict[str, Any],
        kickoff: datetime,
        home_raw: str,
        away_raw: str,
    ) -> str:
        key = raw.get("key")
        if isinstance(key, str) and key.strip():
            slug = canonical_key(key).replace(" ", "_")
            return f"OF_{slug}"
        # Fallback: composite of date + canonical team names.
        safe = "_".join(
            [
                kickoff.strftime("%Y%m%d"),
                canonical_key(home_raw).replace(" ", "_"),
                canonical_key(away_raw).replace(" ", "_"),
            ]
        )
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in safe)
        return f"OF_{safe}"
