"""StatsBomb open-data provider (defensive, local-clone only).

Reads a local clone of the public ``statsbomb/open-data`` GitHub
repository (typically produced by the bootstrap downloader) and
derives xG-style features only where event coverage exists.

The provider is **defensive by design**: it never raises on missing
data and never touches the network. Every public method returns a
dict containing explicit missingness flags so callers can treat the
features as optional.

Directory layout used by the provider (matches the upstream repo)::

    <root>/matches/<competition_id>/<season_id>/<match_id>.json
    <root>/events/<match_id>.json
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc

logger = get_logger(__name__)


# Set-piece play-pattern names treated as set pieces for set_piece_xG.
_SET_PIECE_PATTERNS: frozenset[str] = frozenset({"From Free Kick", "From Corner"})

# Per-match feature keys produced by ``match_features``.
_MATCH_FEATURE_KEYS: tuple[str, ...] = (
    "xg_home",
    "xg_away",
    "shots_home",
    "shots_away",
    "shots_in_box_home",
    "shots_in_box_away",
    "set_piece_xg_home",
    "set_piece_xg_away",
)

# Aggregate feature keys produced by ``aggregate_team_match_features``.
_AGG_FEATURE_KEYS: tuple[str, ...] = (
    "xg_total",
    "shots_total",
    "shots_in_box_total",
    "set_piece_xg_total",
    "matches_used",
)


def _empty_match_features(*, available: bool) -> dict[str, float]:
    """Per-match feature dict with zeros and the availability flag."""
    out: dict[str, float] = dict.fromkeys(_MATCH_FEATURE_KEYS, 0.0)
    out["xg_difference"] = 0.0
    out["statsbomb_available"] = float(available)
    return out


def _empty_agg_features(*, available: bool) -> dict[str, float]:
    """Aggregate feature dict with NaN numeric features and the availability flag.

    NaN (not zero) preserves the distinction between an unavailable
    metric and a real observed zero. The model layer's missingness
    indicator columns + train-fold-only imputation handle NaN.
    """
    out: dict[str, float] = {k: float("nan") for k in _AGG_FEATURE_KEYS}
    out["statsbomb_available"] = float(available)
    return out


class StatsBombOpenDataProvider:
    """Read StatsBomb open-data from a local git clone.

    The provider never touches the network or the database. If the
    repository has not been cloned, every method returns an empty/flag
    dict and logs a warning the first time it is called.
    """

    name = "statsbomb_open_data"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        # match_id -> (competition_id, season_id) so we can locate the
        # metadata file without scanning the whole tree on every call.
        self._index: dict[str, tuple[str, str]] | None = None
        self._indexed_root: Path | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def competition_ids(self) -> list[str]:
        """Return the sorted list of competition directory names present locally.

        Returns an empty list (and logs a warning) if the ``matches/``
        directory is missing.
        """
        matches_dir = self.root / "matches"
        if not matches_dir.is_dir():
            logger.warning(
                "StatsBomb matches directory not found",
                extra={"root": str(self.root)},
            )
            return []
        return sorted(p.name for p in matches_dir.iterdir() if p.is_dir())

    def match_features(self, match_id: str) -> dict[str, float]:
        """Return per-match xG-style features for ``match_id``.

        Returns a dict with the keys listed in
        :data:`_MATCH_FEATURE_KEYS` plus ``xg_difference`` and a
        ``statsbomb_available`` flag. If the match metadata or events
        file is missing, every numeric feature is ``0.0`` and
        ``statsbomb_available`` is ``0.0``.
        """
        if not match_id:
            return _empty_match_features(available=False)

        meta = self._match_meta_for(match_id)
        if meta is None:
            return _empty_match_features(available=False)

        events_path = self._events_path_for(match_id)
        if events_path is None or not events_path.is_file():
            logger.warning(
                "StatsBomb events file missing",
                extra={
                    "match_id": match_id,
                    "expected": str(events_path) if events_path else None,
                },
            )
            return _empty_match_features(available=False)

        try:
            with events_path.open("r", encoding="utf-8") as f:
                events = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read StatsBomb events",
                extra={"match_id": match_id, "error": str(exc)},
            )
            return _empty_match_features(available=False)

        if not isinstance(events, list):
            logger.warning(
                "StatsBomb events payload is not a list",
                extra={"match_id": match_id, "type": type(events).__name__},
            )
            return _empty_match_features(available=False)

        home_id, away_id = self._team_ids_from_meta(meta)
        shots = [e for e in events if isinstance(e, dict) and self._is_shot(e)]

        xg_home = 0.0
        xg_away = 0.0
        shots_home = 0.0
        shots_away = 0.0
        shots_in_box_home = 0.0
        shots_in_box_away = 0.0
        set_piece_xg_home = 0.0
        set_piece_xg_away = 0.0

        for shot in shots:
            xg = self._shot_xg(shot)
            team_id = self._shot_team_id(shot)
            is_set_piece = self._is_set_piece_shot(shot)

            is_home = bool(home_id) and team_id == home_id
            is_away = bool(away_id) and team_id == away_id

            if is_home:
                # Add shot xg to home total.
                xg_home += xg
                shots_home += 1
                # Set-piece shots also count toward shots_in_box and set_piece_xg.
                if is_set_piece:
                    set_piece_xg_home += xg
                    shots_in_box_home += 1
            elif is_away:
                # Add shot xg to away total.
                xg_away += xg
                shots_away += 1
                if is_set_piece:
                    set_piece_xg_away += xg
                    shots_in_box_away += 1
            # Shots with an unresolvable team id are silently dropped so a
            # malformed event cannot break the feature contract.

        return {
            "xg_home": round(xg_home, 4),
            "xg_away": round(xg_away, 4),
            "xg_difference": round(xg_home - xg_away, 4),
            "shots_home": float(shots_home),
            "shots_away": float(shots_away),
            "shots_in_box_home": float(shots_in_box_home),
            "shots_in_box_away": float(shots_in_box_away),
            "set_piece_xg_home": round(set_piece_xg_home, 4),
            "set_piece_xg_away": round(set_piece_xg_away, 4),
            "statsbomb_available": 1.0,
        }

    def aggregate_team_match_features(
        self,
        team_id: str,
        *,
        before: datetime,
    ) -> dict[str, float]:
        """Sum xG / shot features for ``team_id`` across matches ending before ``before``.

        Iterates over the local match index, reads each events file,
        and aggregates the team's shots. If the clone is absent, the
        team has no matches, or ``team_id`` is empty/unknown, returns
        zero numeric features with ``statsbomb_available`` set to
        ``0.0``.
        """
        if not team_id or team_id == "unknown":
            return _empty_agg_features(available=False)

        cutoff = to_utc(before)
        if not self.root.is_dir():
            return _empty_agg_features(available=False)

        index = self._build_index()
        if not index:
            return _empty_agg_features(available=False)

        events_dir = self.root / "events"
        if not events_dir.is_dir():
            return _empty_agg_features(available=False)

        agg_xg = 0.0
        agg_shots = 0.0
        agg_in_box = 0.0
        agg_set_piece = 0.0
        n_matches = 0

        for match_id in index:
            events_path = events_dir / f"{match_id}.json"
            if not events_path.is_file():
                continue

            meta = self._match_meta_for(match_id)
            if meta is None:
                continue

            match_date = self._match_date_from_meta(meta)
            if match_date is None or match_date >= cutoff:
                continue

            home_id, away_id = self._team_ids_from_meta(meta)
            side: str | None = None
            if team_id == home_id:
                side = "home"
            elif team_id == away_id:
                side = "away"
            if side is None:
                continue

            try:
                with events_path.open("r", encoding="utf-8") as f:
                    events = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(events, list):
                continue

            team_side_id = home_id if side == "home" else away_id
            for shot in events:
                if not self._is_shot(shot):
                    continue
                if self._shot_team_id(shot) != team_side_id:
                    continue
                xg = self._shot_xg(shot)
                agg_xg += xg
                agg_shots += 1
                if self._is_set_piece_shot(shot):
                    agg_set_piece += xg
                    agg_in_box += 1
            n_matches += 1

        if n_matches == 0:
            return _empty_agg_features(available=False)

        return {
            "xg_total": round(agg_xg, 4),
            "shots_total": float(agg_shots),
            "shots_in_box_total": float(agg_in_box),
            "set_piece_xg_total": round(agg_set_piece, 4),
            "matches_used": float(n_matches),
            "statsbomb_available": 1.0,
        }

    def aggregate_team_match_features_safe(
        self, team_id: str, *, before: datetime
    ) -> dict[str, float | None]:
        """Like :meth:`aggregate_team_match_features` but returns ``None`` for unavailable
        numeric features so the caller can distinguish them from observed zeros.
        """
        agg = self.aggregate_team_match_features(team_id, before=before)
        if not agg.get("statsbomb_available", 0.0):
            return {k: None for k in _AGG_FEATURE_KEYS}
        return dict(agg)

    def coverage_for_match(self, match_id: str) -> dict[str, bool]:
        """Return per-aspect availability flags for a single match.

        The returned dict always contains:

        - ``match_metadata``: ``True`` if ``matches/.../<match_id>.json`` exists.
        - ``events``: ``True`` if ``events/<match_id>.json`` exists.
        - ``home_features``: ``True`` if both metadata and events are present
          AND the metadata contains a non-empty ``home_team.id``.
        - ``away_features``: ``True`` if both metadata and events are present
          AND the metadata contains a non-empty ``away_team.id``.
        """
        events_path = self._events_path_for(match_id)
        events_ok = events_path is not None and events_path.is_file()

        meta = self._match_meta_for(match_id)
        meta_ok = meta is not None

        home_id = ""
        away_id = ""
        if meta_ok:
            home_id, away_id = self._team_ids_from_meta(meta)

        return {
            "match_metadata": meta_ok,
            "events": events_ok,
            "home_features": meta_ok and events_ok and bool(home_id),
            "away_features": meta_ok and events_ok and bool(away_id),
        }

    # ------------------------------------------------------------------
    # File-system helpers
    # ------------------------------------------------------------------

    def _events_path_for(self, match_id: str) -> Path | None:
        if not match_id:
            return None
        if not (self.root / "events").is_dir():
            return None
        return self.root / "events" / f"{match_id}.json"

    def _match_meta_path_for(self, match_id: str) -> Path | None:
        """Locate ``matches/<competition>/<season>/<match_id>.json``.

        Returns ``None`` if no file can be found. Uses the cached
        index when available, otherwise falls back to a recursive walk
        under ``matches/``.
        """
        if not match_id or not (self.root / "matches").is_dir():
            return None

        index = self._build_index()
        cached = index.get(match_id)
        if cached is not None:
            comp_id, season_id = cached
            candidate = (
                self.root / "matches" / comp_id / season_id / f"{match_id}.json"
            )
            if candidate.is_file():
                return candidate

        for path in (self.root / "matches").rglob(f"{match_id}.json"):
            if path.is_file():
                return path
        return None

    def _match_meta_for(self, match_id: str) -> dict[str, Any] | None:
        path = self._match_meta_path_for(match_id)
        if path is None:
            return None
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "Failed to read StatsBomb match metadata",
                extra={"match_id": match_id, "error": str(exc)},
            )
            return None
        return data if isinstance(data, dict) else None

    def _build_index(self) -> dict[str, tuple[str, str]]:
        """Build a (cacheable) ``match_id -> (competition_id, season_id)`` map."""
        if self._index is not None and self._indexed_root == self.root:
            return self._index

        index: dict[str, tuple[str, str]] = {}
        matches_dir = self.root / "matches"
        if matches_dir.is_dir():
            for comp_dir in sorted(matches_dir.iterdir()):
                if not comp_dir.is_dir():
                    continue
                for season_dir in sorted(comp_dir.iterdir()):
                    if not season_dir.is_dir():
                        continue
                    for match_file in season_dir.glob("*.json"):
                        index[match_file.stem] = (comp_dir.name, season_dir.name)

        self._index = index
        self._indexed_root = self.root
        return index

    # ------------------------------------------------------------------
    # Event parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_shot(event: dict[str, Any]) -> bool:
        if not isinstance(event, dict):
            return False
        type_ = event.get("type")
        if not isinstance(type_, dict):
            return False
        return type_.get("name") == "Shot"

    @staticmethod
    def _shot_xg(shot: dict[str, Any]) -> float:
        shot_blob = shot.get("shot") or {}
        raw = shot_blob.get("statsbomb_xg")
        try:
            return float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _shot_team_id(shot: dict[str, Any]) -> str:
        team = shot.get("team") or {}
        team_id = team.get("id") if isinstance(team, dict) else None
        if team_id is None:
            return ""
        return str(team_id)

    @staticmethod
    def _is_set_piece_shot(shot: dict[str, Any]) -> bool:
        pattern = shot.get("play_pattern") or {}
        if not isinstance(pattern, dict):
            return False
        return pattern.get("name") in _SET_PIECE_PATTERNS

    # ------------------------------------------------------------------
    # Match metadata parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _team_ids_from_meta(meta: dict[str, Any]) -> tuple[str, str]:
        """Return ``(home_team_id, away_team_id)`` as strings.

        Falls back to empty strings if the metadata shape is unexpected.
        """
        home = meta.get("home_team") or {}
        away = meta.get("away_team") or {}
        home_id = home.get("id") if isinstance(home, dict) else None
        away_id = away.get("id") if isinstance(away, dict) else None
        return (
            str(home_id) if home_id is not None else "",
            str(away_id) if away_id is not None else "",
        )

    @staticmethod
    def _match_date_from_meta(meta: dict[str, Any]) -> datetime | None:
        raw = meta.get("match_date")
        if not raw or not isinstance(raw, str):
            return None
        try:
            return to_utc(raw)
        except ValueError:
            return None
