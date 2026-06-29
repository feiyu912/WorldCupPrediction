"""Knockout manifest builder.

Merges tournament-specific result providers into a single,
deduplicated list of reliably-labeled knockout fixtures. The
manifest is the input to training and backtesting of the advance
predictor.

The builder never touches the network or the database — it operates
on in-memory lists returned by the registered providers.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.data.aliases.alias_registry import AliasRegistry
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage classification
# ---------------------------------------------------------------------------

# Substrings (lowercased) that mark a fixture as a knockout stage.
_KNOCKOUT_TERMS: tuple[str, ...] = (
    "round of 16",
    "r16",
    "round-of-16",
    "quarter",
    "qf",
    "quarter-final",
    "quarterfinal",
    "semi",
    "sf",
    "semi-final",
    "semifinal",
    "final",
    "3rd place",
    "third place",
    "play-off",
    "playoff",
    "knockout",
)


def is_knockout_stage(stage: str) -> bool:
    """Return True if ``stage`` contains any knockout term (case-insensitive)."""
    text = (stage or "").strip().lower()
    if not text:
        return False
    return any(term in text for term in _KNOCKOUT_TERMS)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class KnockoutRow:
    """A single reliably-labeled knockout fixture."""

    match_id: str
    kickoff_at: datetime
    competition_id: str
    competition_name: str
    stage: str
    season_or_year: str
    home_team_id: str
    away_team_id: str
    home_goals_90: int | None
    away_goals_90: int | None
    home_advances: bool
    source: str  # which provider contributed it


@dataclass
class QuarantineReason:
    """A single knockout fixture that was rejected and the reason."""

    raw_match_id: str
    reason: str
    detail: str


@dataclass
class KnockoutManifest:
    """Merged, deduplicated list of knockout fixtures."""

    rows: list[KnockoutRow] = field(default_factory=list)
    tournament_coverage: dict[str, int] = field(default_factory=dict)
    quarantined: list[QuarantineReason] = field(default_factory=list)
    total: int = 0


# ---------------------------------------------------------------------------
# Provider protocol (duck-typed)
# ---------------------------------------------------------------------------


class _ResultProvider(Protocol):
    """Minimal provider surface the builder relies on."""

    def fetch_matches(self) -> Iterable[MatchIn]: ...
    def fetch_results(self) -> Iterable[MatchResultIn]: ...


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify_competition(value: str) -> str:
    """Return an alphanum-only lowercase slug suitable for a match id."""
    text = (value or "").strip().lower()
    text = _SLUG_NON_ALNUM.sub("", text)
    return text or "unknown"


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class KnockoutManifestBuilder:
    """Build a deduplicated knockout manifest from one or more providers.

    Providers are processed in the order they are added (registration
    order). For each (kickoff_at, home_team_id, away_team_id) tuple,
    the first provider that contributes a row wins; subsequent
    matches on the same key are quarantined with reason
    ``"duplicate_across_providers"``.
    """

    def __init__(self, alias_registry: AliasRegistry) -> None:
        self.alias_registry = alias_registry
        self._providers: list[tuple[str, Any]] = []

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def add_provider(self, name: str, provider: Any) -> None:
        """Register a result provider.

        The provider is duck-typed: it must expose ``fetch_matches()``
        and ``fetch_results()``. The ``name`` is recorded as the
        ``source`` of any row contributed by this provider.
        """
        if provider is None:
            raise ValueError("Provider must not be None")
        for attr in ("fetch_matches", "fetch_results"):
            if not hasattr(provider, attr):
                raise TypeError(
                    f"Provider {name!r} is missing required method {attr!r}."
                )
        self._providers.append((name, provider))
        logger.info("Registered knockout provider", extra={"provider": name})

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> KnockoutManifest:
        """Run all providers, dedupe, and return a populated manifest."""
        manifest = KnockoutManifest()

        # (kickoff_at isoformat, home_id, away_id) -> generated match_id
        # so subsequent collisions can be quarantined.
        seen_keys: set[tuple[str, str, str]] = set()
        # match_id -> True for collision suffix tracking.
        seen_ids: set[str] = set()

        for provider_name, provider in self._providers:
            try:
                matches_iter = provider.fetch_matches()
                matches = list(matches_iter) if matches_iter is not None else []
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Provider fetch_matches failed",
                    extra={"provider": provider_name, "error": str(exc)},
                )
                matches = []

            try:
                results_iter = provider.fetch_results()
                results = list(results_iter) if results_iter is not None else []
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "Provider fetch_results failed",
                    extra={"provider": provider_name, "error": str(exc)},
                )
                results = []

            matches_by_id: dict[str, MatchIn] = {}
            for m in matches:
                mid = getattr(m, "match_id", None)
                if mid:
                    matches_by_id[mid] = m

            for result in results:
                raw_id = getattr(result, "match_id", "") or ""
                match = matches_by_id.get(raw_id)
                if match is None:
                    continue  # nothing to attach a result to

                # 1. Knockout stage filter.
                stage = getattr(match, "stage", "") or ""
                if not is_knockout_stage(stage):
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=raw_id,
                            reason="not_knockout_stage",
                            detail=f"Stage {stage!r} is not a knockout stage.",
                        )
                    )
                    continue

                kickoff = getattr(match, "kickoff_at", None)
                if not isinstance(kickoff, datetime):
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=raw_id,
                            reason="missing_kickoff",
                            detail="Match has no kickoff_at timestamp.",
                        )
                    )
                    continue

                home_raw = getattr(match, "home_team_id", "") or ""
                away_raw = getattr(match, "away_team_id", "") or ""
                home_id = self._canonicalize(home_raw)
                away_id = self._canonicalize(away_raw)

                # 3. Unknown team quarantine.
                if not home_id or home_id == "unknown" or not away_id or away_id == "unknown":
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=raw_id,
                            reason="unknown_team",
                            detail=f"home={home_raw!r} away={away_raw!r}",
                        )
                    )
                    continue

                # 4. Missing scores quarantine.
                home_goals_90 = getattr(result, "home_goals_90", None)
                away_goals_90 = getattr(result, "away_goals_90", None)
                if home_goals_90 is None or away_goals_90 is None:
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=raw_id,
                            reason="missing_scores",
                            detail=(
                                f"home_goals_90={home_goals_90!r} "
                                f"away_goals_90={away_goals_90!r}"
                            ),
                        )
                    )
                    continue

                # 5. Determine home_advances.
                home_advances = self._resolve_advancer(result, home_goals_90, away_goals_90)
                if home_advances is None:
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=raw_id,
                            reason="no_advancer_on_draw",
                            detail=(
                                f"90-minute draw with no shootout result "
                                f"({home_goals_90}-{away_goals_90})."
                            ),
                        )
                    )
                    continue

                # 2. Duplicate key quarantine (full kickoff_at isoformat).
                date_key = kickoff.isoformat()
                dedupe_key = (date_key, home_id, away_id)
                if dedupe_key in seen_keys:
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=raw_id,
                            reason="duplicate_across_providers",
                            detail=(
                                f"({date_key}, {home_id}, {away_id}) "
                                f"already contributed by a prior provider."
                            ),
                        )
                    )
                    continue
                seen_keys.add(dedupe_key)

                # 6. Build the match_id with collision suffix.
                competition_id = getattr(match, "competition_id", "") or ""
                competition_name = (
                    competition_id
                    or getattr(match, "competition_name", None)
                    or getattr(provider, "tournament_name", None)
                    or competition_id
                    or "unknown"
                )
                slug = _slugify_competition(
                    competition_id or competition_name
                )
                date_slug = kickoff.strftime("%Y%m%d")
                base_id = f"{slug}_{date_slug}_{home_id}_{away_id}"
                generated_id = base_id
                suffix = 1
                while generated_id in seen_ids:
                    suffix += 1
                    generated_id = f"{base_id}-{suffix}"
                seen_ids.add(generated_id)

                season_or_year = (
                    getattr(match, "season_or_year", None)
                    or str(kickoff.year)
                )

                row = KnockoutRow(
                    match_id=generated_id,
                    kickoff_at=kickoff,
                    competition_id=competition_id or competition_name,
                    competition_name=competition_name,
                    stage=stage,
                    season_or_year=season_or_year,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_goals_90=home_goals_90,
                    away_goals_90=away_goals_90,
                    home_advances=bool(home_advances),
                    source=provider_name,
                )
                manifest.rows.append(row)
                manifest.tournament_coverage[competition_name] = (
                    manifest.tournament_coverage.get(competition_name, 0) + 1
                )

        manifest.total = len(manifest.rows)
        logger.info(
            "Built knockout manifest",
            extra={
                "rows": manifest.total,
                "tournaments": len(manifest.tournament_coverage),
                "quarantined": len(manifest.quarantined),
            },
        )
        return manifest

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _canonicalize(self, raw: str) -> str:
        if not raw:
            return "unknown"
        try:
            return self.alias_registry.resolve(raw)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Alias resolution failed",
                extra={"raw": raw, "error": str(exc)},
            )
            return raw or "unknown"

    @staticmethod
    def _resolve_advancer(
        result: MatchResultIn,
        home_goals_90: int,
        away_goals_90: int,
    ) -> bool | None:
        """Return True/False for home-advances, or None if undetermined.

        Resolution order:
        1. ``result.home_advances`` if explicitly set on the result record.
        2. If 90-minute goals decide a winner, use that.
        3. If 90-minute goals are tied, use the penalty shootout result.
        4. Otherwise ``None`` (caller quarantines with ``no_advancer_on_draw``).
        """
        # 1. Provider-supplied value wins when present.
        try:
            value = getattr(result, "home_advances", None)
            if value is True or value is False:
                return bool(value)
        except Exception:
            pass

        # 2. Winner by 90-minute goals.
        if home_goals_90 > away_goals_90:
            return True
        if home_goals_90 < away_goals_90:
            return False

        # 3. Shootout decides a 90-minute draw.
        pen_home = getattr(result, "penalties_home", None)
        pen_away = getattr(result, "penalties_away", None)
        if pen_home is None or pen_away is None:
            return None
        if pen_home > pen_away:
            return True
        if pen_home < pen_away:
            return False
        return None
