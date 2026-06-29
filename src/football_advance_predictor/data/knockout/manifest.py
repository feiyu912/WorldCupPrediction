"""Knockout manifest builder.

Generates a unified, deduped list of labeled knockout matches from
multiple provider sources (martj42 results, openfootball worldcup/euro/
copa-america/gold-cup). Knockout matches are identified by their stage
string.

The label ``home_wins_tie`` is set True when the home team wins the tie
AND advances to a downstream bracket destination. Third-place matches
have no downstream destination and are excluded from the default
training set.

The default training set is R16 + QF + SF + Final for each supported
tournament, which gives 15 matches per World Cup edition
(8 R16 + 4 QF + 2 SF + 1 Final = 15).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.data.aliases.alias_registry import AliasRegistry
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn

logger = get_logger(__name__)


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
)

# Substrings that disqualify a fixture even if it contains a knockout term.
_NON_KNOCKOUT_TERMS: tuple[str, ...] = (
    "group",
    "league",
    "round robin",
    "round-robin",
    "matchday",
    "third place",
    "3rd place",
)

# Substrings that mark a fixture as having a downstream bracket
# destination. Used to filter out third-place matches (which don't).
_DOWNSTREAM_KNOCKOUT_TERMS: tuple[str, ...] = (
    "round of 16",
    "quarter",
    "semi",
    "final",
)


def is_knockout_stage(stage: str) -> bool:
    """Return True if ``stage`` is a knockout stage with a downstream
    bracket destination.
    """
    text = (stage or "").strip().lower()
    if not text:
        return False
    if any(term in text for term in _NON_KNOCKOUT_TERMS):
        return False
    return any(term in text for term in _KNOCKOUT_TERMS)


def has_downstream_bracket(stage: str) -> bool:
    """Return True when the knockout winner advances to a later round.

    Used to filter out third-place finals, which have no downstream
    destination.
    """
    text = (stage or "").strip().lower()
    if not text:
        return False
    return any(term in text for term in _DOWNSTREAM_KNOCKOUT_TERMS)


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class KnockoutRow:
    """A single reliably-labeled knockout fixture.

    The label ``home_wins_tie`` is True iff the home team wins the
    tie and advances to a downstream bracket destination. Third-place
    matches are excluded from the default training set because they
    have no downstream destination.
    """

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
    home_wins_tie: bool
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_id": self.match_id,
            "kickoff_at": self.kickoff_at.isoformat(),
            "competition_id": self.competition_id,
            "competition_name": self.competition_name,
            "stage": self.stage,
            "season_or_year": self.season_or_year,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "home_goals_90": self.home_goals_90,
            "away_goals_90": self.away_goals_90,
            "home_wins_tie": self.home_wins_tie,
            "source": self.source,
        }


@dataclass
class QuarantineReason:
    """A reason a match was excluded from the manifest."""

    raw_match_id: str
    reason: str
    detail: str
    kickoff_at: str | None = None
    home_team_id: str | None = None
    away_team_id: str | None = None
    stage: str | None = None
    tournament_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_match_id": self.raw_match_id,
            "reason": self.reason,
            "detail": self.detail,
            "kickoff_at": self.kickoff_at,
            "home_team_id": self.home_team_id,
            "away_team_id": self.away_team_id,
            "stage": self.stage,
            "tournament_name": self.tournament_name,
        }


@dataclass
class KnockoutManifest:
    """A generated manifest of labeled knockout fixtures."""

    rows: list[KnockoutRow] = field(default_factory=list)
    tournament_coverage: dict[str, int] = field(default_factory=dict)
    quarantined: list[QuarantineReason] = field(default_factory=list)
    excluded_third_place: list[QuarantineReason] = field(default_factory=list)
    expected_vs_found: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "tournament_coverage": dict(self.tournament_coverage),
            "quarantined_count": len(self.quarantined),
            "excluded_third_place_count": len(self.excluded_third_place),
            "expected_vs_found": dict(self.expected_vs_found),
            "rows": [r.to_dict() for r in self.rows],
            "quarantined": [q.to_dict() for q in self.quarantined],
            "excluded_third_place": [q.to_dict() for q in self.excluded_third_place],
        }


class _Provider(Protocol):
    name: str
    tournament_name: str

    def fetch_matches(self) -> list[MatchIn]: ...
    def fetch_results(self) -> list[MatchResultIn]: ...


class KnockoutManifestBuilder:
    """Build a :class:`KnockoutManifest` from one or more provider sources.

    Args:
        alias_registry: System-owned team alias registry used to
            canonicalize team names.
        expected_per_tournament: Optional mapping of tournament name
            -> expected knockout count for reconciliation. Defaults
            to {R16 + QF + SF + Final} = 15 per edition.
    """

    DEFAULT_EXPECTED_PER_TOURNAMENT = 15

    def __init__(
        self,
        alias_registry: AliasRegistry,
        expected_per_tournament: dict[str, int] | None = None,
    ) -> None:
        self.alias_registry = alias_registry
        self._providers: list[_Provider] = []
        self._expected_per_tournament = expected_per_tournament or {}

    def add_provider(self, name: str, provider: Any) -> None:
        if provider is None:
            raise ValueError("Provider must not be None")
        for attr in ("fetch_matches", "fetch_results"):
            if not hasattr(provider, attr):
                raise TypeError(
                    f"Provider {name!r} is missing required method {attr!r}."
                )
        self._providers.append((name, provider))
        logger.info("Registered knockout provider", extra={"provider": name})

    def build(self) -> KnockoutManifest:
        """Run all providers, dedupe, and return a populated manifest."""
        manifest = KnockoutManifest()
        seen: set[tuple[datetime, str, str]] = set()
        tournament_counts: Counter[str] = Counter()

        for provider_name, provider in self._providers:
            tournament_name = getattr(provider, "tournament_name", provider_name)
            try:
                matches = provider.fetch_matches()
            except Exception as exc:
                manifest.quarantined.append(
                    QuarantineReason(
                        raw_match_id="<provider>",
                        reason="provider_fetch_matches_failed",
                        detail=f"{provider_name}: {exc}",
                        tournament_name=tournament_name,
                    )
                )
                continue
            try:
                results = {r.match_id: r for r in provider.fetch_results()}
            except Exception as exc:
                manifest.quarantined.append(
                    QuarantineReason(
                        raw_match_id="<provider>",
                        reason="provider_fetch_results_failed",
                        detail=f"{provider_name}: {exc}",
                        tournament_name=tournament_name,
                    )
                )
                results = {}

            for match in matches:
                stage = (match.stage or "").strip()
                kickoff_iso = (
                    match.kickoff_at.isoformat() if match.kickoff_at else None
                )
                home_id = self.alias_registry.resolve(match.home_team_id, source=provider_name)
                away_id = self.alias_registry.resolve(match.away_team_id, source=provider_name)

                if not is_knockout_stage(stage):
                    # Group or matchday stage — quietly ignore (no quarantine spam).
                    continue

                # Third-place finals are knockout stages but have no
                # downstream bracket destination. Exclude from the
                # default training set; record as a separate exclusion
                # bucket so the reconciliation report can show them.
                if not has_downstream_bracket(stage):
                    manifest.excluded_third_place.append(
                        QuarantineReason(
                            raw_match_id=match.match_id,
                            reason="third_place_no_downstream_bracket",
                            detail=f"stage={stage!r} excluded from default training set",
                            kickoff_at=kickoff_iso,
                            home_team_id=home_id,
                            away_team_id=away_id,
                            stage=stage,
                            tournament_name=tournament_name,
                        )
                    )
                    continue

                if home_id == "unknown" or away_id == "unknown":
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=match.match_id,
                            reason="unknown_team",
                            detail=f"home={home_id!r} away={away_id!r}",
                            kickoff_at=kickoff_iso,
                            home_team_id=home_id,
                            away_team_id=away_id,
                            stage=stage,
                            tournament_name=tournament_name,
                        )
                    )
                    continue

                kickoff = to_utc(match.kickoff_at)
                key = (kickoff, home_id, away_id)
                if key in seen:
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=match.match_id,
                            reason="duplicate_across_providers",
                            detail=f"({kickoff_iso}, {home_id}, {away_id})",
                            kickoff_at=kickoff_iso,
                            home_team_id=home_id,
                            away_team_id=away_id,
                            stage=stage,
                            tournament_name=tournament_name,
                        )
                    )
                    continue

                result = results.get(match.match_id)
                if result is None:
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=match.match_id,
                            reason="missing_result",
                            detail="provider has no result for this match",
                            kickoff_at=kickoff_iso,
                            home_team_id=home_id,
                            away_team_id=away_id,
                            stage=stage,
                            tournament_name=tournament_name,
                        )
                    )
                    continue

                if result.home_goals_90 is None or result.away_goals_90 is None:
                    manifest.quarantined.append(
                        QuarantineReason(
                            raw_match_id=match.match_id,
                            reason="missing_scores",
                            detail="home_goals_90/away_goals_90 are null",
                            kickoff_at=kickoff_iso,
                            home_team_id=home_id,
                            away_team_id=away_id,
                            stage=stage,
                            tournament_name=tournament_name,
                        )
                    )
                    continue

                if result.home_goals_90 == result.away_goals_90:
                    if getattr(result, "home_advances", None) is None:
                        manifest.quarantined.append(
                            QuarantineReason(
                                raw_match_id=match.match_id,
                                reason="no_advancer_on_draw",
                                detail="knockout match drawn 90 minutes with no advancer",
                                kickoff_at=kickoff_iso,
                                home_team_id=home_id,
                                away_team_id=away_id,
                                stage=stage,
                                tournament_name=tournament_name,
                            )
                        )
                        continue

                row = KnockoutRow(
                    match_id=_synth_match_id(match.match_id, tournament_name, kickoff, home_id, away_id),
                    kickoff_at=kickoff,
                    competition_id=match.competition_id,
                    competition_name=tournament_name,
                    stage=stage,
                    season_or_year=match.season_or_year,
                    home_team_id=home_id,
                    away_team_id=away_id,
                    home_goals_90=result.home_goals_90,
                    away_goals_90=result.away_goals_90,
                    # The provider schema is ``home_advances`` (raw boolean);
                    # the manifest label is ``home_wins_tie`` (the same value,
                    # semantically "home wins the tie and advances").
                    home_wins_tie=bool(result.home_advances),
                    source=provider_name,
                )
                manifest.rows.append(row)
                seen.add(key)
                tournament_counts[tournament_name] += 1

        manifest.tournament_coverage = dict(tournament_counts)
        manifest.expected_vs_found = _expected_vs_found(
            tournament_counts=tournament_counts,
            expected_per_tournament=self._expected_per_tournament
            or {n: self.DEFAULT_EXPECTED_PER_TOURNAMENT for n in tournament_counts},
        )
        logger.info(
            "Built knockout manifest",
            extra={
                "rows": len(manifest.rows),
                "quarantined": len(manifest.quarantined),
                "excluded_third_place": len(manifest.excluded_third_place),
                "tournaments": dict(tournament_counts),
            },
        )
        return manifest


def _expected_vs_found(
    tournament_counts: dict[str, int],
    expected_per_tournament: dict[str, int],
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for tournament, found in sorted(tournament_counts.items()):
        expected = expected_per_tournament.get(tournament)
        out[tournament] = {
            "expected": expected,
            "found": found,
            "passes": (expected is None) or (found == expected),
            "delta": (None if expected is None else found - expected),
        }
    for tournament, expected in expected_per_tournament.items():
        if tournament not in tournament_counts:
            out[tournament] = {
                "expected": expected,
                "found": 0,
                "passes": False,
                "delta": -expected,
            }
    return out


def _synth_match_id(
    raw: str, tournament: str, kickoff: datetime, home: str, away: str
) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", tournament.lower()).strip("_")
    date = kickoff.strftime("%Y%m%d")
    return f"{slug}_{date}_{home}_{away}"