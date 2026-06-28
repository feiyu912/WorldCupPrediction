"""Ingestion service: writes provider records into the application database.

The ingestion layer is deliberately idempotent: re-ingesting the same
data should not create duplicate snapshots because the
``MarketOddsSnapshot`` and ``PlayerAvailabilitySnapshot`` tables have
``raw_payload_hash`` plus timestamp uniqueness. The service preserves
the original raw data unchanged in the source files; the database only
contains normalized records with explicit lineage columns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.db.models import (
    Competition,
    MarketOddsSnapshot,
    Match,
    PlayerAvailabilitySnapshot,
    Team,
)
from football_advance_predictor.schemas.availability import AvailabilityIn
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn
from football_advance_predictor.schemas.odds import MarketOddsIn

logger = get_logger(__name__)


class IngestionService:
    """High-level ingestion of matches, odds, and availability."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Teams / competitions
    # ------------------------------------------------------------------

    def upsert_teams(self, teams: list[dict[str, Any]]) -> int:
        count = 0
        for team in teams:
            team_id = team["team_id"]
            existing = self.session.get(Team, team_id)
            if existing is None:
                self.session.add(
                    Team(
                        team_id=team_id,
                        canonical_name=team.get("canonical_name", team_id),
                        fifa_code=team.get("fifa_code"),
                        confederation=team.get("confederation"),
                        active_from=team.get("active_from"),
                        active_to=team.get("active_to"),
                        aliases=team.get("aliases", []),
                    )
                )
                count += 1
        self.session.flush()
        return count

    def upsert_competition(
        self,
        competition_id: str,
        canonical_name: str,
        competition_type: str = "international",
        importance_weight: float = 1.0,
        is_knockout_capable: bool = True,
    ) -> Competition:
        comp = self.session.get(Competition, competition_id)
        if comp is None:
            comp = Competition(
                competition_id=competition_id,
                canonical_name=canonical_name,
                competition_type=competition_type,
                importance_weight=importance_weight,
                is_knockout_capable=is_knockout_capable,
            )
            self.session.add(comp)
            self.session.flush()
        return comp

    # ------------------------------------------------------------------
    # Matches
    # ------------------------------------------------------------------

    def upsert_matches(self, matches: list[MatchIn]) -> int:
        count = 0
        for match in matches:
            existing = self.session.get(Match, match.match_id)
            if existing is None:
                self.session.add(self._match_in_to_model(match))
                count += 1
            else:
                # Update mutable fields only; never overwrite the immutable kickoff_at.
                existing.competition_id = match.competition_id
                existing.stage = match.stage
                existing.season_or_year = match.season_or_year
                existing.home_goals = match.home_goals
                existing.away_goals = match.away_goals
                existing.winner_team_id = match.winner_team_id
                existing.advancing_team_id = match.advancing_team_id
                existing.neutral_venue = match.neutral_venue
                existing.venue_name = match.venue_name
                existing.city = match.city
                existing.country = match.country
                existing.source = match.source
        self.session.flush()
        return count

    def upsert_result(self, result: MatchResultIn) -> None:
        from football_advance_predictor.db.models import MatchResult

        existing = self.session.get(MatchResult, result.match_id)
        if existing is None:
            self.session.add(
                MatchResult(
                    match_id=result.match_id,
                    final_status=result.final_status,
                    home_goals_90=result.home_goals_90,
                    away_goals_90=result.away_goals_90,
                    home_goals_et=result.home_goals_et,
                    away_goals_et=result.away_goals_et,
                    penalties_home=result.penalties_home,
                    penalties_away=result.penalties_away,
                    home_advances=result.home_advances,
                    result_verified_at=result.result_verified_at,
                )
            )
        else:
            existing.final_status = result.final_status
            existing.home_goals_90 = result.home_goals_90
            existing.away_goals_90 = result.away_goals_90
            existing.home_goals_et = result.home_goals_et
            existing.away_goals_et = result.away_goals_et
            existing.penalties_home = result.penalties_home
            existing.penalties_away = result.penalties_away
            existing.home_advances = result.home_advances
            existing.result_verified_at = result.result_verified_at
        self.session.flush()

    # ------------------------------------------------------------------
    # Odds
    # ------------------------------------------------------------------

    def upsert_odds(self, odds: list[MarketOddsIn]) -> int:
        if not odds:
            return 0
        inserted = 0
        for o in odds:
            # Deduplicate by raw_payload_hash; on conflict, do nothing.
            stmt = (
                pg_insert(MarketOddsSnapshot)
                .values(
                    odds_snapshot_id=o.odds_snapshot_id or _generate_id("odds"),
                    match_id=o.match_id,
                    bookmaker=o.bookmaker,
                    market_type=o.market_type,
                    selection=o.selection,
                    decimal_odds=o.decimal_odds,
                    captured_at=o.captured_at,
                    source=o.source,
                    is_live=o.is_live,
                    currency_or_region=o.currency_or_region,
                    source_name=o.source_name,
                    source_record_id=o.source_record_id,
                    source_url=o.source_url,
                    published_at=o.published_at,
                    ingested_at=o.ingested_at,
                    effective_at=o.effective_at,
                    raw_payload_hash=o.raw_payload_hash,
                    source_version=o.source_version,
                )
                .on_conflict_do_nothing(index_elements=["raw_payload_hash"])
            )
            result = self.session.execute(stmt)
            if result.rowcount:
                inserted += 1
        self.session.flush()
        return inserted

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def upsert_availability(self, records: list[AvailabilityIn]) -> int:
        if not records:
            return 0
        inserted = 0
        for record in records:
            stmt = (
                pg_insert(PlayerAvailabilitySnapshot)
                .values(
                    availability_id=record.availability_id or _generate_id("avail"),
                    match_id=record.match_id,
                    team_id=record.team_id,
                    player_id=record.player_id,
                    role=record.role,
                    availability_status=record.availability_status,
                    confidence=record.confidence,
                    published_at=record.published_at,
                    cutoff_eligible=record.cutoff_eligible,
                    source=record.source,
                    raw_text=record.raw_text,
                    source_name=record.source_name,
                    source_record_id=record.source_record_id,
                    source_url=record.source_url,
                    observed_at=record.observed_at,
                    ingested_at=record.ingested_at,
                    effective_at=record.effective_at,
                    raw_payload_hash=record.raw_payload_hash,
                    source_version=record.source_version,
                )
                .on_conflict_do_nothing(index_elements=["raw_payload_hash"])
            )
            result = self.session.execute(stmt)
            if result.rowcount:
                inserted += 1
        self.session.flush()
        return inserted

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_in_to_model(match: MatchIn) -> Match:
        return Match(
            match_id=match.match_id,
            kickoff_at=match.kickoff_at,
            competition_id=match.competition_id,
            stage=match.stage,
            season_or_year=match.season_or_year,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            home_goals=match.home_goals,
            away_goals=match.away_goals,
            winner_team_id=match.winner_team_id,
            advancing_team_id=match.advancing_team_id,
            neutral_venue=match.neutral_venue,
            venue_name=match.venue_name,
            city=match.city,
            country=match.country,
            source=match.source,
        )

    def get_odds_for_match_before(self, match_id: str, cutoff: datetime) -> list[MarketOddsSnapshot]:
        cutoff_utc = cutoff.astimezone(UTC) if cutoff.tzinfo else cutoff.replace(tzinfo=UTC)
        stmt = (
            select(MarketOddsSnapshot)
            .where(MarketOddsSnapshot.match_id == match_id)
            .where(MarketOddsSnapshot.captured_at <= cutoff_utc)
            .order_by(MarketOddsSnapshot.captured_at)
        )
        return list(self.session.scalars(stmt))

    def get_availability_for_match_before(
        self, match_id: str, cutoff: datetime
    ) -> list[PlayerAvailabilitySnapshot]:
        cutoff_utc = cutoff.astimezone(UTC) if cutoff.tzinfo else cutoff.replace(tzinfo=UTC)
        stmt = (
            select(PlayerAvailabilitySnapshot)
            .where(PlayerAvailabilitySnapshot.match_id == match_id)
            .where(PlayerAvailabilitySnapshot.published_at <= cutoff_utc)
        )
        return list(self.session.scalars(stmt))


def _generate_id(prefix: str) -> str:
    import uuid

    return f"{prefix}_{uuid.uuid4().hex[:16]}"
