"""Schema validation tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from football_advance_predictor.schemas.availability import AvailabilityIn
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn
from football_advance_predictor.schemas.odds import MarketOddsIn
from football_advance_predictor.schemas.predictions import (
    ConfidenceBand,
    assign_confidence_band,
)
from pydantic import ValidationError


def test_match_in_minimal_required_fields() -> None:
    m = MatchIn(
        match_id="M1",
        kickoff_at=datetime(2026, 7, 1, tzinfo=UTC),
        competition_id="WC",
        season_or_year="2026",
        home_team_id="FRA",
        away_team_id="SWE",
    )
    assert m.neutral_venue is False
    assert m.source == "local"


def test_match_in_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        MatchIn(  # type: ignore[call-arg]
            match_id="M1",
            kickoff_at=datetime(2026, 7, 1, tzinfo=UTC),
            competition_id="WC",
            season_or_year="2026",
            home_team_id="FRA",
            away_team_id="SWE",
            unknown_field="oops",
        )


def test_market_odds_in_rejects_below_one_odds() -> None:
    with pytest.raises(ValidationError):
        MarketOddsIn(
            match_id="M1",
            bookmaker="B",
            market_type="home_to_advance",
            selection="home",
            decimal_odds=0.5,
            captured_at=datetime(2024, 1, 1, tzinfo=UTC),
            ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
            effective_at=datetime(2024, 1, 1, tzinfo=UTC),
            raw_payload_hash="h",
        )


def test_market_odds_in_accepts_valid_odds() -> None:
    o = MarketOddsIn(
        match_id="M1",
        bookmaker="B",
        market_type="home_to_advance",
        selection="home",
        decimal_odds=1.85,
        captured_at=datetime(2024, 1, 1, tzinfo=UTC),
        ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
        effective_at=datetime(2024, 1, 1, tzinfo=UTC),
        raw_payload_hash="h1",
    )
    assert o.decimal_odds == 1.85


def test_availability_in_requires_status() -> None:
    with pytest.raises(ValidationError):
        AvailabilityIn(
            match_id="M1",
            team_id="T1",
            role="defender",
            published_at=datetime(2024, 1, 1, tzinfo=UTC),
            observed_at=datetime(2024, 1, 1, tzinfo=UTC),
            ingested_at=datetime(2024, 1, 1, tzinfo=UTC),
            effective_at=datetime(2024, 1, 1, tzinfo=UTC),
            raw_payload_hash="h1",
        )


def test_match_result_requires_home_advances() -> None:
    with pytest.raises(ValidationError):
        MatchResultIn(
            match_id="M1",
            home_goals_90=1,
            away_goals_90=0,
            result_verified_at=datetime(2024, 1, 1, tzinfo=UTC),
        )  # type: ignore[call-arg]


def test_assign_confidence_band_thresholds() -> None:
    assert assign_confidence_band(0.65) == ConfidenceBand.CLEAR_LEAN
    assert assign_confidence_band(0.35) == ConfidenceBand.CLEAR_LEAN
    assert assign_confidence_band(0.58) == ConfidenceBand.SLIGHT_LEAN
    assert assign_confidence_band(0.42) == ConfidenceBand.SLIGHT_LEAN
    assert assign_confidence_band(0.5) == ConfidenceBand.NEAR_COIN_FLIP
    assert assign_confidence_band(0.51) == ConfidenceBand.NEAR_COIN_FLIP


def test_assign_confidence_band_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        assign_confidence_band(-0.1)
    with pytest.raises(ValueError):
        assign_confidence_band(1.1)


def test_assign_confidence_band_custom_thresholds() -> None:
    # With stricter thresholds, even 0.7 is a slight lean.
    assert (
        assign_confidence_band(0.7, clear_lean_min=0.8)
        == ConfidenceBand.SLIGHT_LEAN
    )
