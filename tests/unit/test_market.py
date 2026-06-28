"""Tests for market consensus and de-vigging."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from football_advance_predictor.features.market.consensus import (
    MarketAdvanceProbabilityModel,
    de_vig_two_way,
    implied_probability,
)


def test_implied_probability() -> None:
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(4.0) == pytest.approx(0.25)


def test_invalid_odds_rejected() -> None:
    with pytest.raises(ValueError):
        implied_probability(0.5)


def test_de_vig_two_way() -> None:
    p_h, p_a = de_vig_two_way(2.0, 2.0)
    assert p_h == pytest.approx(0.5)
    assert p_a == pytest.approx(0.5)
    p_h, p_a = de_vig_two_way(1.5, 3.0)
    assert p_h + p_a == pytest.approx(1.0)
    assert p_h > p_a


class _StubSnapshot:
    def __init__(self, market, selection, odds, captured_at, bookmaker="B"):
        self.market_type = market
        self.selection = selection
        self.decimal_odds = odds
        self.captured_at = captured_at
        self.bookmaker = bookmaker


def test_consensus_uses_only_snapshots_before_cutoff() -> None:
    snapshots = [
        _StubSnapshot("home_to_advance", "home", 2.0, datetime(2024, 1, 1, tzinfo=UTC)),
        _StubSnapshot("home_to_advance", "away", 2.0, datetime(2024, 1, 1, tzinfo=UTC)),
        _StubSnapshot("home_to_advance", "home", 1.5, datetime(2030, 1, 1, tzinfo=UTC)),
        _StubSnapshot("home_to_advance", "away", 3.0, datetime(2030, 1, 1, tzinfo=UTC)),
    ]
    model = MarketAdvanceProbabilityModel(snapshots, min_bookmakers=1)
    consensus = model.consensus_at(datetime(2025, 1, 1, tzinfo=UTC))
    assert consensus is not None
    assert consensus.home_advance_probability == pytest.approx(0.5)


def test_consensus_returns_none_without_snapshots() -> None:
    model = MarketAdvanceProbabilityModel([], min_bookmakers=1)
    assert model.consensus_at(datetime(2025, 1, 1, tzinfo=UTC)) is None


def test_consensus_min_bookmaker_filter() -> None:
    snapshots = [
        _StubSnapshot("home_to_advance", "home", 2.0, datetime(2024, 1, 1, tzinfo=UTC), bookmaker="A"),
        _StubSnapshot("home_to_advance", "away", 2.0, datetime(2024, 1, 1, tzinfo=UTC), bookmaker="A"),
    ]
    model = MarketAdvanceProbabilityModel(snapshots, min_bookmakers=2)
    assert model.consensus_at(datetime(2025, 1, 1, tzinfo=UTC)) is None
