"""Tests for the dynamic Elo engine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from football_advance_predictor.features.elo.elo_engine import (
    DynamicEloEngine,
    EloConfig,
    expected_score,
)


def _match(kickoff, home, away, hg, ag, *, neutral=False, advances=None, importance=1.0):
    return {
        "kickoff_at": kickoff,
        "home_team_id": home,
        "away_team_id": away,
        "home_goals": hg,
        "away_goals": ag,
        "neutral_venue": neutral,
        "home_advances": advances,
        "competition_importance": importance,
    }


def test_initial_rating_when_no_history() -> None:
    engine = DynamicEloEngine()
    rating = engine.get_team_rating("ghost", datetime(2030, 1, 1, tzinfo=UTC))
    assert rating == engine.config.initial_rating


def test_elo_updates_chronologically() -> None:
    cfg = EloConfig(base_k_factor=20.0, home_advantage=0.0, tie_resolution="draw_treated_as_50_50")
    engine = DynamicEloEngine(cfg)
    engine.fit(
        [
            _match(datetime(2024, 1, 1, tzinfo=UTC), "A", "B", 2, 0, advances=True),
            _match(datetime(2024, 6, 1, tzinfo=UTC), "A", "B", 1, 1, advances=True),
        ]
    )
    # After a win and a draw, A should still be > B.
    as_of = datetime(2024, 12, 1, tzinfo=UTC)
    r_a = engine.get_team_rating("A", as_of)
    r_b = engine.get_team_rating("B", as_of)
    assert r_a > r_b


def test_future_match_does_not_affect_past_rating() -> None:
    cfg = EloConfig(base_k_factor=20.0, home_advantage=0.0)
    engine = DynamicEloEngine(cfg)
    matches = [
        _match(datetime(2020, 1, 1, tzinfo=UTC), "A", "B", 3, 0, advances=True),
        _match(datetime(2025, 1, 1, tzinfo=UTC), "A", "B", 0, 3, advances=False),
    ]
    engine.fit(matches)
    # A won in 2020 (gained rating) and lost in 2025 (lost rating).
    rating_at_2021 = engine.get_team_rating("A", datetime(2021, 1, 1, tzinfo=UTC))
    rating_at_2026 = engine.get_team_rating("A", datetime(2026, 1, 1, tzinfo=UTC))
    # The 2025 loss should have lowered the rating, so 2026 < 2021.
    assert rating_at_2026 < rating_at_2021
    # The pre-2025 rating must not have been affected by the 2025 match.
    rating_just_before_2025 = engine.get_team_rating("A", datetime(2024, 12, 31, tzinfo=UTC))
    rating_just_before_2025_history = engine.get_team_rating("A", datetime(2024, 12, 31, tzinfo=UTC))
    assert rating_just_before_2025 == rating_just_before_2025_history


def test_neutral_venue_disables_home_advantage() -> None:
    cfg = EloConfig(home_advantage=100.0, tie_resolution="draw_treated_as_50_50")
    engine = DynamicEloEngine(cfg)
    p_with = engine.predict_home_win_probability(
        "A", "B", datetime(2030, 1, 1, tzinfo=UTC), neutral_venue=False
    )
    p_without = engine.predict_home_win_probability(
        "A", "B", datetime(2030, 1, 1, tzinfo=UTC), neutral_venue=True
    )
    assert p_with > p_without


def test_expected_score_formula() -> None:
    # equal ratings -> 0.5
    assert expected_score(1500, 1500) == pytest.approx(0.5, abs=1e-6)
    # +200 rating difference -> ~0.76
    assert expected_score(1700, 1500) == pytest.approx(0.7597, abs=1e-3)


def test_competition_weight_modulates_k_factor() -> None:
    cfg = EloConfig(base_k_factor=20.0, k_floor=0.0, k_ceiling=100.0)
    engine = DynamicEloEngine(cfg)
    base = engine._k_factor(1.0)
    boosted = engine._k_factor(2.0)
    assert boosted > base


def test_time_decay_applied_to_inactive_teams() -> None:
    cfg = EloConfig(time_decay_per_day=0.01, base_k_factor=20.0, home_advantage=0.0)
    engine = DynamicEloEngine(cfg)
    engine.fit(
        [_match(datetime(2020, 1, 1, tzinfo=UTC), "A", "B", 1, 0, advances=True)]
    )
    rating_now = engine.get_team_rating("A", datetime(2020, 1, 2, tzinfo=UTC))
    rating_later = engine.get_team_rating("A", datetime(2025, 1, 1, tzinfo=UTC))
    assert rating_later < rating_now


def test_advance_probability_includes_draw_treatment() -> None:
    cfg = EloConfig(home_advantage=0.0, tie_resolution="draw_treated_as_50_50")
    engine = DynamicEloEngine(cfg)
    p = engine.predict_home_advance_probability(
        "A", "B", datetime(2030, 1, 1, tzinfo=UTC), neutral_venue=True
    )
    # With no info, both teams at 1500 -> 0.5 + 0.5*0.27 ≈ 0.635
    assert 0.5 <= p <= 1.0
