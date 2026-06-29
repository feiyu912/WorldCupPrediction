"""Tests for the v1 pre-registered feature set."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from football_advance_predictor.features.v1_features import (
    V1FeatureRow,
    compute_v1_features,
)


def _dt(year, month, day):
    return datetime(year, month, day, tzinfo=timezone.utc)


def test_compute_v1_features_basic() -> None:
    kickoff = _dt(2026, 7, 1)
    cutoff = kickoff - timedelta(hours=24)
    home_elo, away_elo = 1600.0, 1500.0
    home_recent = [
        (kickoff - timedelta(days=30), "brazil", 3),
        (kickoff - timedelta(days=60), "argentina", 0),
        (kickoff - timedelta(days=90), "germany", 1),
    ]
    away_recent = [
        (kickoff - timedelta(days=30), "spain", 0),
        (kickoff - timedelta(days=60), "france", 3),
    ]
    home_gd = [(kickoff - timedelta(days=30), 1), (kickoff - timedelta(days=60), -1)]
    away_gd = [(kickoff - timedelta(days=30), -2), (kickoff - timedelta(days=60), 0)]
    row = compute_v1_features(
        home_team_id="bra",
        away_team_id="arg",
        kickoff_at=kickoff,
        stage_canonical="quarter_final",
        cutoff=cutoff,
        home_elo_at_cutoff=home_elo,
        away_elo_at_cutoff=away_elo,
        home_recent_results=home_recent,
        away_recent_results=away_recent,
        home_recent_goal_diff=home_gd,
        away_recent_goal_diff=away_gd,
        home_last_match_at=cutoff - timedelta(days=3),
        away_last_match_at=cutoff - timedelta(days=5),
    )
    assert row.elo_difference == pytest.approx(100.0, abs=1e-9)
    assert row.elo_home_win_prob == pytest.approx(1 / (1 + 10 ** (-100 / 400)), abs=1e-9)
    # Elo sign: home team 100 points above away → probability > 0.5.
    assert 0.5 < row.elo_home_win_prob < 1.0
    # Rest-day diff = 4-6 = -2 (away team has more rest).
    assert row.rest_days_home == pytest.approx(4.0)
    assert row.rest_days_away == pytest.approx(6.0)
    assert row.rest_days_difference == pytest.approx(-2.0)
    # Stage indicators: quarter_final.
    assert row.is_round_of_16 == 0
    assert row.is_quarter_final == 1
    assert row.is_semi_final == 0
    assert row.is_final == 0
    # Feature dict has all keys.
    fd = row.feature_dict()
    assert set(fd.keys()) == {
        "elo_difference", "elo_home_win_prob",
        "form_home", "form_away", "form_difference",
        "goal_diff_home", "goal_diff_away", "goal_diff_difference",
        "rest_days_home", "rest_days_away", "rest_days_difference",
        "is_round_of_16", "is_quarter_final", "is_semi_final", "is_final",
    }


def test_compute_v1_features_anti_leakage() -> None:
    """cutoff must be strictly before kickoff_at; otherwise raise."""
    kickoff = _dt(2026, 7, 1)
    with pytest.raises(ValueError):
        compute_v1_features(
            home_team_id="a", away_team_id="b",
            kickoff_at=kickoff, stage_canonical="final",
            cutoff=kickoff,  # not before kickoff
            home_elo_at_cutoff=1500.0, away_elo_at_cutoff=1500.0,
            home_recent_results=[], away_recent_results=[],
            home_recent_goal_diff=[], away_recent_goal_diff=[],
            home_last_match_at=None, away_last_match_at=None,
        )


def test_compute_v1_features_default_rest_days() -> None:
    """When last_match_at is None, default to 7 days."""
    kickoff = _dt(2026, 7, 1)
    cutoff = kickoff - timedelta(hours=24)
    row = compute_v1_features(
        home_team_id="a", away_team_id="b",
        kickoff_at=kickoff, stage_canonical="round_of_16",
        cutoff=cutoff,
        home_elo_at_cutoff=1500.0, away_elo_at_cutoff=1500.0,
        home_recent_results=[], away_recent_results=[],
        home_recent_goal_diff=[], away_recent_goal_diff=[],
        home_last_match_at=None, away_last_match_at=None,
    )
    assert row.rest_days_home == 7.0
    assert row.rest_days_away == 7.0
    assert row.rest_days_difference == 0.0


def test_compute_v1_features_complementarity() -> None:
    """Mirroring (A, B) → (B, A) flips the sign of the differences
    but keeps the form / goal-diff / rest-day structures intact.

    The probability orientation is preserved (home_team_id always
    holds the home side). Mirrored training examples are the caller's
    responsibility; this test verifies the feature set is consistent
    with the manifest's home_wins_tie orientation.
    """
    kickoff = _dt(2026, 7, 1)
    cutoff = kickoff - timedelta(hours=24)
    home_recent = [(cutoff - timedelta(days=10), "brazil", 3)]
    away_recent = [(cutoff - timedelta(days=10), "spain", 0)]
    home_gd = [(cutoff - timedelta(days=10), 2)]
    away_gd = [(cutoff - timedelta(days=10), -1)]
    row_a = compute_v1_features(
        home_team_id="a", away_team_id="b",
        kickoff_at=kickoff, stage_canonical="quarter_final",
        cutoff=cutoff,
        home_elo_at_cutoff=1600.0, away_elo_at_cutoff=1500.0,
        home_recent_results=home_recent, away_recent_results=away_recent,
        home_recent_goal_diff=home_gd, away_recent_goal_diff=away_gd,
        home_last_match_at=cutoff - timedelta(days=3),
        away_last_match_at=cutoff - timedelta(days=4),
    )
    row_b = compute_v1_features(
        home_team_id="b", away_team_id="a",
        kickoff_at=kickoff, stage_canonical="quarter_final",
        cutoff=cutoff,
        home_elo_at_cutoff=1500.0, away_elo_at_cutoff=1600.0,
        home_recent_results=away_recent, away_recent_results=home_recent,
        home_recent_goal_diff=away_gd, away_recent_goal_diff=home_gd,
        home_last_match_at=cutoff - timedelta(days=4),
        away_last_match_at=cutoff - timedelta(days=3),
    )
    # Elo difference flips sign, probability orientation stays.
    assert row_a.elo_difference == pytest.approx(100.0, abs=1e-9)
    assert row_b.elo_difference == pytest.approx(-100.0, abs=1e-9)
    # Each side's elo_home_win_prob reflects its own rating.
    p_a = row_a.elo_home_win_prob
    p_b = row_b.elo_home_win_prob
    assert 0.5 < p_a < 1.0
    assert 0.0 < p_b < 0.5
    # Rest-day diff flips sign.
    assert row_a.rest_days_difference == pytest.approx(-1.0)
    assert row_b.rest_days_difference == pytest.approx(1.0)


def test_v1_features_have_15_columns() -> None:
    """Pin the v1 feature schema to 15 features.

    Any new feature added later MUST be a separate, post-v1 column
    so that pre-registered training tables can be reproduced.
    """
    kickoff = _dt(2026, 7, 1)
    cutoff = kickoff - timedelta(hours=24)
    row = compute_v1_features(
        home_team_id="a", away_team_id="b",
        kickoff_at=kickoff, stage_canonical="final",
        cutoff=cutoff,
        home_elo_at_cutoff=1500.0, away_elo_at_cutoff=1500.0,
        home_recent_results=[], away_recent_results=[],
        home_recent_goal_diff=[], away_recent_goal_diff=[],
        home_last_match_at=None, away_last_match_at=None,
    )
    assert len(row.feature_dict()) == 15
