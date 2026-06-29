"""Tests for the OpenFootball tournament provider."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from football_advance_predictor.data.sources.openfootball import OpenFootballTournamentProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def openfootball_json(tmp_path: Path) -> Path:
    """A minimal Open Football tournament: 1 group + 1 knockout round.

    The group match has a tied score with type=draw (must be skipped from
    results). The knockout match has a clear winner via the 'winner' field.
    """
    payload = {
        "name": "Test Tournament",
        "rounds": [
            {
                "name": "Group A",
                "matches": [
                    {
                        "key": "g1",
                        "date": "2022-11-20",
                        "team1": {"name": "Brazil"},
                        "team2": {"name": "Argentina"},
                        "score1": 1,
                        "score2": 1,
                        "type": "draw",
                    },
                    {
                        # Another group match with a real winner (not type=draw).
                        "key": "g2",
                        "date": "2022-11-21",
                        "team1": {"name": "Brazil"},
                        "team2": {"name": "Germany"},
                        "score1": 2,
                        "score2": 0,
                    },
                ],
            },
            {
                "name": "Quarter-final",
                "matches": [
                    {
                        "key": "qf1",
                        "date": "2022-12-10",
                        "team1": {"name": "Brazil"},
                        "team2": {"name": "Argentina"},
                        "score1": 2,
                        "score2": 1,
                        "score1et": 0,
                        "score2et": 0,
                        "score1pen": 0,
                        "score2pen": 0,
                        "winner": 1,
                    },
                    {
                        # 90-minute draw, no `winner` field, type != draw:
                        # result builder returns None for safety.
                        "key": "qf2",
                        "date": "2022-12-11",
                        "team1": {"name": "Germany"},
                        "team2": {"name": "France"},
                        "score1": 1,
                        "score2": 1,
                        "type": "knockout",
                    },
                ],
            },
        ],
    }
    path = tmp_path / "world-cup.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reads_single_round_returns_matches(openfootball_json: Path) -> None:
    provider = OpenFootballTournamentProvider(openfootball_json)
    matches = provider.fetch_matches()
    # All 4 matches across both rounds should be returned.
    assert len(matches) == 4
    stages = sorted({m.stage for m in matches})
    assert "Group A" in stages
    assert "Quarter-final" in stages
    # Match IDs are deterministic and prefix with OF_.
    assert all(m.match_id.startswith("OF_") for m in matches)


def test_winner_field_sets_home_advances(openfootball_json: Path) -> None:
    provider = OpenFootballTournamentProvider(openfootball_json)
    results = provider.fetch_results()
    # Match IDs are derived from the JSON `key` field -> OF_g1, OF_g2, OF_qf1, OF_qf2.
    by_id = {r.match_id: r for r in results}
    qf1 = by_id["OF_qf1"]
    # The qf1 match has explicit winner=1 -> home (team1) advances.
    assert qf1.home_advances is True
    assert qf1.home_goals_90 == 2
    assert qf1.away_goals_90 == 1


def test_skips_draw_type_rows(openfootball_json: Path) -> None:
    provider = OpenFootballTournamentProvider(openfootball_json)
    # fetch_results skips group draws (type == "draw") and drawn knockouts
    # without a winner. Both g1 and qf2 should be absent.
    results = provider.fetch_results()
    match_ids = {r.match_id for r in results}

    # Group A draw (g1) is filtered out.
    assert "OF_g1" not in match_ids
    # The drawn knockout with no winner (qf2) is filtered out.
    assert "OF_qf2" not in match_ids
    # Group match g2 with a real winner AND the qf1 knockout with explicit
    # winner both surface.
    assert "OF_g2" in match_ids
    assert "OF_qf1" in match_ids


def test_match_to_in_skips_invalid_rows(tmp_path: Path) -> None:
    """An unparseable match (no date) is logged and skipped, not raised."""
    bad_path = tmp_path / "broken.json"
    bad_path.write_text(
        json.dumps(
            {
                "name": "Broken",
                "rounds": [
                    {
                        "name": "Group",
                        "matches": [
                            # Missing date -> ValueError -> skipped via except.
                            {"team1": {"name": "A"}, "team2": {"name": "B"}, "score1": 1, "score2": 0},
                            {"date": "2022-01-01", "team1": {"name": "A"}, "team2": {"name": "B"}, "score1": 0, "score2": 1},
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    provider = OpenFootballTournamentProvider(bad_path)
    matches = provider.fetch_matches()
    # Only the well-formed match survives.
    assert len(matches) == 1
