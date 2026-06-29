"""Tests for the martj42 results provider."""

from __future__ import annotations

from pathlib import Path

import pytest
from football_advance_predictor.data.sources.martj42 import MartJ42ResultsProvider

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def results_csv(tmp_path: Path) -> Path:
    path = tmp_path / "results.csv"
    path.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2022-12-10,Brazil,Argentina,2,0,Friendly,Rio,Brazil,FALSE\n"
        "2022-12-13,Germany,France,1,1,Friendly,Berlin,Germany,FALSE\n"
        "2022-12-18,Atlantis,Hyperborea,3,0,Friendly,Atlantis,Atlantis,FALSE\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def shootouts_csv(tmp_path: Path) -> Path:
    path = tmp_path / "shootouts.csv"
    path.write_text(
        "date,home_team,away_team,winner\n"
        "2022-12-13,Germany,France,Germany\n",
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reads_three_matches(results_csv: Path, tmp_path: Path) -> None:
    shootouts = tmp_path / "shootouts.csv"
    shootouts.write_text("date,home_team,away_team,winner\n", encoding="utf-8")
    provider = MartJ42ResultsProvider(results_csv, shootouts)
    matches = provider.fetch_matches()
    assert len(matches) == 3
    ids = [m.match_id for m in matches]
    assert ids[0].startswith("MJ42_")
    assert any("brazil" in i for i in ids)
    assert any("argentina" in i for i in ids)


def test_draw_with_shootout_sets_home_advances(results_csv: Path, shootouts_csv: Path) -> None:
    provider = MartJ42ResultsProvider(results_csv, shootouts_csv)
    matches = provider.fetch_matches()
    results = provider.fetch_results()
    # The Germany-France row is a 1-1 draw with a shootout where Germany won.
    draw_match = next(m for m in matches if "germany" in m.home_team_id)
    draw_result = next(r for r in results if r.match_id == draw_match.match_id)
    # Winner == home team -> home_advances is True.
    assert draw_match.home_goals == 1
    assert draw_match.away_goals == 1
    assert draw_result.home_advances is True


def test_unknown_team_recorded_in_unresolved(results_csv: Path, shootouts_csv: Path) -> None:
    provider = MartJ42ResultsProvider(results_csv, shootouts_csv)
    # Trigger resolution so the unresolved queue is populated.
    provider.fetch_matches()
    names = provider._registry.unresolved_names()
    # Atlantis and Hyperborea are not in built-in defaults and resolve to slugs,
    # so they should be recorded in the unresolved queue.
    assert "Atlantis" in names
    assert "Hyperborea" in names
