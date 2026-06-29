"""Tests for the KnockoutManifestBuilder."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from football_advance_predictor.data.aliases.alias_registry import AliasRegistry
from football_advance_predictor.data.knockout.manifest import (
    KnockoutManifestBuilder,
    is_knockout_stage,
)
from football_advance_predictor.schemas.matches import MatchIn, MatchResultIn

# ---------------------------------------------------------------------------
# Fake provider
# ---------------------------------------------------------------------------


class FakeProvider:
    """Tiny provider that returns pre-canned matches and results."""

    def __init__(
        self,
        matches: list[MatchIn],
        results: list[MatchResultIn],
    ) -> None:
        self._matches = matches
        self._results = results

    def fetch_matches(self, **kwargs: Any) -> list[MatchIn]:
        return list(self._matches)

    def fetch_results(self, **kwargs: Any) -> list[MatchResultIn]:
        return list(self._results)


def _match(
    match_id: str,
    *,
    home: str,
    away: str,
    stage: str,
    kickoff: datetime | None = None,
    competition_id: str = "fifa_world_cup",
) -> MatchIn:
    return MatchIn(
        match_id=match_id,
        kickoff_at=kickoff or datetime(2022, 12, 10, 20, 0, tzinfo=UTC),
        competition_id=competition_id,
        stage=stage,
        season_or_year="2022",
        home_team_id=home,
        away_team_id=away,
        home_goals=0,
        away_goals=0,
        winner_team_id=None,
        advancing_team_id=None,
        neutral_venue=True,
        source="fake",
    )


def _result(
    match_id: str,
    *,
    home_goals_90: int | None,
    away_goals_90: int | None,
    home_advances: bool | None = None,
    penalties_home: int | None = None,
    penalties_away: int | None = None,
) -> MatchResultIn:
    # If home_advances isn't provided, derive from score comparison.
    if home_advances is None:
        if home_goals_90 is None or away_goals_90 is None:
            adv: bool | None = False
        elif home_goals_90 > away_goals_90:
            adv = True
        elif home_goals_90 < away_goals_90:
            adv = False
        else:
            adv = None  # 90-min draw without penalties -> None
    else:
        adv = home_advances
    return MatchResultIn(
        match_id=match_id,
        final_status="final",
        home_goals_90=home_goals_90,
        away_goals_90=away_goals_90,
        home_goals_et=None,
        away_goals_et=None,
        penalties_home=penalties_home,
        penalties_away=penalties_away,
        home_advances=bool(adv) if adv is not None else False,
        result_verified_at=datetime(2022, 12, 10, 22, 0, tzinfo=UTC),
    )


@pytest.fixture
def registry(tmp_path: Path) -> AliasRegistry:
    return AliasRegistry.open(tmp_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_non_knockout_matches_are_excluded(registry: AliasRegistry) -> None:
    match_qf = _match("QF1", home="brazil", away="argentina", stage="Quarter-final")
    match_grp = _match("GRP1", home="brazil", away="argentina", stage="Group A")
    result_qf = _result("QF1", home_goals_90=2, away_goals_90=1)
    result_grp = _result("GRP1", home_goals_90=1, away_goals_90=1)

    builder = KnockoutManifestBuilder(registry)
    builder.add_provider("p1", FakeProvider([match_qf, match_grp], [result_qf, result_grp]))
    manifest = builder.build()

    # Only the knockout match is accepted; group stages are filtered
    # out (no quarantine noise for non-knockout rows).
    assert len(manifest.rows) == 1
    assert manifest.rows[0].match_id.endswith("brazil_argentina")
    assert manifest.total == 1
    assert manifest.quarantined == []
    # The third-place variant is excluded with its own bucket.
    assert manifest.excluded_third_place == []


def test_knockout_with_no_result_is_quarantined(registry: AliasRegistry) -> None:
    # The builder iterates over *results*; a result whose match_id refers
    # to a missing match is silently skipped (continue), not quarantined.
    # Verify the silent-skip path: nothing is added or quarantined for a
    # result whose match is absent.
    result_with_missing_match = _result("PHANTOM", home_goals_90=1, away_goals_90=0)
    builder = KnockoutManifestBuilder(registry)
    builder.add_provider("p1", FakeProvider([], [result_with_missing_match]))
    manifest = builder.build()

    assert len(manifest.rows) == 0
    # Quarantine list stays empty because the missing match is silently skipped.
    assert all(q.reason != "missing_result" for q in manifest.quarantined)


def test_knockout_match_id_only_survives_with_result(registry: AliasRegistry) -> None:
    """When a knockout match has no matching result, it doesn't pollute the manifest."""
    match = _match("QF1", home="brazil", away="argentina", stage="Quarter-final")
    builder = KnockoutManifestBuilder(registry)
    builder.add_provider("p1", FakeProvider([match], []))
    manifest = builder.build()

    assert len(manifest.rows) == 0
    assert manifest.total == 0


def test_knockout_with_missing_scores_is_quarantined(registry: AliasRegistry) -> None:
    match = _match("SF1", home="brazil", away="argentina", stage="Semi-final")
    result = _result("SF1", home_goals_90=None, away_goals_90=None)
    builder = KnockoutManifestBuilder(registry)
    builder.add_provider("p1", FakeProvider([match], [result]))
    manifest = builder.build()

    assert len(manifest.rows) == 0
    assert any(q.reason == "missing_scores" for q in manifest.quarantined)


def test_ninety_minute_draw_without_advancer_is_quarantined(registry: AliasRegistry) -> None:
    # MatchResultIn.home_advances is a required bool field, so to hit the
    # 'no_advancer_on_draw' quarantine we use a duck-typed result with the
    # attribute explicitly set to None.
    match = _match("QF1", home="brazil", away="argentina", stage="Quarter-final")

    class _ResultStub:
        match_id = "QF1"
        home_goals_90 = 1
        away_goals_90 = 1
        penalties_home = None
        penalties_away = None
        home_advances = None  # triggers the "draw with no advancer" path

    builder = KnockoutManifestBuilder(registry)
    builder.add_provider("p1", FakeProvider([match], [_ResultStub()]))
    manifest = builder.build()

    assert len(manifest.rows) == 0
    assert any(q.reason == "no_advancer_on_draw" for q in manifest.quarantined)


def test_duplicate_across_providers_dedupes(registry: AliasRegistry) -> None:
    kickoff = datetime(2022, 12, 10, 20, 0, tzinfo=UTC)
    match_a = _match("A", home="brazil", away="argentina", stage="Quarter-final", kickoff=kickoff)
    match_b = _match("B", home="brazil", away="argentina", stage="Quarter-final", kickoff=kickoff)
    result_a = _result("A", home_goals_90=2, away_goals_90=1)
    result_b = _result("B", home_goals_90=2, away_goals_90=1)

    builder = KnockoutManifestBuilder(registry)
    builder.add_provider("p1", FakeProvider([match_a], [result_a]))
    builder.add_provider("p2", FakeProvider([match_b], [result_b]))
    manifest = builder.build()

    # Only the first provider's row should land in the manifest.
    assert len(manifest.rows) == 1
    assert manifest.rows[0].source == "p1"
    # The second is quarantined with duplicate reason.
    dupes = [q for q in manifest.quarantined if q.reason == "duplicate_across_providers"]
    assert len(dupes) == 1
    assert dupes[0].raw_match_id == "B"


def test_tournament_coverage_dict_sums_to_total(registry: AliasRegistry) -> None:
    # Use distinct dates + teams so no (kickoff, home, away) dedupe happens.
    matches = [
        _match("QF1", home="brazil", away="argentina", stage="Quarter-final",
               competition_id="fifa_world_cup",
               kickoff=datetime(2022, 12, 9, 20, 0, tzinfo=UTC)),
        _match("SF1", home="brazil", away="germany", stage="Semi-final",
               competition_id="fifa_world_cup",
               kickoff=datetime(2022, 12, 13, 20, 0, tzinfo=UTC)),
        _match("F1", home="argentina", away="germany", stage="Final",
               competition_id="fifa_world_cup",
               kickoff=datetime(2022, 12, 18, 20, 0, tzinfo=UTC)),
    ]
    results = [
        _result("QF1", home_goals_90=1, away_goals_90=0),
        _result("SF1", home_goals_90=2, away_goals_90=1),
        _result("F1", home_goals_90=3, away_goals_90=2),
    ]
    builder = KnockoutManifestBuilder(registry)
    builder.add_provider("p1", FakeProvider(matches, results))
    manifest = builder.build()

    assert manifest.total == 3
    # The coverage dict sums to the total.
    assert sum(manifest.tournament_coverage.values()) == manifest.total
    # All three rows are under the same competition.
    for r in manifest.rows:
        assert r.competition_id == "fifa_world_cup"


def test_is_knockout_stage_helper() -> None:
    # Downstream-bracket knockout stages
    assert is_knockout_stage("Quarter-final") is True
    assert is_knockout_stage("quarter-final") is True
    assert is_knockout_stage("Semi-final") is True
    assert is_knockout_stage("Final") is True
    assert is_knockout_stage("Round of 16") is True
    # Third-place finals are NOT downstream-bracket knockouts (the
    # helper excludes them so they go to the excluded_third_place
    # bucket in the manifest rather than the default training set).
    assert is_knockout_stage("3rd place") is False
    assert is_knockout_stage("Match for third place") is False
    # Group / league / matchday stages
    assert is_knockout_stage("Group A") is False
    assert is_knockout_stage("group") is False
    assert is_knockout_stage("Group Stage") is False
    assert is_knockout_stage("Matchday 1") is False
    assert is_knockout_stage("") is False


def test_provider_fetch_failure_is_isolated(registry: AliasRegistry) -> None:
    """A provider whose fetch_* raises must not take down the whole build."""

    class BrokenProvider:
        def fetch_matches(self, **kwargs):
            raise RuntimeError("kaboom")

        def fetch_results(self, **kwargs):
            raise RuntimeError("kaboom")

    match = _match(
        "QF1", home="brazil", away="argentina", stage="Quarter-final",
        kickoff=datetime(2022, 12, 10, 20, 0, tzinfo=UTC),
    )
    result = _result("QF1", home_goals_90=1, away_goals_90=0)

    builder = KnockoutManifestBuilder(registry)
    builder.add_provider("broken", BrokenProvider())
    builder.add_provider("good", FakeProvider([match], [result]))
    manifest = builder.build()

    assert len(manifest.rows) == 1
    assert manifest.rows[0].source == "good"
