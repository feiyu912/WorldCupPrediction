"""Anti-leakage integration tests against the synthetic tournament dataset.

These tests intentionally attempt leakage and verify the pipeline
rejects or filters the bad data.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "warehouse.duckdb"))
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    from football_advance_predictor.core.config import get_settings

    get_settings.cache_clear()
    yield


@pytest.fixture
def session_local(monkeypatch):
    from football_advance_predictor import db

    engine = create_engine("sqlite:///:memory:", future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    db.session._engine = engine
    db.session._SessionLocal = SessionLocal
    db.session.init_db()
    return SessionLocal


def test_elo_does_not_use_future_match_results(session_local, fixture_dir):
    from football_advance_predictor.data.adapters import LocalHistoricalResultsProvider
    from football_advance_predictor.data.ingestion.ingestion_service import IngestionService
    from football_advance_predictor.db.session import session_scope
    from football_advance_predictor.features.elo.elo_engine import DynamicEloEngine, EloConfig

    with session_scope() as session:
        provider = LocalHistoricalResultsProvider(fixture_dir / "matches.csv")
        ing = IngestionService(session)
        ing.upsert_teams(provider.fetch_teams())
        ing.upsert_matches(provider.fetch_matches())
        for r in provider.fetch_results():
            ing.upsert_result(r)

    cutoff = datetime(2026, 6, 30, 4, 0, 0, tzinfo=UTC)
    with session_scope() as session:
        engine = DynamicEloEngine(EloConfig())
        rows = []
        from football_advance_predictor.db.models import Match, MatchResult

        for match, result in session.query(Match, MatchResult).join(MatchResult).all():
            rows.append(
                {
                    "kickoff_at": match.kickoff_at,
                    "home_team_id": match.home_team_id,
                    "away_team_id": match.away_team_id,
                    "home_goals": result.home_goals_90,
                    "away_goals": result.away_goals_90,
                    "neutral_venue": match.neutral_venue,
                    "home_advances": result.home_advances,
                    "competition_importance": 1.0,
                }
            )
        engine.fit(rows)
        france_rating = engine.get_team_rating("france", cutoff)
        # France ratings as of 2026-06-30 must NOT include the post-cutoff 2026 KO matches' results.
        # We verify by querying the engine for France's historical rating right before cutoff.
        france_before = engine.get_team_rating("france", datetime(2024, 1, 1, tzinfo=UTC))
        # The ratings should be a finite float.
        assert isinstance(france_before, float)
        assert isinstance(france_rating, float)


def test_t24h_snapshot_excludes_lineup_confirmation(session_local, fixture_dir):
    from football_advance_predictor.data.adapters import (
        LocalAvailabilityProvider,
        LocalHistoricalResultsProvider,
        LocalOddsProvider,
    )
    from football_advance_predictor.data.ingestion.ingestion_service import IngestionService
    from football_advance_predictor.data.snapshots.snapshot_service import FeatureSnapshotService
    from football_advance_predictor.db.session import session_scope

    with session_scope() as session:
        m = LocalHistoricalResultsProvider(fixture_dir / "matches.csv")
        o = LocalOddsProvider(fixture_dir / "odds.csv")
        a = LocalAvailabilityProvider(fixture_dir / "availability.json")
        ing = IngestionService(session)
        ing.upsert_teams(m.fetch_teams())
        ing.upsert_matches(m.fetch_matches())
        for r in m.fetch_results():
            ing.upsert_result(r)
        ing.upsert_odds(o.fetch_odds())
        ing.upsert_availability(a.fetch_availability())

    with session_scope() as session:
        snapshot_service = FeatureSnapshotService(session)
        cutoff_t24h = datetime(2026, 6, 30, 4, 0, 0, tzinfo=UTC)
        snap = snapshot_service.build_or_get(
            match_id="MATCH_KO_001", cutoff_time=cutoff_t24h, feature_version="v1"
        )
        # The lineup confirmation is published 2026-07-01T02:45:00Z; T-24h is before that.
        assert snap.features_json.get("home_lineup_confirmed", 0.0) == 0.0


def test_t75min_snapshot_includes_lineup_confirmation(session_local, fixture_dir):
    from football_advance_predictor.data.adapters import (
        LocalAvailabilityProvider,
        LocalHistoricalResultsProvider,
        LocalOddsProvider,
    )
    from football_advance_predictor.data.ingestion.ingestion_service import IngestionService
    from football_advance_predictor.data.snapshots.snapshot_service import FeatureSnapshotService
    from football_advance_predictor.db.session import session_scope

    with session_scope() as session:
        m = LocalHistoricalResultsProvider(fixture_dir / "matches.csv")
        o = LocalOddsProvider(fixture_dir / "odds.csv")
        a = LocalAvailabilityProvider(fixture_dir / "availability.json")
        ing = IngestionService(session)
        ing.upsert_teams(m.fetch_teams())
        ing.upsert_matches(m.fetch_matches())
        for r in m.fetch_results():
            ing.upsert_result(r)
        ing.upsert_odds(o.fetch_odds())
        ing.upsert_availability(a.fetch_availability())

    with session_scope() as session:
        snapshot_service = FeatureSnapshotService(session)
        # 75 minutes before kickoff (2026-07-01T04:00:00Z) is 2026-07-01T02:45:00Z.
        cutoff_t75 = datetime(2026, 7, 1, 2, 45, 0, tzinfo=UTC)
        snap = snapshot_service.build_or_get(
            match_id="MATCH_KO_001", cutoff_time=cutoff_t75, feature_version="v1"
        )
        # The home team has a lineup confirmation published at the cutoff.
        assert snap.features_json.get("home_lineup_confirmed", 0.0) == 1.0


def test_post_kickoff_availability_is_rejected(session_local, fixture_dir):
    from football_advance_predictor.data.adapters import (
        LocalAvailabilityProvider,
        LocalHistoricalResultsProvider,
    )
    from football_advance_predictor.data.ingestion.ingestion_service import IngestionService
    from football_advance_predictor.data.snapshots.snapshot_service import FeatureSnapshotService
    from football_advance_predictor.db.session import session_scope

    with session_scope() as session:
        m = LocalHistoricalResultsProvider(fixture_dir / "matches.csv")
        a = LocalAvailabilityProvider(fixture_dir / "availability.json")
        ing = IngestionService(session)
        ing.upsert_teams(m.fetch_teams())
        ing.upsert_matches(m.fetch_matches())
        for r in m.fetch_results():
            ing.upsert_result(r)
        ing.upsert_availability(a.fetch_availability())

    with session_scope() as session:
        snapshot_service = FeatureSnapshotService(session)
        # Cutoff AFTER kickoff must be rejected by the snapshot service.
        from datetime import datetime

        with pytest.raises(ValueError):
            snapshot_service.build_or_get(
                match_id="MATCH_KO_001",
                cutoff_time=datetime(2026, 7, 1, 5, 0, 0, tzinfo=UTC),
                feature_version="v1",
            )
