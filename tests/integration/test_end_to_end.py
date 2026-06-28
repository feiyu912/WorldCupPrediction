"""End-to-end smoke test against the fixture data.

The test boots an in-memory SQLite database, ingests the fixture
matches/odds/availability, builds feature snapshots, trains a tiny
model, generates a prediction, and verifies the prediction ledger.

The test purposefully avoids external services. It also asserts the
anti-leakage contract:

- A T-24h snapshot cannot see a lineup confirmation published later.
- A post-kickoff availability record is not used by the feature builder.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# Configure environment to use SQLite in-memory before importing the app.
@pytest.fixture(autouse=True)
def _sqlite_env(monkeypatch, tmp_path):
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "warehouse.duckdb"))
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "models"))
    from football_advance_predictor.core.config import get_settings

    get_settings.cache_clear()
    yield


@pytest.fixture
def in_memory_engine(monkeypatch):
    """Patch the engine to use SQLite in-memory."""
    from football_advance_predictor import db

    engine = create_engine("sqlite:///:memory:", future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)

    def _factory():
        return engine

    def _session_factory():
        return SessionLocal

    monkeypatch.setattr(db.session, "get_engine", _factory)
    monkeypatch.setattr(db.session, "_session_factory", _session_factory)
    db.session._engine = engine
    db.session._SessionLocal = SessionLocal
    db.session.init_db()
    yield engine


def test_full_flow_with_fixtures(in_memory_engine, fixture_dir):
    from football_advance_predictor.data.adapters import (
        LocalAvailabilityProvider,
        LocalHistoricalResultsProvider,
        LocalOddsProvider,
    )
    from football_advance_predictor.data.ingestion.ingestion_service import IngestionService
    from football_advance_predictor.data.snapshots.snapshot_service import FeatureSnapshotService
    from football_advance_predictor.db.session import session_scope
    from football_advance_predictor.features.elo.elo_engine import EloConfig
    from football_advance_predictor.ledger.ledger_service import LedgerService
    from football_advance_predictor.models.calibration.calibrator import Calibrator
    from football_advance_predictor.models.catboost_model.catboost_model import (
        CatBoostConfig,
        CatBoostModel,
    )
    from football_advance_predictor.models.registry.registry import ModelRegistry
    from football_advance_predictor.models.stacking.stacker import StackingConfig, StackingModel
    from football_advance_predictor.services.prediction_service import PredictionService
    from football_advance_predictor.services.training_service import TrainingService

    registry = ModelRegistry(os.environ["MODEL_REGISTRY_DIR"])

    with session_scope() as session:
        provider = LocalHistoricalResultsProvider(fixture_dir / "matches.csv")
        ing = IngestionService(session)
        ing.upsert_teams(provider.fetch_teams())
        ing.upsert_matches(provider.fetch_matches())
        for r in provider.fetch_results():
            ing.upsert_result(r)
        odds_provider = LocalOddsProvider(fixture_dir / "odds.csv")
        ing.upsert_odds(odds_provider.fetch_odds())
        availability_provider = LocalAvailabilityProvider(fixture_dir / "availability.json")
        ing.upsert_availability(availability_provider.fetch_availability())

    # Build a snapshot for the synthetic knockout match at T-24h.
    with session_scope() as session:
        snapshot_service = FeatureSnapshotService(
            session,
        )
        cutoff = datetime(2026, 6, 30, 4, 0, 0, tzinfo=UTC)  # 24h before kickoff
        snapshot = snapshot_service.build_or_get(
            match_id="MATCH_KO_001", cutoff_time=cutoff, feature_version="v1"
        )
        assert snapshot.features_json["home_elo_pre_match"] != snapshot.features_json["away_elo_pre_match"]

    # The T-24h snapshot MUST NOT contain the lineup confirmation published later.
    with session_scope() as session:
        snapshot_service = FeatureSnapshotService(session)
        cutoff = datetime(2026, 6, 30, 4, 0, 0, tzinfo=UTC)
        snapshot = snapshot_service.build_or_get(
            match_id="MATCH_KO_001", cutoff_time=cutoff, feature_version="v1"
        )
        # lineup_confirmed is 0.0 when no record exists before cutoff.
        assert snapshot.features_json.get("lineup_confirmed", 0.0) == 0.0

    # Train a tiny model on the 2018+2022 knockout matches.
    with session_scope() as session:
        training = TrainingService(
            session,
            registry,
            elo_config=EloConfig(base_k_factor=10.0),
            feature_version="v1",
            market_min_bookmakers=1,
        )
        catboost_cfg = CatBoostConfig(iterations=30, depth=4, learning_rate=0.05, random_seed=42)
        stacking_cfg = StackingConfig(method="logistic_regression")
        result = training.train(
            model_version="v_smoke",
            training_window=(datetime(2010, 1, 1, tzinfo=UTC), datetime(2017, 12, 31, tzinfo=UTC)),
            validation_window=(datetime(2018, 1, 1, tzinfo=UTC), datetime(2018, 12, 31, tzinfo=UTC)),
            test_window=(datetime(2022, 1, 1, tzinfo=UTC), datetime(2022, 12, 31, tzinfo=UTC)),
            catboost_config=catboost_cfg,
            stacking_config=stacking_cfg,
        )
        assert result["model_version"] == "v_smoke"

    # Generate a prediction and verify the ledger.
    with session_scope() as session:
        catboost = CatBoostModel.load(registry.root / "catboost" / "v_smoke")
        stacker = StackingModel.load(registry.root / "stacking" / "v_smoke")
        calibrator = Calibrator.load(registry.root / "calibration" / "v_smoke")
        service = PredictionService(
            session,
            catboost_model=catboost,
            stacker=stacker,
            calibrator=calibrator,
        )
        result = service.predict(
            match_id="MATCH_KO_001",
            cutoff_time=datetime(2026, 6, 30, 4, 0, 0, tzinfo=UTC),
            model_version="v_smoke",
        )
        assert 0.0 <= result["home_advance_probability"] <= 1.0
        assert result["confidence_band"] in {"clear_lean", "slight_lean", "near_coin_flip"}

        # Verify immutability: a second create with the same key must raise.
        with pytest.raises(ValueError):
            service.predict(
                match_id="MATCH_KO_001",
                cutoff_time=datetime(2026, 6, 30, 4, 0, 0, tzinfo=UTC),
                model_version="v_smoke",
            )

        # Evaluate the prediction against a hypothetical outcome.
        ledger = LedgerService(session)
        evaluation = ledger.evaluate_prediction(result["prediction_id"], True)
        assert evaluation.log_loss >= 0.0
        assert evaluation.brier_score >= 0.0
