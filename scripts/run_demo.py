"""End-to-end demo script that ingests fixtures, trains a model, and prints a prediction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from football_advance_predictor.core.logging import configure_logging, get_logger
from football_advance_predictor.data.adapters import (
    LocalAvailabilityProvider,
    LocalHistoricalResultsProvider,
    LocalOddsProvider,
)
from football_advance_predictor.data.ingestion.ingestion_service import IngestionService
from football_advance_predictor.db.session import init_db, session_scope
from football_advance_predictor.features.elo.elo_engine import EloConfig
from football_advance_predictor.models.calibration.calibrator import Calibrator
from football_advance_predictor.models.catboost_model.catboost_model import (
    CatBoostConfig,
    CatBoostModel,
)
from football_advance_predictor.models.registry.registry import ModelRegistry
from football_advance_predictor.models.stacking.stacker import StackingConfig, StackingModel
from football_advance_predictor.services.prediction_service import PredictionService
from football_advance_predictor.services.training_service import TrainingService

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    init_db()
    fixtures = Path("data/fixtures")

    with session_scope() as session:
        provider = LocalHistoricalResultsProvider(fixtures / "matches.csv")
        ing = IngestionService(session)
        ing.upsert_teams(provider.fetch_teams())
        ing.upsert_matches(provider.fetch_matches())
        for r in provider.fetch_results():
            ing.upsert_result(r)
        ing.upsert_odds(LocalOddsProvider(fixtures / "odds.csv").fetch_odds())
        ing.upsert_availability(LocalAvailabilityProvider(fixtures / "availability.json").fetch_availability())

    registry = ModelRegistry(Path("data/processed/models"))
    with session_scope() as session:
        training = TrainingService(session, registry, elo_config=EloConfig(base_k_factor=10.0))
        result = training.train(
            model_version="v0_demo",
            training_window=(datetime(2017, 1, 1, tzinfo=timezone.utc), datetime(2021, 12, 31, tzinfo=timezone.utc)),
            validation_window=(datetime(2022, 1, 1, tzinfo=timezone.utc), datetime(2022, 6, 30, tzinfo=timezone.utc)),
            test_window=(datetime(2022, 7, 1, tzinfo=timezone.utc), datetime(2022, 12, 31, tzinfo=timezone.utc)),
            catboost_config=CatBoostConfig(iterations=30, depth=4, random_seed=42, verbose=False),
            stacking_config=StackingConfig(method="logistic_regression"),
        )
        print(json.dumps(result, indent=2, default=str))

    with session_scope() as session:
        catboost = CatBoostModel.load(registry.root / "catboost" / "v0_demo")
        stacker = StackingModel.load(registry.root / "stacking" / "v0_demo")
        calibrator = Calibrator.load(registry.root / "calibration" / "v0_demo")
        service = PredictionService(session, catboost_model=catboost, stacker=stacker, calibrator=calibrator)
        result = service.predict(
            match_id="MATCH_KO_001",
            cutoff_time=datetime(2026, 6, 30, 4, 0, 0, tzinfo=timezone.utc),
            model_version="v0_demo",
        )
    print("Prediction:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
