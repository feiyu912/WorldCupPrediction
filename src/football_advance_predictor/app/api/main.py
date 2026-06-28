"""FastAPI app exposing core endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.orm import Session

from football_advance_predictor.core.config import get_settings
from football_advance_predictor.core.logging import configure_logging, get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.data.adapters import (
    LocalHistoricalResultsProvider,
)
from football_advance_predictor.data.ingestion.ingestion_service import IngestionService
from football_advance_predictor.data.snapshots.snapshot_service import FeatureSnapshotService
from football_advance_predictor.db.session import get_session, init_db
from football_advance_predictor.ledger.ledger_service import LedgerService
from football_advance_predictor.models.calibration.calibrator import Calibrator
from football_advance_predictor.models.catboost_model.catboost_model import CatBoostModel
from football_advance_predictor.models.registry.registry import ModelRegistry
from football_advance_predictor.models.stacking.stacker import StackingModel
from football_advance_predictor.schemas.availability import AvailabilityIn
from football_advance_predictor.schemas.features import FeatureBuildRequest
from football_advance_predictor.schemas.matches import MatchIn
from football_advance_predictor.schemas.odds import MarketOddsIn
from football_advance_predictor.schemas.predictions import (
    EvaluationOut,
    PredictionOut,
    PredictionRequest,
)
from football_advance_predictor.services.prediction_service import PredictionService

app = FastAPI(title="Football Advance Predictor", version="0.1.0")
logger = get_logger(__name__)
_settings = get_settings()
configure_logging(_settings.log_level)


@app.on_event("startup")
def _startup() -> None:
    try:
        init_db()
    except Exception as exc:  # pragma: no cover - DB might not be running
        logger.warning("init_db failed; continuing without schema bootstrap", extra={"error": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@app.post("/ingest/matches")
def ingest_matches(
    matches: list[MatchIn], session: Session = Depends(get_session)
) -> dict[str, int]:
    service = IngestionService(session)
    n = service.upsert_matches(matches)
    session.commit()
    return {"matches_ingested": n}


@app.post("/ingest/matches/file")
def ingest_matches_file(
    file_path: str, session: Session = Depends(get_session)
) -> dict[str, int]:
    provider = LocalHistoricalResultsProvider(Path(file_path))
    service = IngestionService(session)
    teams = provider.fetch_teams()
    service.upsert_teams(teams)
    matches = provider.fetch_matches()
    n = service.upsert_matches(matches)
    for r in provider.fetch_results():
        service.upsert_result(r)
    session.commit()
    return {"matches_ingested": n, "teams_ingested": len(teams)}


@app.post("/ingest/odds")
def ingest_odds(
    odds: list[MarketOddsIn], session: Session = Depends(get_session)
) -> dict[str, int]:
    service = IngestionService(session)
    n = service.upsert_odds(odds)
    session.commit()
    return {"odds_ingested": n}


@app.post("/ingest/availability")
def ingest_availability(
    records: list[AvailabilityIn], session: Session = Depends(get_session)
) -> dict[str, int]:
    service = IngestionService(session)
    n = service.upsert_availability(records)
    session.commit()
    return {"availability_ingested": n}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@app.post("/features/build")
def build_features(
    request: FeatureBuildRequest, session: Session = Depends(get_session)
) -> dict[str, Any]:
    service = FeatureSnapshotService(session)
    snapshot = service.build_or_get(
        match_id=request.match_id,
        cutoff_time=to_utc(request.cutoff_time),
        feature_version=request.feature_version,
    )
    session.commit()
    return {
        "feature_snapshot_id": snapshot.feature_snapshot_id,
        "match_id": snapshot.match_id,
        "cutoff_time": snapshot.cutoff_time.isoformat(),
        "feature_version": snapshot.feature_version,
        "immutable_hash": snapshot.immutable_hash,
        "feature_keys": list(snapshot.features_json.keys()),
        "source_data_max_timestamp": snapshot.source_data_max_timestamp.isoformat(),
    }


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


@app.post("/predictions", response_model=PredictionOut)
def create_prediction(
    request: PredictionRequest,
    session: Session = Depends(get_session),
) -> PredictionOut:
    settings = get_settings()
    registry = ModelRegistry(settings.model_registry_dir)
    catboost = None
    stacker = None
    calibrator = None
    try:
        catboost = CatBoostModel.load(registry.root / "catboost" / request.model_version)
        stacker = StackingModel.load(registry.root / "stacking" / request.model_version)
        calibrator = Calibrator.load(registry.root / "calibration" / request.model_version)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    service = PredictionService(
        session,
        catboost_model=catboost,
        stacker=stacker,
        calibrator=calibrator,
    )
    try:
        result = service.predict(
            match_id=request.match_id,
            cutoff_time=to_utc(request.cutoff_time),
            model_version=request.model_version,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    session.commit()
    return PredictionOut.model_validate(service.ledger.get_prediction(result["prediction_id"]))


@app.get("/predictions/{prediction_id}", response_model=PredictionOut)
def get_prediction(
    prediction_id: str, session: Session = Depends(get_session)
) -> PredictionOut:
    service = LedgerService(session)
    prediction = service.get_prediction(prediction_id)
    if prediction is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return PredictionOut.model_validate(prediction)


@app.post("/predictions/{prediction_id}/evaluate", response_model=EvaluationOut)
def evaluate_prediction(
    prediction_id: str,
    actual_home_advances: bool,
    session: Session = Depends(get_session),
) -> EvaluationOut:
    service = LedgerService(session)
    try:
        record = service.evaluate_prediction(prediction_id, actual_home_advances)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    session.commit()
    return EvaluationOut.model_validate(record)


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@app.get("/backtests")
def list_backtests() -> dict[str, Any]:
    settings = get_settings()
    base = Path("reports")
    if not base.exists():
        return {"backtests": []}
    items = []
    for path in sorted(base.glob("*_summary.json")):
        items.append({"run_id": path.stem.replace("_summary", ""), "path": str(path)})
    return {"backtests": items}


@app.get("/backtests/{run_id}")
def get_backtest(run_id: str) -> dict[str, Any]:
    base = Path("reports")
    summary = base / f"{run_id}_summary.json"
    if not summary.exists():
        raise HTTPException(status_code=404, detail="Backtest not found")
    import json

    with summary.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


@app.get("/reports/model-comparison")
def model_comparison(model_versions: str) -> dict[str, Any]:
    versions = [v.strip() for v in model_versions.split(",") if v.strip()]
    with session_scope_fresh() as session:
        service = LedgerService(session)
        return {"comparison": service.compare_model_versions(versions)}


@app.get("/reports/calibration")
def calibration_report(run_id: str) -> dict[str, Any]:
    base = Path("reports")
    path = base / f"{run_id}_summary.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    import json

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "run_id": data.get("run_id"),
        "summary": {k: v for k, v in data.get("summary", {}).items() if "calibrat" in k or k == "ece"},
        "per_fold_reliability": [
            {"fold": f["fold"], "reliability": f.get("reliability", [])}
            for f in data.get("per_fold", [])
        ],
    }


@app.get("/reports/feature-importance")
def feature_importance(model_version: str) -> dict[str, Any]:
    settings = get_settings()
    registry = ModelRegistry(settings.model_registry_dir)
    artifact = registry.get(model_type="catboost", model_version=model_version)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Model not found")
    model = CatBoostModel.load(artifact.artifact_path)
    return {"model_version": model_version, "importance": model.feature_importance()}


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def session_scope_fresh():
    from contextlib import contextmanager

    from football_advance_predictor.db.session import _session_factory

    @contextmanager
    def _ctx():
        SessionLocal = _session_factory()
        s = SessionLocal()
        try:
            yield s
        finally:
            s.close()

    return _ctx()
