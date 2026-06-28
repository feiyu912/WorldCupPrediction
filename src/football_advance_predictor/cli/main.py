"""Typer CLI entrypoint for football-advance-predictor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
import yaml
from football_advance_predictor.backtesting.backtest_runner import BacktestRunner
from football_advance_predictor.backtesting.splits.walk_forward import (
    WalkForwardConfig,
)
from football_advance_predictor.core.config import get_settings
from football_advance_predictor.core.logging import configure_logging, get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.data.adapters import (
    LocalAvailabilityProvider,
    LocalHistoricalResultsProvider,
    LocalOddsProvider,
)
from football_advance_predictor.data.ingestion.ingestion_service import IngestionService
from football_advance_predictor.data.snapshots.snapshot_service import FeatureSnapshotService
from football_advance_predictor.db.session import init_db, session_scope
from football_advance_predictor.ledger.ledger_service import LedgerService
from football_advance_predictor.models.calibration.calibrator import (
    CalibrationConfig,
)
from football_advance_predictor.models.catboost_model.catboost_model import (
    CatBoostConfig,
)
from football_advance_predictor.models.registry.registry import ModelRegistry
from football_advance_predictor.models.stacking.stacker import StackingConfig
from football_advance_predictor.services.prediction_service import PredictionService
from football_advance_predictor.services.training_service import TrainingService

app = typer.Typer(help="Football advance predictor CLI", no_args_is_help=True)
logger = get_logger(__name__)

predict_app = typer.Typer(help="Prediction commands")
features_app = typer.Typer(help="Feature commands")
models_app = typer.Typer(help="Model commands")
backtest_app = typer.Typer(help="Backtest commands")
report_app = typer.Typer(help="Report commands")
ingest_app = typer.Typer(help="Ingestion commands")
ledger_app = typer.Typer(help="Ledger commands")

app.add_typer(predict_app, name="predict")
app.add_typer(features_app, name="features")
app.add_typer(models_app, name="models")
app.add_typer(backtest_app, name="backtest")
app.add_typer(report_app, name="report")
app.add_typer(ingest_app, name="ingest")
app.add_typer(ledger_app, name="ledger")


def _setup() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    init_db()


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


@ingest_app.command("matches")
def ingest_matches(file: Path = typer.Option(..., "--file")) -> None:
    """Ingest historical matches from a local CSV file."""
    _setup()
    provider = LocalHistoricalResultsProvider(file)
    with session_scope() as session:
        service = IngestionService(session)
        teams = provider.fetch_teams()
        service.upsert_teams(teams)
        matches = provider.fetch_matches()
        n_matches = service.upsert_matches(matches)
        results = provider.fetch_results()
        for r in results:
            service.upsert_result(r)
    typer.echo(f"Ingested {n_matches} matches, {len(results)} results, {len(teams)} teams.")


@ingest_app.command("odds")
def ingest_odds(file: Path = typer.Option(..., "--file")) -> None:
    """Ingest market odds from a local CSV file."""
    _setup()
    provider = LocalOddsProvider(file)
    with session_scope() as session:
        service = IngestionService(session)
        records = provider.fetch_odds()
        n = service.upsert_odds(records)
    typer.echo(f"Ingested {n} new odds records.")


@ingest_app.command("availability")
def ingest_availability(file: Path = typer.Option(..., "--file")) -> None:
    """Ingest availability records from a local JSON file."""
    _setup()
    provider = LocalAvailabilityProvider(file)
    with session_scope() as session:
        service = IngestionService(session)
        records = provider.fetch_availability()
        n = service.upsert_availability(records)
    typer.echo(f"Ingested {n} new availability records.")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


@features_app.command("build")
def build_features(
    match_id: str = typer.Option(..., "--match-id"),
    cutoff: str = typer.Option(..., "--cutoff"),
    feature_version: str = typer.Option("v1", "--feature-version"),
) -> None:
    """Build a feature snapshot for a (match, cutoff)."""
    _setup()
    cutoff_dt = to_utc(cutoff)
    with session_scope() as session:
        service = FeatureSnapshotService(session)
        snapshot = service.build_or_get(
            match_id=match_id, cutoff_time=cutoff_dt, feature_version=feature_version
        )
    typer.echo(json.dumps({
        "feature_snapshot_id": snapshot.feature_snapshot_id,
        "match_id": snapshot.match_id,
        "cutoff_time": snapshot.cutoff_time.isoformat(),
        "feature_version": snapshot.feature_version,
        "immutable_hash": snapshot.immutable_hash,
        "feature_keys": list(snapshot.features_json.keys()),
    }, indent=2))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@models_app.command("train")
def train_model(config: Path = typer.Option(..., "--config")) -> None:
    """Train a model version from a YAML config."""
    _setup()
    data = _load_yaml(config)
    mvp = data.get("mvp", data)
    settings = get_settings()
    registry = ModelRegistry(settings.model_registry_dir)
    with session_scope() as session:
        service = TrainingService(session, registry)
        result = service.train(
            model_version=mvp["model_version"],
            training_window=(to_utc(mvp["training_window"]["start"]), to_utc(mvp["training_window"]["end"])),
            validation_window=(to_utc(mvp["validation_window"]["start"]), to_utc(mvp["validation_window"]["end"])),
            test_window=(to_utc(mvp["test_window"]["start"]), to_utc(mvp["test_window"]["end"])),
            catboost_config=CatBoostConfig.from_dict(_load_yaml(Path("configs/catboost.yaml"))["catboost"]),
            stacking_config=StackingConfig.from_dict(_load_yaml(Path("configs/stacking.yaml"))["stacking"]),
            calibration_config=CalibrationConfig.from_dict(_load_yaml(Path("configs/calibration.yaml"))["calibration"]),
        )
    typer.echo(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


@predict_app.command("one")
def predict_one(
    match_id: str = typer.Option(..., "--match-id"),
    cutoff: str = typer.Option(..., "--cutoff"),
    model_version: str = typer.Option(..., "--model-version"),
    feature_version: str = typer.Option("v1", "--feature-version"),
) -> None:
    """Produce a single prediction for a (match, cutoff)."""
    _setup()
    settings = get_settings()
    registry = ModelRegistry(settings.model_registry_dir)
    cutoff_dt = to_utc(cutoff)
    from football_advance_predictor.models.calibration.calibrator import Calibrator
    from football_advance_predictor.models.catboost_model.catboost_model import CatBoostModel
    from football_advance_predictor.models.stacking.stacker import StackingModel

    catboost = CatBoostModel.load(registry.root / "catboost" / model_version)
    stacker = StackingModel.load(registry.root / "stacking" / model_version)
    calibrator = Calibrator.load(registry.root / "calibration" / model_version)
    with session_scope() as session:
        service = PredictionService(
            session,
            catboost_model=catboost,
            stacker=stacker,
            calibrator=calibrator,
        )
        result = service.predict(
            match_id=match_id,
            cutoff_time=cutoff_dt,
            model_version=model_version,
            feature_version=feature_version,
        )
    typer.echo(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@backtest_app.command("run")
def backtest_run(
    config: Path = typer.Option(..., "--config"),
    model_version: str = typer.Option("v0_backtest", "--model-version"),
) -> None:
    """Run a temporal backtest across all configured folds."""
    _setup()
    data = _load_yaml(config)
    cfg = WalkForwardConfig.from_yaml(data.get("backtest", data))
    settings = get_settings()
    registry = ModelRegistry(settings.model_registry_dir)
    from football_advance_predictor.db.session import session_scope

    with session_scope() as session:
        runner = BacktestRunner(session, registry)
        result = runner.run(
            model_version=model_version,
            config=cfg,
            catboost_config=CatBoostConfig.from_dict(_load_yaml(Path("configs/catboost.yaml"))["catboost"]),
            stacking_config=StackingConfig.from_dict(_load_yaml(Path("configs/stacking.yaml"))["stacking"]),
            calibration_config=CalibrationConfig.from_dict(_load_yaml(Path("configs/calibration.yaml"))["calibration"]),
        )
    typer.echo(json.dumps(result, indent=2, default=str))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@report_app.command("show")
def report_show(run_id: str = typer.Option(..., "--run-id")) -> None:
    """Print a previously generated backtest report."""
    _setup()
    settings = get_settings()
    path = Path(settings.model_registry_dir) / ".." / "reports" / f"{run_id}_summary.json"
    candidates = [Path("reports") / f"{run_id}_summary.json", path]
    for candidate in candidates:
        if candidate.exists():
            typer.echo(candidate.read_text(encoding="utf-8"))
            return
    typer.echo(f"Report not found: {run_id}")


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


@ledger_app.command("list")
def ledger_list(
    match_id: str | None = typer.Option(None, "--match-id"),
    model_version: str | None = typer.Option(None, "--model-version"),
) -> None:
    """List predictions in the ledger."""
    _setup()
    with session_scope() as session:
        service = LedgerService(session)
        rows = service.list_predictions(match_id=match_id, model_version=model_version)
    out = [
        {
            "prediction_id": p.prediction_id,
            "match_id": p.match_id,
            "cutoff_time": p.cutoff_time.isoformat(),
            "model_version": p.model_version,
            "home_advance_probability": p.home_advance_probability,
            "predicted_advancer_id": p.predicted_advancer_id,
            "confidence_band": p.confidence_band,
        }
        for p in rows
    ]
    typer.echo(json.dumps(out, indent=2, default=str))


@ledger_app.command("evaluate")
def ledger_evaluate(
    prediction_id: str = typer.Option(..., "--prediction-id"),
    actual_home_advances: bool = typer.Option(..., "--actual-home-advances"),
) -> None:
    """Evaluate a single prediction against an actual outcome."""
    _setup()
    with session_scope() as session:
        service = LedgerService(session)
        record = service.evaluate_prediction(prediction_id, actual_home_advances)
    typer.echo(json.dumps({
        "evaluation_id": record.evaluation_id,
        "prediction_id": record.prediction_id,
        "actual_home_advances": record.actual_home_advances,
        "log_loss": record.log_loss,
        "brier_score": record.brier_score,
        "correct_classification": record.correct_classification,
    }, indent=2))


@ledger_app.command("export")
def ledger_export(
    output: Path = typer.Option(..., "--output"),
    match_id: str | None = typer.Option(None, "--match-id"),
) -> None:
    """Export the ledger to CSV or Parquet."""
    _setup()
    fmt = output.suffix.lower()
    with session_scope() as session:
        service = LedgerService(session)
        if fmt == ".parquet":
            service.export_parquet(output, match_id=match_id)
        else:
            service.export_csv(output, match_id=match_id)
    typer.echo(f"Exported ledger to {output}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    app()
