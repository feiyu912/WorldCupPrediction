"""Top-level backtest runner.

The runner iterates the configured folds, trains the model stack on
each train window, generates predictions for the test window, and
records them for reporting. The validation window is used internally
by :class:`TrainingService` for stacking/calibration; the test window
is never touched for training.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_advance_predictor.backtesting.reports.report_service import BacktestReportService
from football_advance_predictor.backtesting.splits.walk_forward import (
    WalkForwardConfig,
    WalkForwardSplitter,
)
from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.db.models import (
    Competition,
    Match,
    MatchResult,
)
from football_advance_predictor.features.elo.elo_engine import EloConfig
from football_advance_predictor.models.calibration.calibrator import (
    CalibrationConfig,
    Calibrator,
)
from football_advance_predictor.models.catboost_model.catboost_model import (
    CatBoostConfig,
    CatBoostModel,
)
from football_advance_predictor.models.elo_model.elo_model import EloModel
from football_advance_predictor.models.market_model.market_model import MarketModel
from football_advance_predictor.models.registry.registry import ModelRegistry
from football_advance_predictor.models.stacking.stacker import StackingConfig, StackingModel
from football_advance_predictor.services.prediction_service import PredictionService

logger = get_logger(__name__)


class BacktestRunner:
    """Run a temporal backtest with expanding or rolling windows."""

    def __init__(
        self,
        session: Session,
        registry: ModelRegistry,
        *,
        feature_version: str = "v1",
        market_min_bookmakers: int = 1,
        elo_config: EloConfig | None = None,
        reports_dir: str | Path = "reports",
    ) -> None:
        self.session = session
        self.registry = registry
        self.feature_version = feature_version
        self.market_min_bookmakers = market_min_bookmakers
        self.elo_config = elo_config or EloConfig()
        self.reports_dir = Path(reports_dir)
        self.report_service = BacktestReportService(self.reports_dir)

    def run(
        self,
        *,
        model_version: str,
        config: WalkForwardConfig,
        catboost_config: CatBoostConfig | None = None,
        stacking_config: StackingConfig | None = None,
        calibration_config: CalibrationConfig | None = None,
    ) -> dict[str, Any]:
        """Run all folds and produce a report."""
        splitter = WalkForwardSplitter(config)
        records: list[dict[str, Any]] = []
        for fold in splitter.folds():
            logger.info(
                "Running fold",
                extra={"fold": fold.name, "test": fold.test_start.isoformat()},
            )
            self._train_fold(
                fold=fold,
                model_version=f"{model_version}_{fold.name}",
                catboost_config=catboost_config or CatBoostConfig(),
                stacking_config=stacking_config or StackingConfig(),
                calibration_config=calibration_config or CalibrationConfig(),
            )
            fold_records = self._predict_test_fold(
                fold=fold,
                model_version=f"{model_version}_{fold.name}",
            )
            for record in fold_records:
                record["fold"] = fold.name
            records.extend(fold_records)

        report = self.report_service.build_report(
            model_version=model_version, records=records
        )
        logger.info("Backtest complete", extra={"run_id": report.run_id})
        return {
            "run_id": report.run_id,
            "summary": report.summary,
            "per_fold": report.per_fold,
            "per_prediction_csv": report.per_prediction_csv,
            "per_prediction_parquet": report.per_prediction_parquet,
        }

    # ------------------------------------------------------------------
    # Fold-level helpers
    # ------------------------------------------------------------------

    def _train_fold(
        self,
        *,
        fold: Any,
        model_version: str,
        catboost_config: CatBoostConfig,
        stacking_config: StackingConfig,
        calibration_config: CalibrationConfig,
    ) -> None:
        from football_advance_predictor.services.training_service import TrainingService

        training = TrainingService(
            session=self.session,
            registry=self.registry,
            elo_config=self.elo_config,
            feature_version=self.feature_version,
            market_min_bookmakers=self.market_min_bookmakers,
        )
        training.train(
            model_version=model_version,
            training_window=(fold.train_start, fold.train_end),
            validation_window=(fold.validation_start, fold.validation_end),
            test_window=(fold.test_start, fold.test_end),
            catboost_config=catboost_config,
            stacking_config=stacking_config,
            calibration_config=calibration_config,
        )

    def _predict_test_fold(self, *, fold: Any, model_version: str) -> list[dict[str, Any]]:
        catboost = CatBoostModel.load(self.registry.root / "catboost" / model_version)
        stacker = StackingModel.load(self.registry.root / "stacking" / model_version)
        calibrator = Calibrator.load(self.registry.root / "calibration" / model_version)
        elo_model = EloModel(self.elo_config).fit(self._elo_history_before(fold.train_start))
        market_model = MarketModel(self.session, min_bookmakers=self.market_min_bookmakers)
        service = PredictionService(
            self.session,
            elo_config=self.elo_config,
            catboost_model=catboost,
            stacker=stacker,
            calibrator=calibrator,
            market_min_bookmakers=self.market_min_bookmakers,
        )

        stmt = (
            select(Match, MatchResult)
            .join(MatchResult, MatchResult.match_id == Match.match_id)
            .where(Match.kickoff_at >= fold.test_start)
            .where(Match.kickoff_at <= fold.test_end)
        )
        rows = self.session.execute(stmt).all()
        out: list[dict[str, Any]] = []
        for match, result in rows:
            cutoff = to_utc(match.kickoff_at) - timedelta(hours=24)
            try:
                prediction = service.predict(
                    match_id=match.match_id,
                    cutoff_time=cutoff,
                    model_version=model_version,
                    feature_version=self.feature_version,
                )
            except (ValueError, LookupError) as exc:
                logger.warning("Skipping prediction in fold", extra={"error": str(exc)})
                continue
            ledger_row = service.ledger.get_prediction(prediction["prediction_id"])
            if ledger_row is None:
                logger.warning("Ledger row missing for prediction", extra={"prediction_id": prediction["prediction_id"]})
                continue
            out.append(
                {
                    "match_id": match.match_id,
                    "cutoff_time": cutoff.isoformat(),
                    "home_advance_probability": prediction["home_advance_probability"],
                    "market_probability": ledger_row.market_probability,
                    "elo_probability": ledger_row.elo_probability,
                    "home_advances": int(result.home_advances),
                }
            )
        return out

    def _elo_history_before(self, before: datetime) -> list[dict[str, Any]]:
        stmt = (
            select(Match, MatchResult, Competition)
            .join(MatchResult, MatchResult.match_id == Match.match_id)
            .outerjoin(Competition, Competition.competition_id == Match.competition_id)
            .where(Match.kickoff_at < to_utc(before))
            .order_by(Match.kickoff_at)
        )
        rows = self.session.execute(stmt).all()
        out: list[dict[str, Any]] = []
        for match, result, competition in rows:
            out.append(
                {
                    "kickoff_at": match.kickoff_at,
                    "home_team_id": match.home_team_id,
                    "away_team_id": match.away_team_id,
                    "home_goals": result.home_goals_90,
                    "away_goals": result.away_goals_90,
                    "neutral_venue": match.neutral_venue,
                    "home_advances": result.home_advances,
                    "competition_importance": (
                        float(competition.importance_weight) if competition else 1.0
                    ),
                }
            )
        return out
