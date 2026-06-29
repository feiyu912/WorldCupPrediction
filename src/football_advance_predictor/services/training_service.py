"""Training service: orchestrates temporal training, OOF stacking, and calibration.

The service is responsible for:

1. Building a training table with one row per knockout match and its
   time-frozen features.
2. Fitting the **default base model** (regularized logistic regression)
   on the training window. CatBoost is opt-in via
   ``catboost.enabled`` AND requires the training set to be at least
   ``catboost.min_samples_to_enable`` rows; otherwise the system stays
   on logistic regression with a warning.
3. Generating out-of-fold predictions on the validation window.
4. Fitting the stacker and calibrator on the validation window.
5. Recording the model run, including metrics and artifact paths.

It does NOT evaluate on the test window. That is the backtester's job.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_advance_predictor.core.hashing import stable_hash
from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.db.models import (
    Competition,
    Match,
    MatchResult,
)
from football_advance_predictor.features.builders.feature_builder import FeatureBuilder
from football_advance_predictor.features.elo.elo_engine import EloConfig
from football_advance_predictor.models.calibration.calibrator import CalibrationConfig, Calibrator
from football_advance_predictor.models.catboost_model.catboost_model import (
    CatBoostConfig,
    CatBoostModel,
)
from football_advance_predictor.models.elo_model.elo_model import EloModel
from football_advance_predictor.models.logistic_baseline import (
    LogisticRegressionBaseline,
    LogisticRegressionBaselineConfig,
)
from football_advance_predictor.models.market_model.market_model import MarketModel
from football_advance_predictor.models.registry.registry import ModelRegistry
from football_advance_predictor.models.stacking.stacker import StackingConfig, StackingModel

logger = get_logger(__name__)


class TrainingService:
    """Train a versioned model stack on a temporal split."""

    def __init__(
        self,
        session: Session,
        registry: ModelRegistry,
        *,
        elo_config: EloConfig | None = None,
        feature_version: str = "v1",
        market_min_bookmakers: int = 1,
        models_config: dict[str, Any] | None = None,
    ) -> None:
        self.session = session
        self.registry = registry
        self.elo_config = elo_config or EloConfig()
        self.feature_version = feature_version
        self.market_min_bookmakers = market_min_bookmakers
        self.models_config = models_config or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        *,
        model_version: str,
        training_window: tuple[datetime, datetime],
        validation_window: tuple[datetime, datetime],
        test_window: tuple[datetime, datetime],
        catboost_config: CatBoostConfig | None = None,
        stacking_config: StackingConfig | None = None,
        calibration_config: CalibrationConfig | None = None,
        logistic_config: LogisticRegressionBaselineConfig | None = None,
    ) -> dict[str, Any]:
        """Run the full training pipeline.

        Returns:
            A dict of metrics and artifact paths.
        """
        catboost_config = catboost_config or CatBoostConfig()
        stacking_config = stacking_config or StackingConfig()
        calibration_config = calibration_config or CalibrationConfig()
        logistic_config = logistic_config or self._default_logistic_config()

        train_start, train_end = training_window
        val_start, val_end = validation_window
        test_start, test_end = test_window

        logger.info("Building training table")
        train_df = self._build_table(train_start, train_end)
        val_df = self._build_table(val_start, val_end)
        test_df = self._build_table(test_start, test_end)
        if train_df.empty:
            raise ValueError("No training examples found in the training window.")
        if val_df.empty:
            raise ValueError("No validation examples found in the validation window.")

        feature_cols = self._feature_columns(train_df)
        target_col = "home_advances"

        # 1. Pick the base learner.
        use_catboost, base_learner_name, _reason = self._select_base_learner(
            n_train=len(train_df), n_val=len(val_df)
        )
        feature_importance: list[tuple[str, float]] = []
        if use_catboost:
            base_learner = CatBoostModel(catboost_config)
            base_learner.fit(
                X_train=train_df[feature_cols],
                y_train=train_df[target_col].astype(int),
                X_val=val_df[feature_cols],
                y_val=val_df[target_col].astype(int),
            )
            base_learner_path = base_learner.save(
                self.registry.root / "catboost" / model_version
            )
            feature_importance = base_learner.feature_importance()
            base_learner_artifacts = {"catboost": str(base_learner_path)}
        else:
            base_learner = LogisticRegressionBaseline(logistic_config)
            base_learner.fit(
                X_train=train_df[feature_cols],
                y_train=train_df[target_col].astype(int),
                X_val=val_df[feature_cols],
                y_val=val_df[target_col].astype(int),
            )
            base_learner_path = base_learner.save(
                self.registry.root / "logistic_baseline" / model_version
            )
            feature_importance = base_learner.feature_importance()
            base_learner_artifacts = {"logistic_baseline": str(base_learner_path)}

        # 2. Build base predictions on the validation window.
        elo_model = EloModel(self.elo_config).fit(
            self._elo_training_data(training_window=(train_start, train_end))
        )
        market_model = MarketModel(self.session, min_bookmakers=self.market_min_bookmakers)

        val_elo, val_market, val_cat, val_y = self._predict_components(
            df=val_df, elo_model=elo_model, market_model=market_model, base_learner=base_learner
        )

        # 3. Fit the stacker on OOF predictions.
        stacker = StackingModel(stacking_config)
        # Replace None with NaN so the stacker fills them.
        market_array = np.array(
            [np.nan if v is None else float(v) for v in val_market], dtype=float
        )
        stacker.fit(market_array, val_elo, val_cat, val_y)
        stacker_path = stacker.save(self.registry.root / "stacking" / model_version)

        stacked_val = stacker.predict_proba(market_array, val_elo, val_cat)

        # 4. Fit the calibrator on validation.
        calibrator = Calibrator(calibration_config).fit(stacked_val, val_y)
        calibrator_path = calibrator.save(self.registry.root / "calibration" / model_version)

        # 5. Compute validation metrics for reporting.
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

        calibrated_val = calibrator.predict(stacked_val)
        val_metrics = {
            "log_loss_stacked": float(log_loss(val_y, np.clip(stacked_val, 1e-6, 1 - 1e-6))),
            "log_loss_calibrated": float(
                log_loss(val_y, np.clip(calibrated_val, 1e-6, 1 - 1e-6))
            ),
            "brier_stacked": float(brier_score_loss(val_y, stacked_val)),
            "brier_calibrated": float(brier_score_loss(val_y, calibrated_val)),
        }
        try:
            val_metrics["roc_auc_calibrated"] = float(roc_auc_score(val_y, calibrated_val))
        except ValueError:
            val_metrics["roc_auc_calibrated"] = None

        # 6. Register artifacts.
        if use_catboost:
            self.registry.register(
                model_type="catboost",
                model_version=model_version,
                artifact_path=base_learner_path,
                feature_version=self.feature_version,
                metrics=val_metrics,
                hyperparameters=asdict(catboost_config),
                feature_hash=stable_hash(feature_cols),
            )
        else:
            self.registry.register(
                model_type="logistic_baseline",
                model_version=model_version,
                artifact_path=base_learner_path,
                feature_version=self.feature_version,
                metrics=val_metrics,
                hyperparameters=asdict(logistic_config),
                feature_hash=stable_hash(feature_cols),
            )
        self.registry.register(
            model_type="stacking",
            model_version=model_version,
            artifact_path=stacker_path,
            feature_version=self.feature_version,
            metrics=val_metrics,
            hyperparameters=asdict(stacking_config),
            feature_hash=stable_hash(feature_cols),
        )
        self.registry.register(
            model_type="calibration",
            model_version=model_version,
            artifact_path=calibrator_path,
            feature_version=self.feature_version,
            metrics=val_metrics,
            hyperparameters=asdict(calibration_config),
            feature_hash=stable_hash(feature_cols),
        )

        # 7. Persist a model run record.
        from football_advance_predictor.db.models import ModelRun

        model_type_str = (
            "catboost_stacked_calibrated" if use_catboost else "logistic_stacked_calibrated"
        )
        model_run = ModelRun(
            model_run_id=f"run_{uuid.uuid4().hex[:16]}",
            model_type=model_type_str,
            model_version=model_version,
            training_start=train_start,
            training_end=train_end,
            validation_start=val_start,
            validation_end=val_end,
            test_start=test_start,
            test_end=test_end,
            feature_version=self.feature_version,
            hyperparameters_json={
                "base_learner": base_learner_name,
                "catboost": asdict(catboost_config),
                "logistic_regression": asdict(logistic_config),
                "stacking": asdict(stacking_config),
                "calibration": asdict(calibration_config),
            },
            metrics_json={
                "validation": val_metrics,
                "feature_importance_top10": feature_importance[:10],
            },
            artifact_path=str(base_learner_path),
        )
        self.session.add(model_run)
        self.session.flush()

        return {
            "model_run_id": model_run.model_run_id,
            "model_version": model_version,
            "base_learner": base_learner_name,
            "feature_count": len(feature_cols),
            "feature_importance": feature_importance,
            "validation_metrics": val_metrics,
            "artifact_paths": {
                **base_learner_artifacts,
                "stacking": str(stacker_path),
                "calibration": str(calibrator_path),
            },
            "train_size": len(train_df),
            "validation_size": len(val_df),
            "test_size": len(test_df),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_table(
        self, start: datetime, end: datetime
    ) -> pd.DataFrame:
        """Build a feature table for matches in the [start, end] window.

        Each row is a knockout match. Features are time-frozen at the
        match's T-24h cutoff by default.
        """
        start_utc = to_utc(start)
        end_utc = to_utc(end)
        stmt = (
            select(Match, MatchResult)
            .join(MatchResult, MatchResult.match_id == Match.match_id)
            .where(Match.kickoff_at >= start_utc)
            .where(Match.kickoff_at <= end_utc)
            .order_by(Match.kickoff_at)
        )
        rows = self.session.execute(stmt).all()
        builder = FeatureBuilder(
            session=self.session,
            elo_config=self.elo_config,
            feature_version=self.feature_version,
            market_min_bookmakers=self.market_min_bookmakers,
        )
        records: list[dict[str, Any]] = []
        for match, result in rows:
            cutoff = self._default_cutoff(match.kickoff_at)
            try:
                features, _ = builder.build(match_id=match.match_id, cutoff_time=cutoff)
            except ValueError as exc:
                logger.warning("Skipping match in table build", extra={"error": str(exc)})
                continue
            features["home_advances"] = int(result.home_advances)
            features["match_id"] = match.match_id
            features["cutoff_time"] = cutoff.isoformat()
            features["kickoff_at"] = match.kickoff_at.isoformat()
            records.append(features)
        return pd.DataFrame.from_records(records)

    def _default_cutoff(self, kickoff_at: datetime) -> datetime:
        kickoff_utc = to_utc(kickoff_at)
        return kickoff_utc - pd.Timedelta(hours=24).to_pytimedelta()

    def _elo_training_data(
        self, training_window: tuple[datetime, datetime] | None = None
    ) -> list[dict[str, Any]]:
        stmt = (
            select(Match, MatchResult, Competition)
            .join(MatchResult, MatchResult.match_id == Match.match_id)
            .outerjoin(Competition, Competition.competition_id == Match.competition_id)
            .order_by(Match.kickoff_at)
        )
        if training_window is not None:
            start, end = training_window
            stmt = stmt.where(Match.kickoff_at < to_utc(start))
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

    def _predict_components(
        self,
        *,
        df: pd.DataFrame,
        elo_model: EloModel,
        market_model: MarketModel,
        base_learner: CatBoostModel | LogisticRegressionBaseline,
    ) -> tuple[np.ndarray, list[float | None], np.ndarray, np.ndarray]:
        elo_probs: list[float] = []
        market_probs: list[float | None] = []
        base_probs: list[float] = []
        y: list[int] = []
        for _, row in df.iterrows():
            cutoff = to_utc(row["cutoff_time"])
            match_id = row["match_id"]
            match = self.session.get(Match, match_id)
            if match is None:
                continue
            elo_probs.append(
                elo_model.predict_proba(
                    home_team_id=match.home_team_id,
                    away_team_id=match.away_team_id,
                    as_of_time=cutoff,
                    neutral_venue=match.neutral_venue,
                )
            )
            market_probs.append(
                market_model.predict_proba(match_id=match_id, as_of_time=cutoff)
            )
            feature_row = pd.DataFrame([row.to_dict()])
            base_probs.append(float(base_learner.predict_proba(feature_row)[0]))
            y.append(int(row["home_advances"]))
        return (
            np.array(elo_probs, dtype=float),
            market_probs,
            np.array(base_probs, dtype=float),
            np.array(y, dtype=int),
        )

    def _select_base_learner(
        self, *, n_train: int, n_val: int
    ) -> tuple[bool, str, str]:
        """Decide whether to enable CatBoost for this training run.

        Returns:
            (use_catboost, base_learner_name, reason).
        """
        catboost_cfg = self.models_config.get("catboost", {}) if self.models_config else {}
        enabled = bool(catboost_cfg.get("enabled", False))
        min_samples = int(catboost_cfg.get("min_samples_to_enable", 200))
        if not enabled:
            return False, "logistic_regression", "catboost_disabled_in_config"
        if n_train < min_samples:
            logger.warning(
                "CatBoost requires more training samples than are available; falling back to logistic regression",
                extra={"n_train": n_train, "min_samples": min_samples},
            )
            return (
                False,
                "logistic_regression",
                f"n_train={n_train} < catboost.min_samples_to_enable={min_samples}",
            )
        if n_val < 1:
            return False, "logistic_regression", "no_validation_examples"
        return True, "catboost", "enabled"

    def _default_logistic_config(self) -> LogisticRegressionBaselineConfig:
        baseline_cfg = (
            self.models_config.get("baseline", {}).get("logistic_regression", {})
            if self.models_config
            else {}
        )
        return LogisticRegressionBaselineConfig.from_dict(baseline_cfg)

    @staticmethod
    def _feature_columns(df: pd.DataFrame) -> list[str]:
        excluded = {"home_advances", "match_id", "cutoff_time", "kickoff_at"}
        return [c for c in df.columns if c not in excluded]
