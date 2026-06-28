"""End-to-end prediction service.

The service glues together:

- feature snapshot building,
- base model predictions (Elo, market, CatBoost),
- stacking,
- calibration,
- ledger write.

It never modifies a previously stored prediction; it creates new ones.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.data.snapshots.snapshot_service import FeatureSnapshotService
from football_advance_predictor.db.models import Match
from football_advance_predictor.features.elo.elo_engine import EloConfig
from football_advance_predictor.ledger.ledger_service import LedgerService
from football_advance_predictor.models.calibration.calibrator import Calibrator
from football_advance_predictor.models.catboost_model.catboost_model import CatBoostModel
from football_advance_predictor.models.elo_model.elo_model import EloModel
from football_advance_predictor.models.market_model.market_model import MarketModel
from football_advance_predictor.models.stacking.stacker import StackingModel

logger = get_logger(__name__)


class PredictionService:
    """Generate immutable predictions for a match at a given cutoff."""

    def __init__(
        self,
        session: Session,
        *,
        elo_config: EloConfig | None = None,
        catboost_model: CatBoostModel | None = None,
        stacker: StackingModel | None = None,
        calibrator: Calibrator | None = None,
        market_min_bookmakers: int = 1,
    ) -> None:
        self.session = session
        self.elo_config = elo_config or EloConfig()
        self.catboost_model = catboost_model
        self.stacker = stacker
        self.calibrator = calibrator
        self.market_min_bookmakers = market_min_bookmakers
        self.ledger = LedgerService(session)
        self.snapshot_service = FeatureSnapshotService(session)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        *,
        match_id: str,
        cutoff_time: datetime,
        model_version: str,
        feature_version: str = "v1",
        clear_lean_min: float = 0.62,
        slight_lean_min: float = 0.55,
    ) -> dict[str, Any]:
        """Build a feature snapshot, run all base models, and persist a prediction.

        Returns:
            A dict with prediction_id, probabilities, and explanation.
        """
        cutoff = to_utc(cutoff_time)
        match = self.session.get(Match, match_id)
        if match is None:
            raise ValueError(f"Match not found: {match_id}")
        if cutoff >= to_utc(match.kickoff_at):
            raise ValueError(
                f"Cutoff {cutoff.isoformat()} must be before kickoff "
                f"{match.kickoff_at.isoformat()}."
            )

        # 1. Build or get a feature snapshot.
        snapshot = self.snapshot_service.build_or_get(
            match_id=match_id, cutoff_time=cutoff, feature_version=feature_version
        )

        # 2. Run base models.
        elo_model = EloModel(self.elo_config)
        # Cold-start the engine with all known match results.
        elo_model.fit(self._elo_training_data())

        elo_prob = elo_model.predict_proba(
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            as_of_time=cutoff,
            neutral_venue=match.neutral_venue,
        )

        market_model = MarketModel(self.session, min_bookmakers=self.market_min_bookmakers)
        market_prob = market_model.predict_proba(match_id=match_id, as_of_time=cutoff)
        market_consensus = market_model.consensus_at(match_id=match_id, as_of_time=cutoff)

        # 3. CatBoost: requires a trained model.
        if self.catboost_model is not None:
            feature_row = self._build_inference_row(snapshot.features_json)
            catboost_prob = float(self.catboost_model.predict_proba(feature_row)[0])
        else:
            catboost_prob = 0.5
            logger.warning(
                "No CatBoost model loaded; using neutral 0.5 for catboost_probability"
            )

        # 4. Stack.
        if self.stacker is not None:
            stacked = float(
                self.stacker.predict_proba(
                    market=np.array([market_prob if market_prob is not None else 0.5]),
                    elo=np.array([elo_prob]),
                    catboost=np.array([catboost_prob]),
                )[0]
            )
        else:
            # Default: weighted blend favoring market.
            stacked = _blend(market_prob, elo_prob, catboost_prob)

        # 5. Calibrate.
        if self.calibrator is not None:
            calibrated = float(self.calibrator.predict(np.array([stacked]))[0])
        else:
            calibrated = float(np.clip(stacked, 1e-6, 1.0 - 1e-6))

        # 6. Build explanation payload (factual, bounded).
        explanation = self._build_explanation(
            snapshot_features=snapshot.features_json,
            market_prob=market_prob,
            elo_prob=elo_prob,
            catboost_prob=catboost_prob,
            market_diagnostics=market_consensus.diagnostics if market_consensus else None,
        )

        prediction = self.ledger.create_prediction(
            match_id=match_id,
            cutoff_time=cutoff,
            model_version=model_version,
            home_team_id=match.home_team_id,
            away_team_id=match.away_team_id,
            market_probability=market_prob,
            elo_probability=elo_prob,
            catboost_probability=catboost_prob,
            stacked_probability=stacked,
            calibrated_probability=calibrated,
            feature_snapshot_id=snapshot.feature_snapshot_id,
            explanation_payload=explanation,
            clear_lean_min=clear_lean_min,
            slight_lean_min=slight_lean_min,
        )
        return {
            "prediction_id": prediction.prediction_id,
            "match_id": match_id,
            "cutoff_time": cutoff.isoformat(),
            "model_version": model_version,
            "home_advance_probability": prediction.home_advance_probability,
            "away_advance_probability": prediction.away_advance_probability,
            "predicted_advancer_id": prediction.predicted_advancer_id,
            "confidence_band": prediction.confidence_band,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _elo_training_data(self) -> list[dict[str, Any]]:
        from football_advance_predictor.db.models import Competition, Match, MatchResult

        stmt = (
            select(Match, MatchResult, Competition)
            .join(MatchResult, MatchResult.match_id == Match.match_id)
            .outerjoin(Competition, Competition.competition_id == Match.competition_id)
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

    @staticmethod
    def _build_inference_row(features: dict[str, Any]) -> Any:
        import pandas as pd

        return pd.DataFrame([features])

    def _build_explanation(
        self,
        *,
        snapshot_features: dict[str, Any],
        market_prob: float | None,
        elo_prob: float,
        catboost_prob: float,
        market_diagnostics: Any,
    ) -> dict[str, Any]:
        positive: list[dict[str, str]] = []
        negative: list[dict[str, str]] = []

        if snapshot_features.get("elo_difference", 0.0) > 5:
            positive.append({"feature": "elo_difference", "effect": "favors_home"})
        elif snapshot_features.get("elo_difference", 0.0) < -5:
            negative.append({"feature": "elo_difference", "effect": "favors_away"})

        if market_prob is not None and market_prob > 0.55:
            positive.append({"feature": "market_advance_probability", "effect": "favors_home"})
        elif market_prob is not None and market_prob < 0.45:
            negative.append({"feature": "market_advance_probability", "effect": "favors_away"})

        confirmed_out_diff = snapshot_features.get("confirmed_out_midfielders_diff", 0.0)
        if confirmed_out_diff and confirmed_out_diff > 0:
            negative.append({"feature": "confirmed_out_midfielders", "effect": "favors_away"})
        elif confirmed_out_diff and confirmed_out_diff < 0:
            positive.append({"feature": "confirmed_out_midfielders", "effect": "favors_home"})

        return {
            "top_positive_factors": positive,
            "top_negative_factors": negative,
            "data_completeness": {
                "market_available": market_prob is not None,
                "lineups_confirmed": bool(snapshot_features.get("lineup_confirmed", 0.0)),
                "xg_features_available": False,
            },
            "elo_probability": elo_prob,
            "catboost_probability": catboost_prob,
            "market_bookmaker_count": (
                market_diagnostics.bookmaker_count if market_diagnostics else 0
            ),
            "market_overround": market_diagnostics.overround if market_diagnostics else None,
            "market_dispersion": market_diagnostics.dispersion if market_diagnostics else None,
        }


def _blend(market_prob: float | None, elo_prob: float, catboost_prob: float) -> float:
    """Simple weighted blend fallback for when no stacker is available.

    Market is weighted most heavily when available; otherwise the blend
    rebalances.
    """
    if market_prob is None:
        return 0.5 * elo_prob + 0.5 * catboost_prob
    return 0.5 * market_prob + 0.1 * elo_prob + 0.4 * catboost_prob
