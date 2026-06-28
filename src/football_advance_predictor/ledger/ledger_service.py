"""Immutable prediction ledger.

Once a prediction is written, it is never updated. Evaluation
records reference the prediction and never modify it. This guarantees
the ledger can answer "what did model v0.1 say at T-24h?" deterministically.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from football_advance_predictor.core.hashing import stable_hash
from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.db.models import EvaluationRecord, Prediction
from football_advance_predictor.schemas.predictions import assign_confidence_band

logger = get_logger(__name__)


class LedgerService:
    """CRUD-style service for the immutable prediction ledger."""

    def __init__(self, session: Session) -> None:
        self.session = session

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def create_prediction(
        self,
        *,
        match_id: str,
        cutoff_time: datetime,
        model_version: str,
        home_team_id: str,
        away_team_id: str,
        market_probability: float | None,
        elo_probability: float | None,
        catboost_probability: float | None,
        stacked_probability: float | None,
        calibrated_probability: float,
        feature_snapshot_id: str | None,
        explanation_payload: dict[str, Any],
        clear_lean_min: float = 0.62,
        slight_lean_min: float = 0.55,
    ) -> Prediction:
        """Create a new prediction. Raises if a duplicate exists.

        The (match_id, cutoff_time, model_version) tuple is unique.
        """
        cutoff = to_utc(cutoff_time)
        existing = self.session.scalar(
            select(Prediction).where(
                Prediction.match_id == match_id,
                Prediction.cutoff_time == cutoff,
                Prediction.model_version == model_version,
            )
        )
        if existing is not None:
            raise ValueError(
                f"Prediction already exists for match={match_id} cutoff={cutoff.isoformat()} "
                f"model={model_version} (prediction_id={existing.prediction_id})"
            )
        band = assign_confidence_band(
            calibrated_probability, clear_lean_min=clear_lean_min, slight_lean_min=slight_lean_min
        )
        advancer = home_team_id if calibrated_probability >= 0.5 else away_team_id
        immutable_hash = stable_hash(
            {
                "match_id": match_id,
                "cutoff_time": cutoff.isoformat(),
                "model_version": model_version,
                "calibrated_probability": calibrated_probability,
                "advancer": advancer,
                "band": band.value,
            }
        )
        prediction = Prediction(
            prediction_id=f"pred_{uuid.uuid4().hex[:16]}",
            match_id=match_id,
            cutoff_time=cutoff,
            model_version=model_version,
            feature_snapshot_id=feature_snapshot_id,
            market_probability=market_probability,
            elo_probability=elo_probability,
            catboost_probability=catboost_probability,
            stacked_probability=stacked_probability,
            calibrated_probability=calibrated_probability,
            home_advance_probability=calibrated_probability,
            away_advance_probability=1.0 - calibrated_probability,
            predicted_advancer_id=advancer,
            confidence_band=band.value,
            status="active",
            explanation_payload=explanation_payload,
            immutable_hash=immutable_hash,
        )
        self.session.add(prediction)
        self.session.flush()
        logger.info(
            "Created prediction",
            extra={
                "prediction_id": prediction.prediction_id,
                "match_id": match_id,
                "model_version": model_version,
            },
        )
        return prediction

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_prediction(self, prediction_id: str) -> Prediction | None:
        return self.session.get(Prediction, prediction_id)

    def list_predictions(
        self,
        *,
        match_id: str | None = None,
        model_version: str | None = None,
        since: datetime | None = None,
    ) -> list[Prediction]:
        stmt = select(Prediction).order_by(Prediction.created_at.desc())
        if match_id is not None:
            stmt = stmt.where(Prediction.match_id == match_id)
        if model_version is not None:
            stmt = stmt.where(Prediction.model_version == model_version)
        if since is not None:
            stmt = stmt.where(Prediction.cutoff_time >= to_utc(since))
        return list(self.session.scalars(stmt))

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate_prediction(
        self, prediction_id: str, actual_home_advances: bool
    ) -> EvaluationRecord:
        """Compute log loss + brier for a prediction. Immutable.

        The prediction itself is never modified. A new evaluation row is
        appended per call.
        """
        prediction = self.session.get(Prediction, prediction_id)
        if prediction is None:
            raise ValueError(f"Prediction not found: {prediction_id}")
        p = _clip(prediction.calibrated_probability)
        y = 1.0 if actual_home_advances else 0.0
        log_loss = -(y * _safe_log(p) + (1.0 - y) * _safe_log(1.0 - p))
        brier = (p - y) ** 2
        record = EvaluationRecord(
            evaluation_id=f"eval_{uuid.uuid4().hex[:16]}",
            prediction_id=prediction_id,
            actual_home_advances=actual_home_advances,
            log_loss=float(log_loss),
            brier_score=float(brier),
            correct_classification=_is_correct(prediction, actual_home_advances),
            evaluated_at=to_utc(datetime.now(tz=UTC)),
        )
        self.session.add(record)
        self.session.flush()
        return record

    def evaluate_match(self, match_id: str, actual_home_advances: bool) -> list[EvaluationRecord]:
        """Evaluate every prediction stored for ``match_id``."""
        predictions = self.list_predictions(match_id=match_id)
        return [self.evaluate_prediction(p.prediction_id, actual_home_advances) for p in predictions]

    # ------------------------------------------------------------------
    # Compare model versions
    # ------------------------------------------------------------------

    def compare_model_versions(
        self, model_versions: Iterable[str]
    ) -> list[dict[str, Any]]:
        """Aggregate per-model metrics across all stored predictions.

        Returns a list of dicts with mean log loss, brier, accuracy,
        and number of evaluated predictions. Predictions without an
        evaluation record are skipped.
        """
        results: list[dict[str, Any]] = []
        for version in model_versions:
            stmt = (
                select(Prediction, EvaluationRecord)
                .join(EvaluationRecord, EvaluationRecord.prediction_id == Prediction.prediction_id)
                .where(Prediction.model_version == version)
            )
            rows = self.session.execute(stmt).all()
            if not rows:
                results.append(
                    {
                        "model_version": version,
                        "n_predictions": 0,
                        "mean_log_loss": None,
                        "mean_brier": None,
                        "accuracy": None,
                    }
                )
                continue
            log_losses = [r.EvaluationRecord.log_loss for r in rows]
            briers = [r.EvaluationRecord.brier_score for r in rows]
            accuracies = [r.EvaluationRecord.correct_classification for r in rows]
            results.append(
                {
                    "model_version": version,
                    "n_predictions": len(rows),
                    "mean_log_loss": sum(log_losses) / len(log_losses),
                    "mean_brier": sum(briers) / len(briers),
                    "accuracy": sum(accuracies) / len(accuracies),
                }
            )
        return results

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    def export_csv(self, output: str | Path, *, match_id: str | None = None) -> None:
        rows = self._as_rows(self.list_predictions(match_id=match_id))
        df = pd.DataFrame(rows)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        logger.info("Exported ledger to CSV", extra={"output": str(output), "n": len(df)})

    def export_parquet(self, output: str | Path, *, match_id: str | None = None) -> None:
        rows = self._as_rows(self.list_predictions(match_id=match_id))
        df = pd.DataFrame(rows)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output, index=False)
        logger.info("Exported ledger to Parquet", extra={"output": str(output), "n": len(df)})

    @staticmethod
    def _as_rows(predictions: Iterable[Prediction]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in predictions:
            out.append(
                {
                    "prediction_id": p.prediction_id,
                    "match_id": p.match_id,
                    "cutoff_time": p.cutoff_time,
                    "model_version": p.model_version,
                    "home_advance_probability": p.home_advance_probability,
                    "away_advance_probability": p.away_advance_probability,
                    "predicted_advancer_id": p.predicted_advancer_id,
                    "confidence_band": p.confidence_band,
                    "market_probability": p.market_probability,
                    "elo_probability": p.elo_probability,
                    "catboost_probability": p.catboost_probability,
                    "stacked_probability": p.stacked_probability,
                    "calibrated_probability": p.calibrated_probability,
                    "immutable_hash": p.immutable_hash,
                    "created_at": p.created_at,
                }
            )
        return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clip(prob: float, lo: float = 1e-6, hi: float = 1.0 - 1e-6) -> float:
    return max(lo, min(hi, prob))


def _safe_log(p: float) -> float:
    import math

    return math.log(max(p, 1e-12))


def _is_correct(prediction: Prediction, actual_home_advances: bool) -> bool:
    """Return True iff the predicted advancer matches the actual outcome.

    The ``Prediction.predicted_advancer_id`` is the home team iff the
    calibrated ``home_advance_probability`` is >= 0.5. Comparing that
    threshold to the actual outcome gives us correctness without
    needing the home-team id at evaluation time.
    """
    predicted_p_home = prediction.home_advance_probability
    if actual_home_advances:
        return predicted_p_home >= 0.5
    return predicted_p_home < 0.5
