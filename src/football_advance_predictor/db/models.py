"""SQLAlchemy ORM models for the application database.

Every table includes explicit lineage columns so we can audit where a
record came from and when it was observed. The anti-leakage contract
relies on these columns.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from football_advance_predictor.db.base import Base, TimestampMixin

# ---------------------------------------------------------------------------
# Teams, competitions, matches
# ---------------------------------------------------------------------------


class Team(Base, TimestampMixin):
    __tablename__ = "teams"

    team_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    fifa_code: Mapped[str | None] = mapped_column(String(8), nullable=True, index=True)
    confederation: Mapped[str | None] = mapped_column(String(32), nullable=True)
    active_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    active_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)

    __table_args__ = (UniqueConstraint("fifa_code", name="uq_teams_fifa_code"),)


class Competition(Base, TimestampMixin):
    __tablename__ = "competitions"

    competition_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    competition_type: Mapped[str] = mapped_column(String(32), nullable=False)
    importance_weight: Mapped[float] = mapped_column(Float, default=1.0, nullable=False)
    is_knockout_capable: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Match(Base, TimestampMixin):
    __tablename__ = "matches"

    match_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kickoff_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    competition_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("competitions.competition_id"), nullable=False, index=True
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False, default="unknown")
    season_or_year: Mapped[str] = mapped_column(String(16), nullable=False)
    home_team_id: Mapped[str] = mapped_column(String(64), ForeignKey("teams.team_id"), nullable=False, index=True)
    away_team_id: Mapped[str] = mapped_column(String(64), ForeignKey("teams.team_id"), nullable=False, index=True)
    home_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    winner_team_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("teams.team_id"), nullable=True
    )
    advancing_team_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("teams.team_id"), nullable=True
    )
    neutral_venue: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    venue_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    city: Mapped[str | None] = mapped_column(String(128), nullable=True)
    country: Mapped[str | None] = mapped_column(String(64), nullable=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="local")

    __table_args__ = (
        Index("ix_matches_kickoff_competition", "kickoff_at", "competition_id"),
    )


class MatchResult(Base):
    __tablename__ = "match_results"

    match_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("matches.match_id"), primary_key=True
    )
    final_status: Mapped[str] = mapped_column(String(16), nullable=False, default="unknown")
    home_goals_90: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals_90: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_goals_et: Mapped[int | None] = mapped_column(Integer, nullable=True)
    away_goals_et: Mapped[int | None] = mapped_column(Integer, nullable=True)
    penalties_home: Mapped[int | None] = mapped_column(Integer, nullable=True)
    penalties_away: Mapped[int | None] = mapped_column(Integer, nullable=True)
    home_advances: Mapped[bool] = mapped_column(Boolean, nullable=False)
    result_verified_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


# ---------------------------------------------------------------------------
# Market odds snapshots
# ---------------------------------------------------------------------------


class MarketOddsSnapshot(Base, TimestampMixin):
    __tablename__ = "market_odds_snapshots"

    odds_snapshot_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    match_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("matches.match_id"), nullable=False, index=True
    )
    bookmaker: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    selection: Mapped[str] = mapped_column(String(32), nullable=False)
    decimal_odds: Mapped[float] = mapped_column(Float, nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="local")
    is_live: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    currency_or_region: Mapped[str | None] = mapped_column(String(16), nullable=True)

    # Lineage columns
    source_name: Mapped[str] = mapped_column(String(64), nullable=False, default="local")
    source_record_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    effective_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    raw_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        Index(
            "ix_odds_match_market_captured",
            "match_id",
            "market_type",
            "selection",
            "captured_at",
        ),
        UniqueConstraint("raw_payload_hash", name="uq_market_odds_raw_payload_hash"),
    )


# ---------------------------------------------------------------------------
# Player availability snapshots
# ---------------------------------------------------------------------------


class PlayerAvailabilitySnapshot(Base, TimestampMixin):
    __tablename__ = "player_availability_snapshots"

    availability_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    match_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("matches.match_id"), nullable=False, index=True
    )
    team_id: Mapped[str] = mapped_column(String(64), ForeignKey("teams.team_id"), nullable=False)
    player_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    availability_status: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    cutoff_eligible: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False, default="local")
    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Lineage
    source_name: Mapped[str] = mapped_column(String(64), nullable=False, default="local")
    source_record_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_version: Mapped[str | None] = mapped_column(String(32), nullable=True)

    __table_args__ = (
        UniqueConstraint("raw_payload_hash", name="uq_availability_raw_payload_hash"),
    )


# ---------------------------------------------------------------------------
# Feature snapshots (immutable)
# ---------------------------------------------------------------------------


class FeatureSnapshot(Base, TimestampMixin):
    __tablename__ = "feature_snapshots"

    feature_snapshot_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    match_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("matches.match_id"), nullable=False, index=True
    )
    cutoff_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    features_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    source_data_max_timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    immutable_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint(
            "match_id", "cutoff_time", "feature_version", name="uq_snapshot_match_cutoff_version"
        ),
    )


# ---------------------------------------------------------------------------
# Predictions ledger (immutable)
# ---------------------------------------------------------------------------


class Prediction(Base, TimestampMixin):
    __tablename__ = "predictions"

    prediction_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    match_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("matches.match_id"), nullable=False, index=True
    )
    cutoff_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    feature_snapshot_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("feature_snapshots.feature_snapshot_id"), nullable=True
    )

    market_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    elo_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    catboost_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    stacked_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    calibrated_probability: Mapped[float] = mapped_column(Float, nullable=False)

    home_advance_probability: Mapped[float] = mapped_column(Float, nullable=False)
    away_advance_probability: Mapped[float] = mapped_column(Float, nullable=False)
    predicted_advancer_id: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence_band: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")

    # Immutable explanation payload
    explanation_payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    immutable_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    __table_args__ = (
        UniqueConstraint(
            "match_id", "cutoff_time", "model_version", name="uq_prediction_match_cutoff_model"
        ),
    )


# ---------------------------------------------------------------------------
# Evaluations and model runs
# ---------------------------------------------------------------------------


class EvaluationRecord(Base, TimestampMixin):
    __tablename__ = "evaluation_records"

    evaluation_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    prediction_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("predictions.prediction_id"), nullable=False, index=True
    )
    actual_home_advances: Mapped[bool] = mapped_column(Boolean, nullable=False)
    log_loss: Mapped[float] = mapped_column(Float, nullable=False)
    brier_score: Mapped[float] = mapped_column(Float, nullable=False)
    correct_classification: Mapped[bool] = mapped_column(Boolean, nullable=False)
    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )


class ModelRun(Base, TimestampMixin):
    __tablename__ = "model_runs"

    model_run_id: Mapped[str] = mapped_column(
        String(64), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    training_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    training_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    validation_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    validation_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    test_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    test_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(32), nullable=False)
    hyperparameters_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    metrics_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    artifact_path: Mapped[str] = mapped_column(String(512), nullable=False)
