"""initial schema

Revision ID: 0001_init
Revises:
Create Date: 2026-06-29 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "teams",
        sa.Column("team_id", sa.String(64), primary_key=True),
        sa.Column("canonical_name", sa.String(128), nullable=False),
        sa.Column("fifa_code", sa.String(8), nullable=True),
        sa.Column("confederation", sa.String(32), nullable=True),
        sa.Column("active_from", sa.DateTime(timezone=True), nullable=True),
        sa.Column("active_to", sa.DateTime(timezone=True), nullable=True),
        sa.Column("aliases", postgresql.JSON, nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("fifa_code", name="uq_teams_fifa_code"),
    )
    op.create_index("ix_teams_canonical_name", "teams", ["canonical_name"])
    op.create_index("ix_teams_fifa_code", "teams", ["fifa_code"])

    op.create_table(
        "competitions",
        sa.Column("competition_id", sa.String(64), primary_key=True),
        sa.Column("canonical_name", sa.String(128), nullable=False),
        sa.Column("competition_type", sa.String(32), nullable=False),
        sa.Column("importance_weight", sa.Float, nullable=False, server_default="1.0"),
        sa.Column("is_knockout_capable", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_competitions_canonical_name", "competitions", ["canonical_name"])

    op.create_table(
        "matches",
        sa.Column("match_id", sa.String(64), primary_key=True),
        sa.Column("kickoff_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("competition_id", sa.String(64), sa.ForeignKey("competitions.competition_id"), nullable=False),
        sa.Column("stage", sa.String(64), nullable=False, server_default="unknown"),
        sa.Column("season_or_year", sa.String(16), nullable=False),
        sa.Column("home_team_id", sa.String(64), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("away_team_id", sa.String(64), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("home_goals", sa.Integer, nullable=True),
        sa.Column("away_goals", sa.Integer, nullable=True),
        sa.Column("winner_team_id", sa.String(64), sa.ForeignKey("teams.team_id"), nullable=True),
        sa.Column("advancing_team_id", sa.String(64), sa.ForeignKey("teams.team_id"), nullable=True),
        sa.Column("neutral_venue", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("venue_name", sa.String(128), nullable=True),
        sa.Column("city", sa.String(128), nullable=True),
        sa.Column("country", sa.String(64), nullable=True),
        sa.Column("source", sa.String(64), nullable=False, server_default="local"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_matches_kickoff_at", "matches", ["kickoff_at"])
    op.create_index("ix_matches_competition_id", "matches", ["competition_id"])
    op.create_index("ix_matches_home_team_id", "matches", ["home_team_id"])
    op.create_index("ix_matches_away_team_id", "matches", ["away_team_id"])
    op.create_index("ix_matches_kickoff_competition", "matches", ["kickoff_at", "competition_id"])

    op.create_table(
        "match_results",
        sa.Column("match_id", sa.String(64), sa.ForeignKey("matches.match_id"), primary_key=True),
        sa.Column("final_status", sa.String(16), nullable=False, server_default="unknown"),
        sa.Column("home_goals_90", sa.Integer, nullable=True),
        sa.Column("away_goals_90", sa.Integer, nullable=True),
        sa.Column("home_goals_et", sa.Integer, nullable=True),
        sa.Column("away_goals_et", sa.Integer, nullable=True),
        sa.Column("penalties_home", sa.Integer, nullable=True),
        sa.Column("penalties_away", sa.Integer, nullable=True),
        sa.Column("home_advances", sa.Boolean, nullable=False),
        sa.Column("result_verified_at", sa.DateTime(timezone=True), nullable=False),
    )

    op.create_table(
        "market_odds_snapshots",
        sa.Column("odds_snapshot_id", sa.String(64), primary_key=True),
        sa.Column("match_id", sa.String(64), sa.ForeignKey("matches.match_id"), nullable=False),
        sa.Column("bookmaker", sa.String(64), nullable=False),
        sa.Column("market_type", sa.String(32), nullable=False),
        sa.Column("selection", sa.String(32), nullable=False),
        sa.Column("decimal_odds", sa.Float, nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(64), nullable=False, server_default="local"),
        sa.Column("is_live", sa.Boolean, nullable=False, server_default=sa.text("false")),
        sa.Column("currency_or_region", sa.String(16), nullable=True),
        sa.Column("source_name", sa.String(64), nullable=False, server_default="local"),
        sa.Column("source_record_id", sa.String(128), nullable=True),
        sa.Column("source_url", sa.String(512), nullable=True),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload_hash", sa.String(64), nullable=False),
        sa.Column("source_version", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_market_odds_match_id", "market_odds_snapshots", ["match_id"])
    op.create_index("ix_market_odds_captured_at", "market_odds_snapshots", ["captured_at"])
    op.create_index("ix_odds_match_market_captured", "market_odds_snapshots", ["match_id", "market_type", "selection", "captured_at"])

    op.create_table(
        "player_availability_snapshots",
        sa.Column("availability_id", sa.String(64), primary_key=True),
        sa.Column("match_id", sa.String(64), sa.ForeignKey("matches.match_id"), nullable=False),
        sa.Column("team_id", sa.String(64), sa.ForeignKey("teams.team_id"), nullable=False),
        sa.Column("player_id", sa.String(64), nullable=True),
        sa.Column("role", sa.String(32), nullable=False, server_default="unknown"),
        sa.Column("availability_status", sa.String(32), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False, server_default="0.5"),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("cutoff_eligible", sa.Boolean, nullable=False, server_default=sa.text("true")),
        sa.Column("source", sa.String(64), nullable=False, server_default="local"),
        sa.Column("raw_text", sa.Text, nullable=True),
        sa.Column("source_name", sa.String(64), nullable=False, server_default="local"),
        sa.Column("source_record_id", sa.String(128), nullable=True),
        sa.Column("source_url", sa.String(512), nullable=True),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("raw_payload_hash", sa.String(64), nullable=False),
        sa.Column("source_version", sa.String(32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_player_availability_match_id", "player_availability_snapshots", ["match_id"])

    op.create_table(
        "feature_snapshots",
        sa.Column("feature_snapshot_id", sa.String(64), primary_key=True),
        sa.Column("match_id", sa.String(64), sa.ForeignKey("matches.match_id"), nullable=False),
        sa.Column("cutoff_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_version", sa.String(32), nullable=False),
        sa.Column("features_json", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("source_data_max_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("immutable_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("match_id", "cutoff_time", "feature_version", name="uq_snapshot_match_cutoff_version"),
    )
    op.create_index("ix_feature_snapshots_match_id", "feature_snapshots", ["match_id"])
    op.create_index("ix_feature_snapshots_immutable_hash", "feature_snapshots", ["immutable_hash"])

    op.create_table(
        "predictions",
        sa.Column("prediction_id", sa.String(64), primary_key=True),
        sa.Column("match_id", sa.String(64), sa.ForeignKey("matches.match_id"), nullable=False),
        sa.Column("cutoff_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("feature_snapshot_id", sa.String(64), sa.ForeignKey("feature_snapshots.feature_snapshot_id"), nullable=True),
        sa.Column("market_probability", sa.Float, nullable=True),
        sa.Column("elo_probability", sa.Float, nullable=True),
        sa.Column("catboost_probability", sa.Float, nullable=True),
        sa.Column("stacked_probability", sa.Float, nullable=True),
        sa.Column("calibrated_probability", sa.Float, nullable=False),
        sa.Column("home_advance_probability", sa.Float, nullable=False),
        sa.Column("away_advance_probability", sa.Float, nullable=False),
        sa.Column("predicted_advancer_id", sa.String(64), nullable=False),
        sa.Column("confidence_band", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="active"),
        sa.Column("explanation_payload", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("immutable_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.UniqueConstraint("match_id", "cutoff_time", "model_version", name="uq_prediction_match_cutoff_model"),
    )
    op.create_index("ix_predictions_match_id", "predictions", ["match_id"])
    op.create_index("ix_predictions_cutoff_time", "predictions", ["cutoff_time"])
    op.create_index("ix_predictions_model_version", "predictions", ["model_version"])
    op.create_index("ix_predictions_immutable_hash", "predictions", ["immutable_hash"])

    op.create_table(
        "evaluation_records",
        sa.Column("evaluation_id", sa.String(64), primary_key=True),
        sa.Column("prediction_id", sa.String(64), sa.ForeignKey("predictions.prediction_id"), nullable=False),
        sa.Column("actual_home_advances", sa.Boolean, nullable=False),
        sa.Column("log_loss", sa.Float, nullable=False),
        sa.Column("brier_score", sa.Float, nullable=False),
        sa.Column("correct_classification", sa.Boolean, nullable=False),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_evaluation_records_prediction_id", "evaluation_records", ["prediction_id"])

    op.create_table(
        "model_runs",
        sa.Column("model_run_id", sa.String(64), primary_key=True),
        sa.Column("model_type", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(32), nullable=False),
        sa.Column("training_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("training_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validation_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("validation_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("test_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("test_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("feature_version", sa.String(32), nullable=False),
        sa.Column("hyperparameters_json", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("metrics_json", postgresql.JSON, nullable=False, server_default="{}"),
        sa.Column("artifact_path", sa.String(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
    )
    op.create_index("ix_model_runs_model_version", "model_runs", ["model_version"])


def downgrade() -> None:
    op.drop_table("model_runs")
    op.drop_table("evaluation_records")
    op.drop_table("predictions")
    op.drop_table("feature_snapshots")
    op.drop_table("player_availability_snapshots")
    op.drop_table("market_odds_snapshots")
    op.drop_table("match_results")
    op.drop_table("matches")
    op.drop_table("competitions")
    op.drop_table("teams")
