"""Pydantic schemas for market odds snapshots."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MarketOddsIn(BaseModel):
    """Input schema for a market odds snapshot.

    The schema requires explicit lineage columns. ``captured_at`` is the
    time the odds were observed (must be strictly before ``effective_at``).
    """

    model_config = ConfigDict(extra="forbid")

    odds_snapshot_id: str | None = None
    match_id: str
    bookmaker: str
    market_type: str = Field(
        ...,
        description=(
            "One of: moneyline_90, draw_90, away_90, home_to_advance, away_to_advance"
        ),
    )
    selection: str = Field(..., description="Selection label, e.g. 'home', 'away', 'draw'")
    decimal_odds: float = Field(..., gt=1.0)
    captured_at: datetime
    source: str = "local"
    is_live: bool = False
    currency_or_region: str | None = None

    # Lineage
    source_name: str = "local"
    source_record_id: str | None = None
    source_url: str | None = None
    published_at: datetime | None = None
    ingested_at: datetime
    effective_at: datetime
    raw_payload_hash: str
    source_version: str | None = None


class MarketOddsOut(MarketOddsIn):
    """Output schema for an odds snapshot."""

    model_config = ConfigDict(from_attributes=True)
    odds_snapshot_id: str
