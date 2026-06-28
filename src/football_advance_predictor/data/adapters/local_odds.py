"""Local market odds provider.

Reads a CSV with timestamped odds snapshots. Each row is treated as a
standalone snapshot with its own lineage. ``captured_at`` is the time
the odds were observed.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from football_advance_predictor.core.hashing import stable_hash
from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.data.adapters.base import OddsProvider
from football_advance_predictor.schemas.odds import MarketOddsIn

logger = get_logger(__name__)


class LocalOddsProvider(OddsProvider):
    """Read market odds from a local CSV file."""

    name = "local_odds"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Odds CSV not found: {self.path}")

    def fetch_odds(self, **kwargs: Any) -> list[MarketOddsIn]:
        with self.path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [{k.lower().strip(): v for k, v in row.items() if k} for row in reader]
        out: list[MarketOddsIn] = []
        for idx, row in enumerate(rows):
            try:
                out.append(self._row_to_odds(row))
            except Exception as exc:
                logger.warning("Failed to parse odds row", extra={"row_index": idx, "error": str(exc)})
        return out

    def _row_to_odds(self, row: dict[str, str]) -> MarketOddsIn:
        captured_at = to_utc(row["captured_at"])
        ingested_at = to_utc(row.get("ingested_at") or captured_at.isoformat())
        effective_at = to_utc(row.get("effective_at") or captured_at.isoformat())
        decimal_odds = float(row["decimal_odds"])
        raw = {k: v for k, v in row.items()}
        raw_payload_hash = row.get("raw_payload_hash") or stable_hash(raw)
        return MarketOddsIn(
            match_id=row["match_id"].strip(),
            bookmaker=row.get("bookmaker", "unknown").strip(),
            market_type=row["market_type"].strip(),
            selection=row["selection"].strip().lower(),
            decimal_odds=decimal_odds,
            captured_at=captured_at,
            source=row.get("source", "local"),
            is_live=(row.get("is_live", "false").strip().lower() in {"1", "true", "t", "yes"}),
            currency_or_region=row.get("currency_or_region"),
            source_name=row.get("source_name", "local"),
            source_record_id=row.get("source_record_id"),
            source_url=row.get("source_url"),
            published_at=to_utc(row["published_at"]) if row.get("published_at") else None,
            ingested_at=ingested_at,
            effective_at=effective_at,
            raw_payload_hash=raw_payload_hash,
            source_version=row.get("source_version"),
        )
