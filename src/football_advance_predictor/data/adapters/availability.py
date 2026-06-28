"""Local availability / lineup provider.

Reads a JSON file where each record describes an availability event
(injury news, lineup confirmation, suspension, etc.). Each record
includes explicit ``observed_at`` and ``published_at`` timestamps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from football_advance_predictor.core.hashing import stable_hash
from football_advance_predictor.core.logging import get_logger
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.data.adapters.base import AvailabilityProvider
from football_advance_predictor.schemas.availability import AvailabilityIn

logger = get_logger(__name__)


class LocalAvailabilityProvider(AvailabilityProvider):
    """Read availability records from a local JSON file."""

    name = "local_availability"

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Availability JSON not found: {self.path}")

    def fetch_availability(self, **kwargs: Any) -> list[AvailabilityIn]:
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Availability JSON must be a list of records")
        out: list[AvailabilityIn] = []
        for idx, record in enumerate(data):
            try:
                out.append(self._record_to_availability(record))
            except Exception as exc:
                logger.warning(
                    "Failed to parse availability record",
                    extra={"record_index": idx, "error": str(exc)},
                )
        return out

    def _record_to_availability(self, record: dict[str, Any]) -> AvailabilityIn:
        observed_at = to_utc(record["observed_at"])
        ingested_at = to_utc(record.get("ingested_at") or observed_at.isoformat())
        effective_at = to_utc(record.get("effective_at") or observed_at.isoformat())
        published_at = to_utc(record.get("published_at") or observed_at.isoformat())
        raw_payload_hash = record.get("raw_payload_hash") or stable_hash(record)
        return AvailabilityIn(
            match_id=record["match_id"],
            team_id=record["team_id"],
            player_id=record.get("player_id"),
            role=record.get("role", "unknown"),
            availability_status=record["availability_status"],
            confidence=float(record.get("confidence", 0.5)),
            published_at=published_at,
            cutoff_eligible=bool(record.get("cutoff_eligible", True)),
            source=record.get("source", "local"),
            raw_text=record.get("raw_text"),
            source_name=record.get("source_name", "local"),
            source_record_id=record.get("source_record_id"),
            source_url=record.get("source_url"),
            observed_at=observed_at,
            ingested_at=ingested_at,
            effective_at=effective_at,
            raw_payload_hash=raw_payload_hash,
            source_version=record.get("source_version"),
        )
