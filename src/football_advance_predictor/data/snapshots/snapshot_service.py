"""Service for building and persisting immutable feature snapshots.

A feature snapshot is a contract: it pins down the feature values used
for a particular (match, cutoff_time, feature_version) triple. Once
created, it is immutable. Re-running the build returns the same hash.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from football_advance_predictor.core.hashing import stable_hash
from football_advance_predictor.core.time import to_utc
from football_advance_predictor.db.models import FeatureSnapshot
from football_advance_predictor.features.builders.feature_builder import FeatureBuilder


class FeatureSnapshotService:
    """Build, persist, and read feature snapshots."""

    def __init__(self, session: Session, feature_builder: FeatureBuilder | None = None) -> None:
        self.session = session
        self.feature_builder = feature_builder or FeatureBuilder(session=session)

    def build_or_get(
        self,
        match_id: str,
        cutoff_time: datetime,
        feature_version: str,
    ) -> FeatureSnapshot:
        """Build a snapshot if missing; otherwise return the existing one.

        Args:
            match_id: The match identifier.
            cutoff_time: Cutoff timestamp (must be before kickoff).
            feature_version: Logical feature version (e.g. ``"v1"``).

        Returns:
            The persisted :class:`FeatureSnapshot` row.
        """
        cutoff = to_utc(cutoff_time)
        existing = self.session.scalar(
            select(FeatureSnapshot).where(
                FeatureSnapshot.match_id == match_id,
                FeatureSnapshot.cutoff_time == cutoff,
                FeatureSnapshot.feature_version == feature_version,
            )
        )
        if existing is not None:
            return existing

        features, source_max_ts = self.feature_builder.build(match_id=match_id, cutoff_time=cutoff)
        snapshot = FeatureSnapshot(
            match_id=match_id,
            cutoff_time=cutoff,
            feature_version=feature_version,
            features_json=features,
            source_data_max_timestamp=source_max_ts,
            immutable_hash=stable_hash(
                {
                    "match_id": match_id,
                    "cutoff_time": cutoff.isoformat(),
                    "feature_version": feature_version,
                    "features": features,
                    "source_data_max_timestamp": source_max_ts.isoformat(),
                }
            ),
        )
        self.session.add(snapshot)
        self.session.flush()
        return snapshot

    def features_dict(self, snapshot: FeatureSnapshot) -> dict[str, Any]:
        return dict(snapshot.features_json)
