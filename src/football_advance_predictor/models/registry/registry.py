"""File-based model artifact registry.

Each model version is stored under
``<registry_root>/<model_type>/<model_version>/`` with a manifest
file describing the artifact contents. The registry is the source of
truth for which model versions are available.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ModelArtifact:
    """A registered model artifact."""

    model_type: str
    model_version: str
    artifact_path: str
    feature_version: str
    created_at: datetime
    metrics: dict[str, Any]
    hyperparameters: dict[str, Any]
    feature_hash: str
    additional_files: list[str]


class ModelRegistry:
    """Local file-based model registry.

    Args:
        root: Root directory of the registry. Created if missing.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        model_type: str,
        model_version: str,
        artifact_path: str | Path,
        feature_version: str,
        metrics: dict[str, Any] | None = None,
        hyperparameters: dict[str, Any] | None = None,
        feature_hash: str = "",
        additional_files: list[str] | None = None,
    ) -> ModelArtifact:
        artifact_path = Path(artifact_path)
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact path missing: {artifact_path}")
        manifest = ModelArtifact(
            model_type=model_type,
            model_version=model_version,
            artifact_path=str(artifact_path),
            feature_version=feature_version,
            created_at=datetime.now(tz=UTC),
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            feature_hash=feature_hash,
            additional_files=additional_files or [],
        )
        manifest_path = self.root / model_type / model_version / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2, sort_keys=True, default=str)
        logger.info(
            "Registered model",
            extra={"model_type": model_type, "model_version": model_version},
        )
        return manifest

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(
        self, *, model_type: str, model_version: str
    ) -> ModelArtifact | None:
        manifest_path = self.root / model_type / model_version / "manifest.json"
        if not manifest_path.exists():
            return None
        with manifest_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return ModelArtifact(**data)

    def list_versions(self, model_type: str) -> list[str]:
        base = self.root / model_type
        if not base.exists():
            return []
        return sorted([p.name for p in base.iterdir() if p.is_dir()])

    def latest(self, model_type: str) -> ModelArtifact | None:
        versions = self.list_versions(model_type)
        if not versions:
            return None
        return self.get(model_type=model_type, model_version=versions[-1])
