"""Source registry: versioned JSON listing every data source the system trusts.

The registry is the single source of truth for *which* URLs the
bootstrap layer is allowed to download and *which* commit SHAs are
pinned. Updating the registry is the only way to add or refresh a
source.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceSpec:
    """A single source entry in the registry."""

    name: str
    kind: str
    url_template: str
    pinned_sha: str
    local_filename: str | None
    local_path: str | None
    expected_columns: tuple[str, ...]
    expected_keys: tuple[str, ...]
    description: str

    @property
    def resolved_url(self) -> str:
        return self.url_template.format(sha=self.pinned_sha)


class SourceRegistry:
    """Read-only registry of pinned data sources.

    Args:
        path: Path to the registry JSON file.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Source registry not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as f:
            self._data: dict[str, Any] = json.load(f)
        self.schema_version = int(self._data.get("schema_version", 1))
        self.updated_at = self._data.get("updated_at", "")

    def get(self, name: str) -> SourceSpec:
        if name not in self._data["sources"]:
            raise KeyError(f"Unknown source: {name!r}")
        raw = self._data["sources"][name]
        return SourceSpec(
            name=name,
            kind=raw["kind"],
            url_template=raw["url_template"],
            pinned_sha=raw["pinned_sha"],
            local_filename=raw.get("local_filename"),
            local_path=raw.get("local_path"),
            expected_columns=tuple(raw.get("expected_columns", ())),
            expected_keys=tuple(raw.get("expected_keys", ())),
            description=raw.get("description", ""),
        )

    def all_names(self) -> list[str]:
        return sorted(self._data["sources"].keys())

    def all_required(self) -> list[SourceSpec]:
        """Sources required for a minimal bootstrap (results + shootouts + worldcup)."""
        return [self.get(n) for n in ("martj42_results", "martj42_shootouts", "openfootball_worldcup")]

    def all_optional(self) -> list[SourceSpec]:
        """Sources that are downloaded opportunistically when reachable."""
        return [self.get(n) for n in self.all_names() if n not in {"martj42_results", "martj42_shootouts", "openfootball_worldcup"}]
