"""Source lock: persist the exact resolved Git commit SHA for every source.

The lock is a versioned JSON file at ``data/raw/sources/lock.json`` that
records the requested ref, the resolved 40-character SHA, the retrieval
timestamp, the raw file hash, and the source URL used.

Once a lock exists, every subsequent bootstrap must use the locked
SHA. The downloader refuses to fall back to HEAD once a source is
locked; the only way to update is the explicit ``data update-sources``
command.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
_FULL_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


@dataclass
class LockedSource:
    """A single source entry in the lock file."""

    name: str
    requested_ref: str
    resolved_sha: str
    retrieved_at: str
    source_url: str
    raw_sha256: str
    local_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceLock:
    """Versioned source lock."""

    schema_version: int = SCHEMA_VERSION
    created_at: str = ""
    updated_at: str = ""
    sources: dict[str, LockedSource] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "SourceLock":
        path = Path(path)
        if not path.exists():
            lock = cls(created_at=_now_iso(), updated_at=_now_iso())
            return lock
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        lock = cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )
        for name, raw in data.get("sources", {}).items():
            lock.sources[name] = LockedSource(**raw)
        return lock

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now_iso()
        payload = {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "sources": {name: s.to_dict() for name, s in self.sources.items()},
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logger.info(
            "Wrote source lock",
            extra={"path": str(path), "n_sources": len(self.sources)},
        )

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def get(self, name: str) -> LockedSource | None:
        return self.sources.get(name)

    def is_locked(self, name: str) -> bool:
        return name in self.sources

    def names(self) -> list[str]:
        return sorted(self.sources)

    def validate_full_sha(self, name: str) -> bool:
        """Return True if the locked SHA is a full 40-character Git commit SHA."""
        s = self.sources.get(name)
        if s is None:
            return False
        return bool(_FULL_SHA_PATTERN.match(s.resolved_sha))

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def set_or_update(
        self,
        *,
        name: str,
        requested_ref: str,
        resolved_sha: str,
        source_url: str,
        raw_sha256: str,
        local_path: str,
    ) -> None:
        if not _FULL_SHA_PATTERN.match(resolved_sha):
            raise ValueError(
                f"resolved_sha must be a 40-character commit hash; got {resolved_sha!r}"
            )
        existing = self.sources.get(name)
        now = _now_iso()
        if existing is not None and existing.resolved_sha == resolved_sha:
            # No-op; just update retrieval metadata.
            existing.retrieved_at = now
            return
        self.sources[name] = LockedSource(
            name=name,
            requested_ref=requested_ref,
            resolved_sha=resolved_sha,
            retrieved_at=now,
            source_url=source_url,
            raw_sha256=raw_sha256,
            local_path=local_path,
        )

    def clear(self) -> None:
        self.sources.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "n_locked_sources": len(self.sources),
            "sources": {n: s.to_dict() for n, s in self.sources.items()},
        }


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
