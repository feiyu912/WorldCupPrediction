"""System-owned team alias registry.

The registry is a versioned JSON file. It is seeded automatically from
observed source names and from a small built-in default table (covering
the most common country-name variants). Names that the resolver cannot
canonicalize are appended to a separate unresolved-queue file so a
maintainer can review them later.

The registry is the single source of truth for team canonical IDs;
nothing else in the system hand-author teams.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)

# Built-in defaults cover the most common country-name variants. The
# registry is allowed to grow these via the seed-and-extend flow.
_DEFAULT_ALIASES: dict[str, str] = {
    "usa": "united_states",
    "us": "united_states",
    "u.s.a.": "united_states",
    "united states": "united_states",
    "united states of america": "united_states",
    "uk": "england",
    "great britain": "england",
    "britain": "england",
    "south korea": "south_korea",
    "korea republic": "south_korea",
    "korea, south": "south_korea",
    "korea south": "south_korea",
    "korea dpr": "north_korea",
    "north korea": "north_korea",
    "ivory coast": "ivory_coast",
    "cote d'ivoire": "ivory_coast",
    "côte d’ivoire": "ivory_coast",
    "czech republic": "czech_republic",
    "czechia": "czech_republic",
    "russia": "russia",
    "turkey": "turkiye",
    "türkiye": "turkiye",
    "iran": "iran",
    "uae": "united_arab_emirates",
    "u.a.e.": "united_arab_emirates",
    "trinidad": "trinidad_and_tobago",
    "trinidad & tobago": "trinidad_and_tobago",
    "trinidad and tobago": "trinidad_and_tobago",
    "bosnia": "bosnia_and_herzegovina",
    "bosnia & herzegovina": "bosnia_and_herzegovina",
    "cabo verde": "cape_verde",
    "cape verde": "cape_verde",
    "eswatini": "eswatini",
    "swaziland": "eswatini",
    "czech": "czech_republic",
    "holland": "netherlands",
    "the netherlands": "netherlands",
    "korea": "south_korea",
    "republic of ireland": "ireland",
    "northern ireland": "northern_ireland",
}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def canonical_key(value: str) -> str:
    """Return a stable lowercase key for alias lookup."""
    text = value.strip().lower()
    text = _strip_accents(text)
    text = re.sub(r"[\s\-_]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


@dataclass
class AliasRegistryEntry:
    """A single alias → canonical team_id mapping."""

    alias: str
    team_id: str
    source: str  # which provider/seed contributed this entry
    added_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


@dataclass
class AliasRegistry:
    """Versioned alias registry with a backing JSON file and an unresolved queue.

    Args:
        path: Path to ``alias_registry.json`` (created if missing).
        unresolved_path: Path to ``unresolved.jsonl`` (one unresolved
            name per line, JSON-encoded).
    """

    path: Path
    unresolved_path: Path
    _entries: dict[str, AliasRegistryEntry] = field(default_factory=dict)
    _version: int = 1
    _auto_seed_applied: bool = False

    @classmethod
    def open(
        cls,
        directory: str | Path,
    ) -> AliasRegistry:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        registry = cls(
            path=directory / "alias_registry.json",
            unresolved_path=directory / "unresolved.jsonl",
        )
        registry._load()
        return registry

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            # Seed with defaults.
            for raw_alias, raw_target in _DEFAULT_ALIASES.items():
                self._entries[canonical_key(raw_alias)] = AliasRegistryEntry(
                    alias=raw_alias, team_id=raw_target, source="builtin_default"
                )
            self._auto_seed_applied = True
            self.save()
            return
        with self.path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._version = int(data.get("version", 1))
        for entry in data.get("entries", []):
            e = AliasRegistryEntry(**entry)
            self._entries[canonical_key(e.alias)] = e

    def save(self) -> None:
        payload = {
            "version": self._version,
            "updated_at": datetime.now(tz=UTC).isoformat(),
            "entries": [asdict(e) for e in sorted(self._entries.values(), key=lambda x: x.alias)],
        }
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        logger.info(
            "Saved alias registry",
            extra={"path": str(self.path), "n_entries": len(self._entries)},
        )

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def resolve(self, raw_name: str, *, source: str = "runtime") -> str:
        """Return the canonical team_id for ``raw_name``.

        Resolution order:
        1. Exact match in the registry (case-insensitive, normalized).
        2. Built-in default table (already loaded).
        3. Slug fallback (lower-cased, accent-stripped, spaces -> "_").
        Unresolved names are appended to the unresolved queue.
        """
        if not raw_name:
            return "unknown"
        key = canonical_key(raw_name)
        if key in self._entries:
            return self._entries[key].team_id
        # Slug fallback.
        slug = key.replace(" ", "_") if key else "unknown"
        # Record the unresolved name for later review.
        self._record_unresolved(raw_name, source)
        return slug

    def resolve_bulk(self, names: Iterable[str], *, source: str = "runtime") -> dict[str, str]:
        return {n: self.resolve(n, source=source) for n in names}

    # ------------------------------------------------------------------
    # Seeding from observed source data
    # ------------------------------------------------------------------

    def seed_from_observed(self, names: Iterable[str], *, source: str) -> int:
        """Add observed raw names to the registry, keyed by their slug.

        Returns the number of new aliases added.
        """
        added = 0
        for raw in names:
            if not raw:
                continue
            key = canonical_key(raw)
            if key in self._entries:
                continue
            slug = key.replace(" ", "_") if key else "unknown"
            self._entries[key] = AliasRegistryEntry(
                alias=raw, team_id=slug, source=source
            )
            added += 1
        if added:
            self.save()
        return added

    def add_explicit(self, raw_alias: str, team_id: str, *, source: str) -> bool:
        """Add or update an explicit alias mapping. Returns True if added."""
        key = canonical_key(raw_alias)
        if key in self._entries and self._entries[key].team_id == team_id:
            return False
        self._entries[key] = AliasRegistryEntry(
            alias=raw_alias, team_id=team_id, source=source
        )
        self.save()
        return True

    # ------------------------------------------------------------------
    # Unresolved queue
    # ------------------------------------------------------------------

    def _record_unresolved(self, raw_name: str, source: str) -> None:
        record = {
            "raw_name": raw_name,
            "source": source,
            "observed_at": datetime.now(tz=UTC).isoformat(),
        }
        with self.unresolved_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def unresolved_count(self) -> int:
        if not self.unresolved_path.exists():
            return 0
        n = 0
        with self.unresolved_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    n += 1
        return n

    def unresolved_names(self) -> list[str]:
        if not self.unresolved_path.exists():
            return []
        out: list[str] = []
        with self.unresolved_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    out.append(json.loads(line)["raw_name"])
                except Exception:
                    continue
        return out

    def clear_unresolved(self) -> int:
        if not self.unresolved_path.exists():
            return 0
        n = self.unresolved_count()
        self.unresolved_path.unlink()
        return n

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def size(self) -> int:
        return len(self._entries)

    def entries(self) -> list[AliasRegistryEntry]:
        return sorted(self._entries.values(), key=lambda e: e.alias)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self._version,
            "size": len(self._entries),
            "unresolved_count": self.unresolved_count(),
        }
