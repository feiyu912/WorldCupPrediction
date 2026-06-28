"""Team-name normalization.

Different sources use different names for the same team (e.g.
"USA" vs "United States" vs "US"). The resolver:

1. canonicalizes whitespace and case,
2. checks an explicit alias table,
3. falls back to a deterministic slug,
4. quarantines unresolved cases.

The resolver is intentionally not a learning system. New aliases must
be added explicitly to the alias table.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


class TeamResolutionError(ValueError):
    """Raised when a team name cannot be resolved."""


_DEFAULT_ALIASES: dict[str, str] = {
    "usa": "united_states",
    "us": "united_states",
    "u.s.a.": "united_states",
    "united states": "united_states",
    "united states of america": "united_states",
    "uk": "england",
    "great britain": "england",
    "south korea": "south_korea",
    "korea republic": "south_korea",
    "korea, south": "south_korea",
    "ivory coast": "ivory_coast",
    "cote d'ivoire": "ivory_coast",
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
}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )


def _canonical_key(value: str) -> str:
    text = value.strip().lower()
    text = _strip_accents(text)
    text = re.sub(r"[\s\-_]+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text.strip()


@dataclass
class TeamNameResolver:
    """Resolve free-form team names to a stable ``team_id`` slug.

    Args:
        aliases: Optional additional alias -> canonical mapping.
        quarantine_unresolved: If True, unresolved names are stored in
            ``unresolved`` rather than raising.
    """

    aliases: dict[str, str] = field(default_factory=dict)
    quarantine_unresolved: bool = True

    unresolved: list[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        merged: dict[str, str] = {}
        for k, v in _DEFAULT_ALIASES.items():
            merged[_canonical_key(k)] = _canonical_key(v).replace(" ", "_")
        for k, v in self.aliases.items():
            merged[_canonical_key(k)] = _canonical_key(v).replace(" ", "_")
        self._alias_lookup = merged

    def resolve(self, name: str) -> str:
        """Return a stable team_id slug for the given name.

        Args:
            name: Free-form team name.

        Raises:
            TeamResolutionError: If the name cannot be resolved and
                ``quarantine_unresolved`` is False.
        """
        if name is None:
            raise TeamResolutionError("Team name is None")
        canonical = _canonical_key(name)
        if not canonical:
            if self.quarantine_unresolved:
                self.unresolved.append(name)
                return "unknown"
            raise TeamResolutionError(f"Empty team name: {name!r}")
        if canonical in self._alias_lookup:
            return self._alias_lookup[canonical]
        # Fall back to a slug.
        slug = canonical.replace(" ", "_")
        if slug in self._alias_lookup.values():
            return slug
        # New team: store in unresolved for manual review.
        if self.quarantine_unresolved:
            self.unresolved.append(name)
        return slug

    def bulk_resolve(self, names: Iterable[str]) -> dict[str, str]:
        """Resolve multiple names and return a mapping ``name -> team_id``."""
        return {name: self.resolve(name) for name in names}

    def export_unresolved(self) -> list[str]:
        return list(self.unresolved)
