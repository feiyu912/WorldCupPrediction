"""Tests for the team-name resolver."""

from __future__ import annotations

import pytest
from football_advance_predictor.data.normalization.team_resolver import (
    TeamNameResolver,
    TeamResolutionError,
)


def test_resolve_known_aliases() -> None:
    resolver = TeamNameResolver()
    assert resolver.resolve("USA") == "united_states"
    assert resolver.resolve("U.S.A.") == "united_states"
    assert resolver.resolve("South Korea") == "south_korea"
    assert resolver.resolve("Korea Republic") == "south_korea"
    assert resolver.resolve("Côte d'Ivoire") == "ivory_coast"


def test_resolve_unknown_quarantined() -> None:
    resolver = TeamNameResolver(quarantine_unresolved=True)
    team_id = resolver.resolve("Atlantis United")
    assert team_id == "atlantis_united"
    assert "Atlantis United" in resolver.export_unresolved()


def test_resolve_unknown_raises_when_not_quarantined() -> None:
    resolver = TeamNameResolver(quarantine_unresolved=False)
    # Empty string is a special case; we test whitespace-only fallback instead.
    with pytest.raises(TeamResolutionError):
        resolver.resolve("   ")


def test_bulk_resolve() -> None:
    resolver = TeamNameResolver()
    out = resolver.bulk_resolve(["USA", "UK", "Brazil"])
    assert out["USA"] == "united_states"
    assert out["UK"] == "england"
    assert out["Brazil"] == "brazil"
