"""Tests for the AliasRegistry."""

from __future__ import annotations

from pathlib import Path

from football_advance_predictor.data.aliases.alias_registry import (
    AliasRegistry,
    canonical_key,
)


def test_open_in_fresh_dir_seeds_defaults(tmp_path: Path) -> None:
    reg = AliasRegistry.open(tmp_path)
    # The built-in defaults cover well-known country variants.
    assert reg.size() > 0
    # Sanity-check a few well-known mappings.
    assert reg.resolve("USA") == "united_states"
    assert reg.resolve("South Korea") == "south_korea"
    assert reg.resolve("Côte d'Ivoire") == "ivory_coast"
    assert reg.resolve("Holland") == "netherlands"
    # File was created on open.
    assert (tmp_path / "alias_registry.json").exists()


def test_resolve_usa_returns_united_states(tmp_path: Path) -> None:
    reg = AliasRegistry.open(tmp_path)
    # Various spellings all map to united_states.
    assert reg.resolve("USA") == "united_states"
    assert reg.resolve("U.S.A.") == "united_states"
    assert reg.resolve("us") == "united_states"


def test_unknown_name_records_unresolved(tmp_path: Path) -> None:
    reg = AliasRegistry.open(tmp_path)
    # "Atlantis" is unknown -> falls back to slug and is recorded.
    before = reg.unresolved_count()
    result = reg.resolve("Atlantis")
    after = reg.unresolved_count()
    assert result == "atlantis"
    assert after == before + 1
    assert "Atlantis" in reg.unresolved_names()


def test_seed_from_observed_adds_new_and_skips_known(tmp_path: Path) -> None:
    reg = AliasRegistry.open(tmp_path)
    initial_size = reg.size()

    added = reg.seed_from_observed(
        ["Brazil", "Atlantis FC", "USA"],
        source="test_source",
    )
    # 'Brazil' and 'Atlantis FC' are new; 'USA' is already a default alias.
    assert added == 2
    assert reg.size() == initial_size + 2

    # Re-running with the same names does not add anything new.
    added_again = reg.seed_from_observed(
        ["Brazil", "Atlantis FC", "USA"],
        source="test_source",
    )
    assert added_again == 0
    assert reg.size() == initial_size + 2

    # Brazil resolves to its own slug; the existing USA alias is preserved.
    assert reg.resolve("Brazil") == "brazil"
    assert reg.resolve("USA") == "united_states"


def test_unresolved_count_reflects_queue_size(tmp_path: Path) -> None:
    reg = AliasRegistry.open(tmp_path)
    assert reg.unresolved_count() == 0

    reg.resolve("Atlantis")
    reg.resolve("Lemuria")
    assert reg.unresolved_count() == 2

    reg.clear_unresolved()
    assert reg.unresolved_count() == 0


def test_add_explicit_updates_existing_alias(tmp_path: Path) -> None:
    reg = AliasRegistry.open(tmp_path)
    # USA maps to united_states by default.
    assert reg.resolve("USA") == "united_states"

    # Add an explicit override.
    added = reg.add_explicit("USA", "us_custom", source="test_source")
    assert added is True
    assert reg.resolve("USA") == "us_custom"

    # Re-adding the exact same mapping returns False (no change).
    not_added = reg.add_explicit("USA", "us_custom", source="test_source")
    assert not_added is False


def test_canonical_key_lowercases_and_strips_accents() -> None:
    assert canonical_key("Côte d'Ivoire") == canonical_key("cote divoire")
    assert canonical_key("  USA ") == canonical_key("usa")
    # Punctuation is stripped, spaces collapsed.
    assert canonical_key("South-Korea") == "south korea"
