"""Tests for the SourceRegistry."""

from __future__ import annotations

from pathlib import Path

import pytest
from football_advance_predictor.data.bootstrap.source_registry import (
    SourceRegistry,
    SourceSpec,
)

# Use the actual registry shipped with the repo so tests reflect the
# real pinned URLs/SHAs the bootstrap layer is configured against.
REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "sources"
    / "registry.json"
)


def _load_registry() -> SourceRegistry:
    assert REGISTRY_PATH.exists(), (
        f"Expected source registry at {REGISTRY_PATH}, but it is missing."
    )
    return SourceRegistry(REGISTRY_PATH)


def test_open_returns_source_spec_entries() -> None:
    registry = _load_registry()
    names = registry.all_names()
    assert "martj42_results" in names
    assert "martj42_shootouts" in names
    # At least one openfootball source must be present.
    assert any(n.startswith("openfootball") for n in names)

    spec = registry.get("martj42_results")
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "results_csv"
    assert spec.url_template.startswith("https://raw.githubusercontent.com/")
    assert spec.pinned_sha, "pinned_sha must not be empty"
    assert spec.local_filename == "martj42_results.csv"
    assert "date" in spec.expected_columns


def test_pinned_urls_render_with_correct_sha() -> None:
    registry = _load_registry()
    results_spec = registry.get("martj42_results")
    # URL template has a single '{sha}' placeholder.
    resolved = results_spec.resolved_url
    assert "{sha}" not in resolved
    assert results_spec.pinned_sha in resolved
    assert results_spec.resolved_url == (
        "https://raw.githubusercontent.com/martj42/international_results/"
        f"{results_spec.pinned_sha}/results.csv"
    )

    shootouts_spec = registry.get("martj42_shootouts")
    assert shootouts_spec.pinned_sha in shootouts_spec.resolved_url
    assert shootouts_spec.resolved_url.endswith(
        f"/international_results/{shootouts_spec.pinned_sha}/shootouts.csv"
    )


def test_all_required_returns_results_shootouts() -> None:
    registry = _load_registry()
    required = registry.all_required()
    required_names = {s.name for s in required}
    # The required set depends on registry.json; the MVP shipping
    # registry requires only martj42 (openfootball is optional because
    # it is per-year structured JSON).
    for name in required_names:
        assert name.startswith("martj42")
    for s in required:
        assert isinstance(s, SourceSpec)


def test_all_optional_returns_the_rest() -> None:
    registry = _load_registry()
    optional = registry.all_optional()
    optional_names = {s.name for s in optional}
    required_names = {"martj42_results", "martj42_shootouts"}
    assert optional_names.isdisjoint(required_names)
    # The registry ships optional openfootball per-year entries plus statsbomb.
    assert any("openfootball" in n for n in optional_names)
    assert "statsbomb_open_data" in optional_names


def test_missing_registry_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        SourceRegistry(missing)
