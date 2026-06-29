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
    assert "openfootball_worldcup" in names

    spec = registry.get("martj42_results")
    assert isinstance(spec, SourceSpec)
    assert spec.kind == "results_csv"
    assert spec.url_template.startswith("https://raw.githubusercontent.com/")
    assert spec.pinned_sha == "HEAD"
    assert spec.local_filename == "martj42_results.csv"
    assert "date" in spec.expected_columns


def test_pinned_urls_render_with_correct_sha() -> None:
    registry = _load_registry()
    results_spec = registry.get("martj42_results")
    # URL template has a single '{sha}' placeholder.
    resolved = results_spec.resolved_url
    assert "{sha}" not in resolved
    assert resolved == (
        "https://raw.githubusercontent.com/martj42/international-results/HEAD/results.csv"
    )

    shootouts_spec = registry.get("martj42_shootouts")
    assert shootouts_spec.resolved_url.endswith("/international-results/HEAD/shootouts.csv")

    wc_spec = registry.get("openfootball_worldcup")
    # json url has no {sha}; sha is included verbatim
    assert "master" in wc_spec.resolved_url
    assert wc_spec.resolved_url == wc_spec.url_template.format(sha=wc_spec.pinned_sha)


def test_all_required_returns_results_shootouts_worldcup() -> None:
    registry = _load_registry()
    required = registry.all_required()
    required_names = {s.name for s in required}
    assert required_names == {"martj42_results", "martj42_shootouts", "openfootball_worldcup"}
    for s in required:
        assert isinstance(s, SourceSpec)


def test_all_optional_returns_the_rest() -> None:
    registry = _load_registry()
    optional = registry.all_optional()
    optional_names = {s.name for s in optional}
    # Required names must not appear in optional.
    required_names = {"martj42_results", "martj42_shootouts", "openfootball_worldcup"}
    assert optional_names.isdisjoint(required_names)
    # The registry ships optional entries for european/openfootball plus statsbomb.
    assert "openfootball_euro" in optional_names
    assert "statsbomb_open_data" in optional_names


def test_missing_registry_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        SourceRegistry(missing)
