"""Tests for the SourceDownloader with a mock HTTP fetcher."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from football_advance_predictor.data.bootstrap.source_downloader import (
    DownloadError,
    SourceDownloader,
)
from football_advance_predictor.data.bootstrap.source_registry import SourceRegistry

REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "sources"
    / "registry.json"
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeFetcher:
    """Configurable HTTP fetcher for tests."""

    def __init__(self, payload: bytes | None = None, *, raise_exc: Exception | None = None) -> None:
        self._payload = payload
        self._raise = raise_exc
        self.calls: list[str] = []

    def fetch(self, url: str) -> bytes:
        self.calls.append(url)
        if self._raise is not None:
            raise self._raise
        assert self._payload is not None
        return self._payload


def _write_registry(path: Path, *, sources: dict[str, dict[str, Any]]) -> None:
    payload = {"schema_version": 1, "updated_at": "2026-06-29", "sources": sources}
    path.write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def results_registry(tmp_path: Path) -> tuple[Path, Path, SourceRegistry]:
    """Write a temp registry with one results_csv and one shootouts_csv source."""
    registry_path = tmp_path / "registry.json"
    sources = {
        "martj42_results": {
            "kind": "results_csv",
            "url_template": "https://example.test/results/{sha}.csv",
            "pinned_sha": "abc123",
            "local_filename": "martj42_results.csv",
            "expected_columns": [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
                "city",
                "country",
                "neutral",
            ],
            "description": "results",
        },
        "martj42_shootouts": {
            "kind": "shootouts_csv",
            "url_template": "https://example.test/shootouts/{sha}.csv",
            "pinned_sha": "abc123",
            "local_filename": "martj42_shootouts.csv",
            "expected_columns": ["date", "home_team", "away_team", "winner"],
            "description": "shootouts",
        },
    }
    _write_registry(registry_path, sources=sources)
    return registry_path, tmp_path, SourceRegistry(registry_path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cache_hit_returns_zero_bytes(tmp_path: Path, results_registry) -> None:
    _registry_path, raw_dir, registry = results_registry
    target = raw_dir / "martj42_results.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    # The cached file must satisfy the schema; otherwise the downloader
    # propagates DownloadError from the schema validator.
    valid_csv = b"date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n2024-01-01,A,B,0,0,Friendly,X,Y,FALSE\n"
    target.write_bytes(valid_csv)

    downloader = SourceDownloader(registry, raw_dir, fetcher=FakeFetcher())
    result = downloader.download("martj42_results")

    assert result.cache_hit is True
    assert result.bytes_written == 0
    assert result.local_path == target
    assert result.sha256 == hashlib.sha256(valid_csv).hexdigest()
    assert result.schema_valid is True


def test_fresh_download_writes_file_with_correct_sha(tmp_path: Path, results_registry) -> None:
    _registry_path, raw_dir, registry = results_registry
    payload = (
        b"date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        b"2024-01-01,Brazil,Argentina,2,0,Friendly,Rio,Brazil,FALSE\n"
    )

    downloader = SourceDownloader(
        registry, raw_dir, fetcher=FakeFetcher(payload=payload)
    )
    result = downloader.download("martj42_results")

    assert result.cache_hit is False
    assert result.bytes_written == len(payload)
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    target = raw_dir / "martj42_results.csv"
    assert target.read_bytes() == payload
    assert result.schema_valid is True


def test_schema_validation_failure_raises(tmp_path: Path, results_registry) -> None:
    _registry_path, raw_dir, registry = results_registry
    # Missing required columns
    bad_payload = b"date,home_team,away_team\n2024-01-01,Brazil,Argentina\n"
    downloader = SourceDownloader(
        registry, raw_dir, fetcher=FakeFetcher(payload=bad_payload)
    )

    with pytest.raises(DownloadError, match="schema mismatch"):
        downloader.download("martj42_results")


def test_network_failure_propagates(tmp_path: Path, results_registry) -> None:
    _registry_path, raw_dir, registry = results_registry
    downloader = SourceDownloader(
        registry,
        raw_dir,
        fetcher=FakeFetcher(raise_exc=DownloadError("network down")),
    )
    with pytest.raises(DownloadError, match="network down"):
        downloader.download("martj42_results")


def test_write_manifest_writes_json(tmp_path: Path, results_registry) -> None:
    _registry_path, raw_dir, registry = results_registry
    payload = b"date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n2024-01-01,Brazil,Argentina,1,0,Friendly,X,Y,FALSE\n"
    downloader = SourceDownloader(
        registry, raw_dir, fetcher=FakeFetcher(payload=payload)
    )
    # Pre-create cached shootouts file so we can list two results.
    shootouts = raw_dir / "martj42_shootouts.csv"
    shootouts.write_bytes(b"date,home_team,away_team,winner\n")

    manifest_path = tmp_path / "manifests" / "download_manifest.json"
    results = [
        downloader.download("martj42_results"),
        downloader.download("martj42_shootouts"),
    ]
    downloader.write_manifest(results, manifest_path)

    assert manifest_path.exists()
    content = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "generated_at" in content
    assert len(content["sources"]) == 2
    names = sorted(s["name"] for s in content["sources"])
    assert names == ["martj42_results", "martj42_shootouts"]
    # Spot-check metadata for each entry.
    for entry in content["sources"]:
        assert "pinned_sha" in entry
        assert "resolved_url" in entry
        assert "sha256" in entry
        assert "cache_hit" in entry


def test_offline_mode_with_cache_miss_raises(tmp_path: Path, results_registry) -> None:
    _registry_path, raw_dir, registry = results_registry
    downloader = SourceDownloader(
        registry,
        raw_dir,
        fetcher=FakeFetcher(payload=b"unused"),
        offline=True,
    )
    with pytest.raises(DownloadError, match="offline mode is on"):
        downloader.download("martj42_results")
