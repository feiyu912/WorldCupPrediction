"""Pytest configuration: shared fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Ensure tests don't hit a real database. Use SQLite in-memory or a temp file.
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "football_test")
os.environ.setdefault("POSTGRES_USER", "football")
os.environ.setdefault("POSTGRES_PASSWORD", "football")
os.environ.setdefault("DUCKDB_PATH", str(Path("./data/test_warehouse.duckdb").resolve()))
os.environ.setdefault("MODEL_REGISTRY_DIR", str(Path("./data/test_models").resolve()))


@pytest.fixture(autouse=True)
def reset_settings_cache() -> None:
    from football_advance_predictor.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "fixtures"
