"""Pytest configuration: shared fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# Ensure tests don't hit a real database. Use SQLite in-memory or a temp file.
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "football_test")
os.environ.setdefault("POSTGRES_USER", "football")
os.environ.setdefault("POSTGRES_PASSWORD", "football")
os.environ.setdefault("DUCKDB_PATH", str(Path("./data/test_warehouse.duckdb").resolve()))
os.environ.setdefault("MODEL_REGISTRY_DIR", str(Path("./data/test_models").resolve()))


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Iterator[None]:
    from football_advance_predictor.core.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).parent.parent / "data" / "fixtures"


@pytest.fixture
def sqlite_engine(monkeypatch) -> Iterator:
    """In-memory SQLite engine with all models registered and tables created."""
    from football_advance_predictor import db
    from football_advance_predictor.db.session import init_db

    engine = create_engine("sqlite:///:memory:", future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    monkeypatch.setattr(db.session, "_engine", engine, raising=False)
    monkeypatch.setattr(db.session, "_SessionLocal", SessionLocal, raising=False)
    init_db()
    yield engine
    engine.dispose()


@pytest.fixture
def sqlite_session_factory(sqlite_engine) -> Iterator[sessionmaker[Session]]:
    """Session factory bound to the in-memory SQLite engine."""
    from football_advance_predictor import db

    SessionLocal = sessionmaker(bind=sqlite_engine, expire_on_commit=False, autoflush=False)
    monkeypatch_set = getattr(db.session, "_SessionLocal", None)
    if monkeypatch_set is not None:
        db.session._SessionLocal = SessionLocal  # type: ignore[attr-defined]
    yield SessionLocal
