"""Smoke tests for the API surface (in-memory SQLite)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "warehouse.duckdb"))
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    from football_advance_predictor.core.config import get_settings

    get_settings.cache_clear()
    yield


@pytest.fixture
def client(monkeypatch):
    from football_advance_predictor import db
    from football_advance_predictor.app.api import main as api_main
    from football_advance_predictor.db.session import get_session

    engine = create_engine("sqlite:///:memory:", future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    db.session._engine = engine
    db.session._SessionLocal = SessionLocal
    db.session.init_db()

    def _override():
        s = SessionLocal()
        try:
            yield s
        finally:
            s.close()

    api_main.app.dependency_overrides[get_session] = _override
    return TestClient(api_main.app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
