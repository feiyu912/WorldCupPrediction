"""End-to-end API endpoint tests using FastAPI TestClient."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "warehouse.duckdb"))
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    from football_advance_predictor.core.config import get_settings

    get_settings.cache_clear()
    yield


@pytest.fixture
def client(monkeypatch, tmp_path) -> Iterator[TestClient]:
    from football_advance_predictor import db
    from football_advance_predictor.app.api import main as api_main
    from football_advance_predictor.db.session import get_session, init_db

    # Use a file-backed SQLite (shared across connections) rather than
    # :memory: because in-memory SQLite is connection-private and the
    # API and the test fixture would otherwise see different databases.
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    # Patch the module-level engine so ``init_db`` and ``get_session``
    # both use the test SQLite engine.
    db.session._engine = engine
    db.session._SessionLocal = SessionLocal
    init_db()

    def _override():
        s = SessionLocal()
        try:
            yield s
        finally:
            s.close()

    # Override the dependency on the existing app (which already has all
    # the routes registered at module import time).
    api_main.app.dependency_overrides[get_session] = _override
    with TestClient(api_main.app) as c:
        yield c
    api_main.app.dependency_overrides.clear()


def _ingest_minimal_match(client):
    return client.post(
        "/ingest/matches",
        json=[
            {
                "match_id": "M_API_001",
                "kickoff_at": "2026-07-01T04:00:00Z",
                "competition_id": "WC",
                "stage": "Quarter-final",
                "season_or_year": "2026",
                "home_team_id": "fra",
                "away_team_id": "swe",
                "neutral_venue": True,
            }
        ],
    )


def test_health_endpoint(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ingest_matches_endpoint_creates_teams(client):
    # First call creates teams.
    r = _ingest_minimal_match(client)
    assert r.status_code == 200
    assert r.json()["matches_ingested"] == 1


def test_ingest_matches_endpoint_idempotent(client):
    _ingest_minimal_match(client)
    r = _ingest_minimal_match(client)
    assert r.status_code == 200
    # Idempotent: the second call should upsert, not insert.
    # ``matches_ingested`` reports new rows; expect 0 the second time.
    assert r.json()["matches_ingested"] == 0


def test_ingest_odds_endpoint(client):
    _ingest_minimal_match(client)
    r = client.post(
        "/ingest/odds",
        json=[
            {
                "match_id": "M_API_001",
                "bookmaker": "B",
                "market_type": "home_to_advance",
                "selection": "home",
                "decimal_odds": 1.5,
                "captured_at": "2026-06-30T04:00:00Z",
                "ingested_at": "2026-06-30T04:01:00Z",
                "effective_at": "2026-06-30T04:00:00Z",
                "raw_payload_hash": "h_api_001",
            }
        ],
    )
    assert r.status_code == 200
    assert r.json()["odds_ingested"] == 1


def test_ingest_availability_endpoint(client):
    _ingest_minimal_match(client)
    r = client.post(
        "/ingest/availability",
        json=[
            {
                "match_id": "M_API_001",
                "team_id": "fra",
                "role": "attacker",
                "availability_status": "available",
                "confidence": 0.9,
                "published_at": "2026-06-30T04:00:00Z",
                "observed_at": "2026-06-30T04:00:00Z",
                "ingested_at": "2026-06-30T04:01:00Z",
                "effective_at": "2026-06-30T04:00:00Z",
                "raw_payload_hash": "h_avail_api_001",
            }
        ],
    )
    assert r.status_code == 200
    assert r.json()["availability_ingested"] == 1


def test_features_build_endpoint_rejects_post_kickoff(client):
    _ingest_minimal_match(client)
    r = client.post(
        "/features/build",
        json={
            "match_id": "M_API_001",
            "cutoff_time": "2026-07-02T00:00:00Z",  # AFTER kickoff
            "feature_version": "v1",
        },
    )
    # Global exception handler maps ValueError to 400.
    assert r.status_code == 400
    assert "Cutoff must be strictly before kickoff" in r.json()["error"]


def test_backtests_list_endpoint_returns_empty_when_no_reports(client, tmp_path):
    r = client.get("/backtests")
    assert r.status_code == 200
    assert r.json() == {"backtests": []}


def test_backtests_get_404_when_missing(client):
    r = client.get("/backtests/UNKNOWN_RUN_ID")
    assert r.status_code == 404


def test_reports_model_comparison_with_no_evaluations(client):
    r = client.get("/reports/model-comparison?model_versions=v0")
    assert r.status_code == 200
    data = r.json()
    assert "comparison" in data
    # No evaluations stored: every model returns a row with None metrics.
    assert all(c.get("n_predictions") == 0 for c in data["comparison"])
