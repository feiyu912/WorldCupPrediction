"""CLI smoke tests using Typer's test runner."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("DUCKDB_PATH", str(tmp_path / "warehouse.duckdb"))
    monkeypatch.setenv("MODEL_REGISTRY_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    from football_advance_predictor.core.config import get_settings

    get_settings.cache_clear()
    yield


def test_cli_help_runs():
    from football_advance_predictor.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Football advance predictor CLI" in result.output


def test_cli_subcommand_helps():
    from football_advance_predictor.cli.main import app

    runner = CliRunner()
    for cmd in ("ingest", "features", "models", "backtest", "report", "predict", "ledger"):
        result = runner.invoke(app, [cmd, "--help"])
        assert result.exit_code == 0, f"{cmd} --help failed: {result.output}"
        assert "Usage:" in result.output or "Options:" in result.output


def test_cli_ingest_matches_command_runs(fixture_dir, tmp_path, monkeypatch):
    """A full ingest call against a tiny CSV succeeds (SQLite-only via monkeypatch)."""
    from football_advance_predictor import db
    from football_advance_predictor.cli.main import app
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", future=True)
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    db.session._engine = engine
    db.session._SessionLocal = SessionLocal
    db.session.init_db()

    monkeypatch.setattr(
        "football_advance_predictor.core.config.get_settings",
        lambda: type("S", (), {
            "log_level": "WARNING",
            "model_registry_dir": tmp_path / "models",
            "reports_dir": tmp_path / "reports",
            "database_url": "sqlite:///:memory:",
        })(),
    )

    # Also patch get_settings at the cli module level
    from football_advance_predictor.cli import main as cli_main

    monkeypatch.setattr(cli_main, "get_settings", lambda: type("S", (), {
        "log_level": "WARNING",
        "model_registry_dir": tmp_path / "models",
        "reports_dir": tmp_path / "reports",
        "database_url": "sqlite:///:memory:",
    })())

    runner = CliRunner()
    result = runner.invoke(app, ["ingest", "matches", "--file", str(fixture_dir / "matches.csv")])
    # The CLI calls init_db, which now works with sqlite.
    # The test is "doesn't crash with a database error" - a successful run prints the count.
    assert result.exit_code == 0, f"ingest matches failed: {result.output}"
    assert "Ingested" in result.output


def test_cli_no_args_shows_help():
    from football_advance_predictor.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, [])
    # Typer with ``no_args_is_help=True`` exits with 0 in real shell but
    # CliRunner treats it as exit code 2; assert the help is shown
    # regardless.
    assert "Usage:" in result.output or "Commands:" in result.output
