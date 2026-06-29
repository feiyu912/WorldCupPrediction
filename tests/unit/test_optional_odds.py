"""Tests for the env-gated optional historical-odds provider."""

from __future__ import annotations

from football_advance_predictor.data.adapters.optional_odds import (
    NoOpHistoricalOddsProvider,
    _SkeletonAuthedNoFetchProvider,
    build_default_optional_provider,
)


def test_unset_env_returns_noop(monkeypatch) -> None:
    monkeypatch.delenv("EXTERNAL_ODDS_API_KEY", raising=False)
    provider = build_default_optional_provider()
    assert isinstance(provider, NoOpHistoricalOddsProvider)
    assert provider.name == "noop_historical_odds"
    # Returns an empty list.
    assert provider.fetch_odds() == []


def test_noop_fetch_odds_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("EXTERNAL_ODDS_API_KEY", raising=False)
    provider = build_default_optional_provider()
    assert provider.fetch_odds(match_id="whatever") == []
    assert provider.fetch_odds() == []


def test_set_env_returns_skeleton_provider(monkeypatch) -> None:
    monkeypatch.setenv("EXTERNAL_ODDS_API_KEY", "test-key-12345")
    provider = build_default_optional_provider()
    # Even with a key set, the skeleton is a stub. No network calls.
    assert isinstance(provider, _SkeletonAuthedNoFetchProvider)
    assert provider.name == "skeleton_optional_odds"
    assert provider.fetch_odds() == []
