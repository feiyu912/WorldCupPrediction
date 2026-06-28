"""Anti-leakage tests for time utilities and snapshot filtering."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from football_advance_predictor.core.hashing import stable_hash
from football_advance_predictor.core.time import (
    assert_cutoff_before,
    assert_observed_before,
    to_utc,
)


def test_to_utc_accepts_naive() -> None:
    dt = to_utc("2026-01-01T00:00:00")
    assert dt.tzinfo is not None


def test_to_utc_accepts_zulu() -> None:
    dt = to_utc("2026-01-01T00:00:00Z")
    assert dt.tzinfo is not None


def test_to_utc_rejects_invalid_string() -> None:
    with pytest.raises(ValueError):
        to_utc("not-a-date")


def test_assert_cutoff_before_passes() -> None:
    assert_cutoff_before(
        datetime(2026, 1, 1, tzinfo=UTC),
        datetime(2026, 1, 2, tzinfo=UTC),
        label="test",
    )


def test_assert_cutoff_before_fails_when_equal() -> None:
    dt = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        assert_cutoff_before(dt, dt, label="test")


def test_assert_observed_before_rejects_future() -> None:
    with pytest.raises(ValueError):
        assert_observed_before(
            datetime(2026, 1, 2, tzinfo=UTC),
            datetime(2026, 1, 1, tzinfo=UTC),
            label="test",
        )


def test_stable_hash_is_deterministic_and_order_invariant() -> None:
    a = stable_hash({"a": 1, "b": 2, "c": [1, 2, 3]})
    b = stable_hash({"c": [1, 2, 3], "b": 2, "a": 1})
    assert a == b
    c = stable_hash({"a": 1, "b": 3, "c": [1, 2, 3]})
    assert a != c
