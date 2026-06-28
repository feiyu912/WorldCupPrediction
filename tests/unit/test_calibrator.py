"""Calibrator unit tests."""

from __future__ import annotations

import numpy as np
import pytest
from football_advance_predictor.models.calibration.calibrator import (
    CalibrationConfig,
    Calibrator,
)


def test_isotonic_perfect_predictions_identity() -> None:
    cal = Calibrator(CalibrationConfig(method="isotonic"))
    probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    y = np.array([0, 0, 1, 1, 1])
    cal.fit(probs, y)
    out = cal.predict(probs)
    # Monotonic non-decreasing.
    assert all(out[i] <= out[i + 1] for i in range(len(out) - 1))
    assert all(0 < v < 1 for v in out)


def test_isotonic_save_load_roundtrip(tmp_path) -> None:
    cal = Calibrator(CalibrationConfig(method="isotonic"))
    probs = np.array([0.2, 0.5, 0.8])
    y = np.array([0, 1, 1])
    cal.fit(probs, y)
    out_before = cal.predict(probs)
    path = cal.save(tmp_path)
    loaded = Calibrator.load(path)
    out_after = loaded.predict(probs)
    np.testing.assert_allclose(out_before, out_after, atol=1e-6)


def test_platt_sigmoid_basic() -> None:
    cal = Calibrator(CalibrationConfig(method="platt"))
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    cal.fit(probs, y)
    out = cal.predict(probs)
    assert all(0 < v < 1 for v in out)
    # Monotonic on the sigmoid parameterization.
    assert all(out[i] <= out[i + 1] for i in range(len(out) - 1))


def test_predict_before_fit_raises() -> None:
    cal = Calibrator(CalibrationConfig(method="isotonic"))
    with pytest.raises(RuntimeError):
        cal.predict(np.array([0.5]))


def test_unknown_method_raises() -> None:
    # Construction with an unknown method is permissive; the error fires
    # on fit / predict.
    cal = Calibrator(CalibrationConfig(method="bogus"))
    with pytest.raises(ValueError):
        cal.fit(np.array([0.1, 0.9]), np.array([0, 1]))
    with pytest.raises(ValueError):
        cal.predict(np.array([0.5]))


def test_unknown_method_in_fit_raises() -> None:
    cal = Calibrator(CalibrationConfig(method="bogus"))
    with pytest.raises(ValueError):
        cal.fit(np.array([0.1, 0.9]), np.array([0, 1]))


def test_isotonic_output_bounded() -> None:
    cal = Calibrator(CalibrationConfig(method="isotonic", isotonic_out_of_bounds="clip"))
    probs = np.array([0.01, 0.5, 0.99])
    y = np.array([0, 1, 1])
    cal.fit(probs, y)
    # Out-of-bounds clipping keeps output within (eps, 1-eps).
    out = cal.predict(np.array([-0.5, 1.5]))
    assert (0 < out[0] < 1) and (0 < out[1] < 1)
