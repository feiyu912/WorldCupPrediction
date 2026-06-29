"""Symmetry test: P(A advances | A vs B) + P(B advances | B vs A) ≈ 1.

The test materializes each match twice with the home/away sides
swapped and verifies the probability sums are within a tolerance. The
two mirrored examples must be in the SAME temporal fold (no leakage).

Usage::

    result = symmetry_test(predict_fn, matches, tolerance=0.05)
    assert result["passes"]
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class SymmetryResult:
    n_pairs: int
    n_passes: int
    n_fails: int
    mean_abs_residual: float
    max_abs_residual: float
    tolerance: float
    passes: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_pairs": self.n_pairs,
            "n_passes": self.n_passes,
            "n_fails": self.n_fails,
            "mean_abs_residual": self.mean_abs_residual,
            "max_abs_residual": self.max_abs_residual,
            "tolerance": self.tolerance,
            "passes": self.passes,
        }


def symmetry_test(
    predict_proba: Callable[[dict[str, Any]], float],
    matches: list[dict[str, Any]],
    *,
    tolerance: float = 0.05,
) -> SymmetryResult:
    """Run a mirrored-pair symmetry test.

    Args:
        predict_proba: Callable that takes a feature dict for one match
            and returns P(home_advances).
        matches: List of feature dicts (any order). Each dict must have
            at least ``home_team_id``, ``away_team_id``, and
            ``cutoff_time``.
        tolerance: Maximum allowed |p_ab + p_ba - 1|.

    Returns:
        A :class:`SymmetryResult` with pass/fail counts and residuals.
    """
    n_pairs = 0
    n_passes = 0
    n_fails = 0
    residuals: list[float] = []
    for m in matches:
        forward = predict_proba(m)
        mirrored = dict(m)
        mirrored["home_team_id"], mirrored["away_team_id"] = (
            m["away_team_id"],
            m["home_team_id"],
        )
        reverse = predict_proba(mirrored)
        residual = abs(forward + reverse - 1.0)
        residuals.append(residual)
        n_pairs += 1
        if residual <= tolerance:
            n_passes += 1
        else:
            n_fails += 1
    return SymmetryResult(
        n_pairs=n_pairs,
        n_passes=n_passes,
        n_fails=n_fails,
        mean_abs_residual=sum(residuals) / max(1, len(residuals)),
        max_abs_residual=max(residuals) if residuals else 0.0,
        tolerance=tolerance,
        passes=(n_fails == 0 and n_pairs > 0),
    )
