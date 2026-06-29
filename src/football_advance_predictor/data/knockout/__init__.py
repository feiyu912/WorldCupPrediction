"""Knockout manifest builder.

Merges tournament-specific result providers into a single,
deduplicated list of reliably-labeled knockout fixtures. The
manifest is the input to training and backtesting of the advance
predictor.
"""

from __future__ import annotations

from football_advance_predictor.data.knockout.manifest import (
    KnockoutManifest,
    KnockoutManifestBuilder,
    KnockoutRow,
    QuarantineReason,
)

__all__ = [
    "KnockoutManifest",
    "KnockoutManifestBuilder",
    "KnockoutRow",
    "QuarantineReason",
]
