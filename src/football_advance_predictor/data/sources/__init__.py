"""Source-specific adapters for raw downloaded data files.

Each adapter reads the raw files produced by the bootstrap downloader
(``data/raw/sources/<source>``) and normalizes them into the canonical
``MatchIn`` / ``MatchResultIn`` records. Adapters are pure parsers:
they never touch the network or the database.
"""

from __future__ import annotations

from football_advance_predictor.data.sources.martj42 import MartJ42ResultsProvider
from football_advance_predictor.data.sources.openfootball import (
    OpenFootballTournamentProvider,
)
from football_advance_predictor.data.sources.statsbomb_open_data import (
    StatsBombOpenDataProvider,
)

__all__ = [
    "MartJ42ResultsProvider",
    "OpenFootballTournamentProvider",
    "StatsBombOpenDataProvider",
]
