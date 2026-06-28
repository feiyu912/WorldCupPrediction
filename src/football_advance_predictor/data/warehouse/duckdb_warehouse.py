"""DuckDB warehouse wrapper.

Provides a thin context-manager-friendly interface to a single DuckDB
file. Use the warehouse for offline feature engineering and exports
to Parquet. Do not use it as the system of record for predictions.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import duckdb

from football_advance_predictor.core.config import get_settings
from football_advance_predictor.core.logging import get_logger

logger = get_logger(__name__)


class DuckDBWarehouse:
    """Thin DuckDB wrapper.

    Args:
        path: Path to the DuckDB file. Defaults to settings.duckdb_path.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = get_settings().duckdb_path
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[duckdb.DuckDBPyConnection]:
        conn = duckdb.connect(str(self.path))
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, sql: str, params: list | None = None) -> list[tuple]:
        with self.connect() as conn:
            result = conn.execute(sql, params or [])
            return result.fetchall()

    def export_to_parquet(self, sql: str, output: str | Path) -> None:
        with self.connect() as conn:
            conn.execute(f"COPY ({sql}) TO '{output}' (FORMAT PARQUET)")
        logger.info("Exported query to Parquet", extra={"output": str(output)})

    def export_to_csv(self, sql: str, output: str | Path) -> None:
        with self.connect() as conn:
            conn.execute(f"COPY ({sql}) TO '{output}' (FORMAT CSV, HEADER)")
        logger.info("Exported query to CSV", extra={"output": str(output)})
