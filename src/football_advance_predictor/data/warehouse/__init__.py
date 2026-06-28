"""DuckDB-based warehouse helpers for analytical feature generation.

The warehouse is intentionally separate from the application database.
It exists for offline feature engineering on Parquet/CSV exports.
"""

from football_advance_predictor.data.warehouse.duckdb_warehouse import DuckDBWarehouse

__all__ = ["DuckDBWarehouse"]
