"""Application settings loaded from environment variables.

All environment variables are read via pydantic-settings. Sensible local
defaults are provided for development so the demo can run without a real
``.env`` file, but secrets should never be hardcoded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Top-level application settings.

    Values are loaded from the environment (and optional ``.env`` file).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # PostgreSQL
    postgres_host: str = Field(default="localhost")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="football")
    postgres_user: str = Field(default="football")
    postgres_password: str = Field(default="football")

    # DuckDB warehouse
    duckdb_path: Path = Field(default=Path("./data/warehouse.duckdb"))

    # Model registry
    model_registry_dir: Path = Field(default=Path("./data/processed/models"))

    # Backtest report output directory
    reports_dir: Path = Field(default=Path("./reports"))

    # Global deterministic seed
    global_random_seed: int = Field(default=42)

    # Optional external provider credentials (NEVER hardcode elsewhere)
    external_odds_api_key: str = Field(default="")
    external_footballdata_token: str = Field(default="")
    external_odds_base_url: str = Field(default="")
    external_footballdata_base_url: str = Field(
        default="https://api.football-data.org"
    )

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> AppSettings:
    """Return a memoized settings instance.

    Returns:
        AppSettings: The cached settings object.
    """
    return AppSettings()
