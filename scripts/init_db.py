"""Bootstrap script to create database tables."""

from __future__ import annotations

from football_advance_predictor.core.logging import configure_logging
from football_advance_predictor.db.session import init_db


def main() -> None:
    configure_logging()
    init_db()
    print("Database schema initialized.")


if __name__ == "__main__":
    main()
