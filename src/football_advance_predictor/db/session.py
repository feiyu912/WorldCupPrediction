"""Database engine, session factory, and bootstrap utilities."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from football_advance_predictor.core.config import get_settings
from football_advance_predictor.core.logging import get_logger

if TYPE_CHECKING:
    from football_advance_predictor.db.base import Base

logger = get_logger(__name__)


_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_engine(*, echo: bool = False) -> Engine:
    """Return a memoized SQLAlchemy engine.

    Args:
        echo: Whether to echo SQL statements.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        logger.info("Creating database engine", host=settings.postgres_host, db=settings.postgres_db)
        _engine = create_engine(settings.database_url, echo=echo, future=True, pool_pre_ping=True)
    return _engine


def _session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, autoflush=False)
    return _SessionLocal


def get_session() -> Iterator[Session]:
    """FastAPI-style dependency that yields a database session."""
    SessionLocal = _session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context manager for transactional scope.

    Commits on success, rolls back on exception.
    """
    SessionLocal = _session_factory()
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(base: type[Base] | None = None) -> None:
    """Create all tables for the given declarative base.

    In production, prefer Alembic migrations. This helper is used by tests
    and the ``scripts/init_db.py`` bootstrap.
    """
    if base is None:
        from football_advance_predictor.db.base import Base as _Base

        base = _Base
    base.metadata.create_all(bind=get_engine())
    logger.info("Initialized database schema")
