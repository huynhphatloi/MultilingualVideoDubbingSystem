"""SQLAlchemy engine/session wiring.

The engine is built **lazily**. Creating it at import time would resolve the
DBAPI immediately, which means importing anything under ``app.jobs`` would
require psycopg to be installed and ``DATABASE_URL`` to be valid - painful for
tests and for tooling that only wants the models.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def _build_engine(url: str) -> Engine:
    kwargs: dict = {"pool_pre_ping": True, "future": True}
    if not url.startswith("sqlite"):
        # SQLite's dialect rejects the pooling knobs entirely.
        kwargs |= {"pool_size": 10, "max_overflow": 20, "pool_recycle": 1800}
    return create_engine(url, **kwargs)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _build_engine(settings.database_url)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(), autoflush=False, expire_on_commit=False, future=True
        )
    return _session_factory


def configure(engine: Engine) -> None:
    """Point the whole app at a different engine (used by the test-suite)."""
    global _engine, _session_factory
    _engine = engine
    _session_factory = sessionmaker(
        bind=engine, autoflush=False, expire_on_commit=False, future=True
    )


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope. Commits on success, rolls back on error."""
    session = get_session_factory()()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Iterator[Session]:
    """FastAPI dependency."""
    with session_scope() as session:
        yield session


def init_db(retries: int = 10, delay: float = 3.0) -> None:
    """Create tables, tolerating a Postgres that is still booting."""
    import time

    from app.jobs import models  # noqa: F401  (register mappers)

    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            Base.metadata.create_all(bind=get_engine())
            log.info("database schema ready")
            return
        except Exception as exc:  # pragma: no cover
            last = exc
            log.warning("db not ready (attempt %d/%d): %s", attempt, retries, exc)
            time.sleep(delay)
    raise RuntimeError(f"Could not initialise database: {last}")
