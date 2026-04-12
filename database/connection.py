import pathlib
import sqlite3
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import settings

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _resolve_database_url(url: str) -> str:
    """Turn a relative SQLite path into an absolute one.

    ``sqlite:///./scout.db`` resolves relative to the *process* cwd, which
    differs between local dev and Render.  Anchoring to the project root
    makes the path reliable everywhere.
    """
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        rel = url.split("sqlite:///", 1)[-1]
        abs_path = (_PROJECT_ROOT / rel).resolve()
        return f"sqlite:///{abs_path}"
    return url


DATABASE_URL: str = _resolve_database_url(settings.database_url)

# SQLite needs check_same_thread=False; ignored by PostgreSQL
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def _make_readonly_engine():
    """Return a read-only engine for the SQL agent.

    For SQLite, opens the file in immutable URI mode so any DML (INSERT/UPDATE/
    DELETE/DROP) is rejected at the driver level — a hard guarantee regardless of
    what the LLM produces.  For other databases, returns the normal engine; access
    controls should be enforced at the DB level instead.
    """
    if DATABASE_URL.startswith("sqlite"):
        # Build a file: URI with read-only + immutable flags and hand it
        # to sqlite3 directly via a creator function.  This avoids
        # SQLAlchemy parsing the ?mode=ro&immutable=1 query string as
        # its own parameters instead of passing them to the driver.
        path = DATABASE_URL.split("sqlite:///", 1)[-1]
        ro_uri = f"file:{path}?mode=ro&immutable=1"
        return create_engine(
            "sqlite://",
            creator=lambda: sqlite3.connect(ro_uri, uri=True, check_same_thread=False),
        )
    return engine


readonly_engine = _make_readonly_engine()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables (idempotent — skips tables that already exist).

    Called by ``seed_all()`` at the start of every seed run to ensure the
    schema is present before data is loaded.  The API never calls this —
    it relies on the pre-seeded ``scout.db`` that ships with the repo.
    """
    from database.models import Base  # noqa: F401 — ensures models are registered
    Base.metadata.create_all(bind=engine)
