from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import settings

DATABASE_URL: str = settings.database_url

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
        # Strip leading scheme so we can rebuild with URI flags.
        # sqlite:///./scout.db  → file:scout.db?mode=ro&immutable=1
        path = DATABASE_URL.split("sqlite:///", 1)[-1]
        ro_uri = f"file:{path}?mode=ro&immutable=1"
        return create_engine(
            f"sqlite:///{ro_uri}",
            connect_args={"uri": True, "check_same_thread": False},
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
