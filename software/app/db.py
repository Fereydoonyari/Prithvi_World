"""SQLAlchemy engine and session. SQLite by default (zero extra services)."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from engine.paths import DB_PATH, DATA_DIR, ensure_project_dirs


class Base(DeclarativeBase):
    pass


def _database_url() -> str:
    import os

    return os.environ.get("WORKBENCH_DATABASE_URL") or f"sqlite:///{DB_PATH.as_posix()}"


ensure_project_dirs()
DATA_DIR.mkdir(parents=True, exist_ok=True)

connect_args = {"check_same_thread": False} if _database_url().startswith("sqlite") else {}
engine = create_engine(_database_url(), connect_args=connect_args, future=True)

if _database_url().startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _sqlite_pragma(dbapi_conn, _rec):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def ensure_schema() -> None:
    """Add columns introduced after the first SQLite create_all()."""
    with engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info(aois)")).fetchall()}
        if cols and "is_shared" not in cols:
            conn.execute(text("ALTER TABLE aois ADD COLUMN is_shared INTEGER NOT NULL DEFAULT 0"))
            conn.execute(text("UPDATE aois SET is_shared = 1 WHERE owner_id IS NULL"))
        conn.execute(text("DROP TABLE IF EXISTS saved_views"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
