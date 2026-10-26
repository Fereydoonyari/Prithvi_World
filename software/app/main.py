"""PrithviWORLD Workbench — FastAPI entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from engine.paths import OVERLAYS_DIR, SOFTWARE_ROOT, ensure_project_dirs

from .api import api
from .db import Base, SessionLocal, engine, ensure_schema
from .seed import seed_if_empty
from .services import start_worker, stop_worker

FRONTEND = SOFTWARE_ROOT / "frontend"


def _load_dotenv() -> None:
    """Load software/.env into os.environ without overriding existing vars."""
    import os

    path = SOFTWARE_ROOT / ".env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = val


_load_dotenv()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    ensure_project_dirs()
    Base.metadata.create_all(bind=engine)
    ensure_schema()
    db = SessionLocal()
    try:
        seed_if_empty(db)
    finally:
        db.close()
    start_worker()
    yield
    stop_worker()


app = FastAPI(title="PrithviWORLD Workbench", lifespan=lifespan)
app.include_router(api)

OVERLAYS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/media/overlays", StaticFiles(directory=str(OVERLAYS_DIR)), name="overlays")
app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
