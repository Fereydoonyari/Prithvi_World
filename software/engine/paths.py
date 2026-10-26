"""Filesystem layout for the PrithviWORLD Workbench.

This package is self-contained. It does not import anything outside this package.
All data lives under ``software/data/``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SOFTWARE_ROOT = Path(__file__).resolve().parents[1]
if str(SOFTWARE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOFTWARE_ROOT))

REPO_ROOT = SOFTWARE_ROOT
CONFIG_PATH = SOFTWARE_ROOT / "config" / "default.json"
DATA_DIR = SOFTWARE_ROOT / "data"
HLS_DATA_DIR = DATA_DIR / "hls"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = DATA_DIR / "results"
MODEL_DIR = DATA_DIR / "models" / "prithvi"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
OVERLAYS_DIR = RESULTS_DIR / "overlays"
DB_PATH = DATA_DIR / "workbench.db"


def location_hls_dir(location_id: str, data_root: Path | None = None) -> Path:
    root = Path(data_root) if data_root is not None else HLS_DATA_DIR
    return root / location_id / "HLS"


def location_embedding_month_dir(
    location_id: str,
    month_key: str,
    embeddings_root: Path | None = None,
) -> Path:
    root = Path(embeddings_root) if embeddings_root is not None else EMBEDDINGS_DIR
    return root / location_id / month_key


def load_config(path: Path | None = None) -> dict:
    path = path or CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: Path) -> None:
    if path.is_dir() or os.path.lexists(path):
        return
    try:
        path.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        return


def ensure_project_dirs(location_ids: list[str] | None = None) -> None:
    for sub in [
        HLS_DATA_DIR,
        EMBEDDINGS_DIR,
        RESULTS_DIR,
        MODEL_DIR,
        CHECKPOINTS_DIR,
        OVERLAYS_DIR,
        DATA_DIR / "uploads",
    ]:
        _ensure_dir(sub)
    if location_ids:
        for loc in location_ids:
            _ensure_dir(HLS_DATA_DIR / loc / "HLS")
            _ensure_dir(EMBEDDINGS_DIR / loc)
