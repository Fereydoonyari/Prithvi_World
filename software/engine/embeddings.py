"""Frozen Prithvi forward pass and embedding I/O (tokens + CLS + metadata)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .hls import (
    FMASK_CLOUD_BITS,
    PRITHVI_BANDS_HLSS30,
    load_observation,
    normalize_prithvi,
    parse_hls_date,
    parse_hls_filename,
    resolve_band_path,
)
from .paths import EMBEDDINGS_DIR, location_embedding_month_dir


def _embeddings_root(embeddings_root: Path | None = None) -> Path:
    return Path(embeddings_root) if embeddings_root is not None else EMBEDDINGS_DIR


def embedding_paths(
    location_id: str,
    month_key: str,
    embeddings_root: Path | None = None,
) -> dict[str, Path]:
    d = location_embedding_month_dir(
        location_id, month_key, embeddings_root=_embeddings_root(embeddings_root)
    )
    return {
        "dir": d,
        "tokens": d / "tokens.npy",
        "cls": d / "cls.npy",
        "metadata": d / "metadata.json",
    }


def list_embedding_keys(
    location_id: str,
    embeddings_root: Path | None = None,
) -> list[str]:
    root = _embeddings_root(embeddings_root) / location_id
    if not root.exists():
        return []
    keys = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "tokens.npy").exists() and (child / "cls.npy").exists():
            keys.append(child.name)
    return keys


def load_location_sequence(
    location_id: str,
    embeddings_root: Path | None = None,
) -> list[dict[str, Any]]:
    seq = []
    for key in list_embedding_keys(location_id, embeddings_root=embeddings_root):
        item = load_embedding(location_id, key, embeddings_root=embeddings_root)
        item["key"] = key
        seq.append(item)
    return seq


def embeddings_exist(
    location_id: str,
    month_key: str,
    embeddings_root: Path | None = None,
) -> bool:
    p = embedding_paths(location_id, month_key, embeddings_root=embeddings_root)
    return p["tokens"].exists() and p["cls"].exists() and p["metadata"].exists()


@torch.no_grad()
def extract_single_date_embedding(
    model,
    device: torch.device,
    bands: dict[str, str],
    row: int,
    col: int,
    lat: float,
    lon: float,
    date_id: str,
    patch_size: int = 224,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = load_observation(
        bands,
        row=row,
        col=col,
        patch_size=patch_size,
        band_names=PRITHVI_BANDS_HLSS30,
        apply_fmask=True,
        fmask_bits=FMASK_CLOUD_BITS,
    )
    x = torch.from_numpy(normalize_prithvi(raw)).unsqueeze(1).unsqueeze(0).to(device)
    year, doy = parse_hls_date(date_id)
    tcoords = torch.tensor([[[year, doy]]], dtype=torch.float32, device=device)
    lcoords = torch.tensor([[lat, lon]], dtype=torch.float32, device=device)
    feats = model.forward_features(x, tcoords, lcoords)
    latent = feats[-1]
    cls = latent[:, 0, :].detach().cpu().numpy()[0]
    grid = patch_size // 16
    tokens = (
        latent[:, 1:, :]
        .reshape(1, 1, grid, grid, -1)
        .detach()
        .cpu()
        .numpy()[0, 0]
    )
    return tokens.astype(np.float32), cls.astype(np.float32), raw


def save_embedding(
    location_id: str,
    month_key: str,
    tokens: np.ndarray,
    cls: np.ndarray,
    metadata: dict[str, Any],
    embeddings_root: Path | None = None,
) -> dict[str, Path]:
    paths = embedding_paths(location_id, month_key, embeddings_root=embeddings_root)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    if tokens.shape != (14, 14, 768):
        raise ValueError(f"Expected tokens (14,14,768), got {tokens.shape}")
    if cls.shape != (768,):
        raise ValueError(f"Expected cls (768,), got {cls.shape}")
    np.save(paths["tokens"], tokens)
    np.save(paths["cls"], cls)
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return paths


def load_embedding(
    location_id: str,
    month_key: str,
    embeddings_root: Path | None = None,
) -> dict[str, Any]:
    paths = embedding_paths(location_id, month_key, embeddings_root=embeddings_root)
    for key in ("tokens", "cls", "metadata"):
        if not paths[key].exists():
            raise FileNotFoundError(f"Missing embedding artifact: {paths[key]}")
    with open(paths["metadata"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    return {
        "tokens": np.load(paths["tokens"]),
        "cls": np.load(paths["cls"]),
        "metadata": meta,
        "paths": paths,
    }


def build_metadata(
    *,
    location_id: str,
    location_name: str,
    latitude: float,
    longitude: float,
    month_key: str,
    date_id: str,
    clear_fraction: float,
    model_name: str,
    device: str,
    bands_dict: dict[str, str],
    tile: str | None = None,
    center_row: int | None = None,
    center_col: int | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    source_files = []
    for band in PRITHVI_BANDS_HLSS30 + ["Fmask"]:
        p = resolve_band_path(bands_dict, band)
        if p:
            source_files.append(Path(p).name)
    if tile is None:
        b02 = resolve_band_path(bands_dict, "B02")
        tile = parse_hls_filename(b02)["tile"] if b02 else "unknown"
    meta = {
        "location_id": location_id,
        "location_name": location_name,
        "latitude": latitude,
        "longitude": longitude,
        "date": month_key,
        "acquisition_date_id": date_id,
        "clear_fraction": clear_fraction,
        "hls_tile": tile,
        "model_name": model_name,
        "embedding_shape": [14, 14, 768],
        "cls_shape": [768],
        "bands": list(PRITHVI_BANDS_HLSS30),
        "preprocessing": (
            "HLSS30 B02,B03,B04,B8A,B11,B12; nodata→NaN; "
            "FMASK_CLOUD_BITS (no water bit); nan→0.0001; Prithvi mean/std; "
            "single-frame forward_features"
        ),
        "device": device,
        "source_files": source_files,
        "center_row": center_row,
        "center_col": center_col,
        "patch_size": 224,
        "num_frames": 1,
    }
    if extra:
        meta.update(extra)
    return meta
