"""NDVI / NBR on the same 14×14 grid as Prithvi tokens."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from .hls import (
    FMASK_CLOUD_BITS,
    index_hls_directory,
    load_observation,
    lonlat_to_pixel,
)
from .similarity import cosine_similarity_map

INDEX_BANDS: dict[str, tuple[str, str]] = {
    "ndvi": ("B04", "B8A"),
    "nbr": ("B12", "B8A"),
}


def index_patch(
    bands: dict[str, str],
    row: int,
    col: int,
    kind: str = "ndvi",
    patch_size: int = 224,
    apply_fmask: bool = True,
) -> np.ndarray:
    if kind not in INDEX_BANDS:
        raise ValueError(f"unknown index {kind!r}; expected one of {sorted(INDEX_BANDS)}")
    other, nir_band = INDEX_BANDS[kind]
    arr = load_observation(
        bands,
        row=row,
        col=col,
        patch_size=patch_size,
        band_names=[other, nir_band],
        apply_fmask=apply_fmask,
        fmask_bits=FMASK_CLOUD_BITS,
    )
    other_arr, nir = arr[0], arr[1]
    denom = nir + other_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = (nir - other_arr) / denom
    idx = np.where(np.abs(denom) < 1e-6, np.nan, idx)
    return np.clip(idx, -1.0, 1.0).astype(np.float32)


def ndvi_patch(bands, row, col, patch_size: int = 224, apply_fmask: bool = True) -> np.ndarray:
    return index_patch(bands, row, col, kind="ndvi", patch_size=patch_size, apply_fmask=apply_fmask)


def ndvi_from_sixband_patch(patch: np.ndarray) -> np.ndarray:
    """NDVI from a Prithvi-ordered 6-band patch (B02,B03,B04,B8A,B11,B12)."""
    arr = np.asarray(patch, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] < 4:
        raise ValueError(f"Expected (6,H,W) Prithvi patch, got {arr.shape}")
    red, nir = arr[2], arr[3]  # B04, B8A
    denom = nir + red
    with np.errstate(invalid="ignore", divide="ignore"):
        idx = (nir - red) / denom
    idx = np.where(np.abs(denom) < 1e-6, np.nan, idx)
    return np.clip(idx, -1.0, 1.0).astype(np.float32)


def index_cell_grid(values: np.ndarray, grid: int = 14) -> np.ndarray:
    h, w = values.shape
    ph, pw = h // grid, w // grid
    cells = values[: ph * grid, : pw * grid].reshape(grid, ph, grid, pw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmean(cells, axis=(1, 3))
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def index_grids_for_location(
    lon: float,
    lat: float,
    hls_dir: Path,
    date_ids: list[str],
    kind: str = "ndvi",
    patch_size: int = 224,
    grid: int = 14,
    cache_path: Path | None = None,
    force: bool = False,
) -> dict[str, np.ndarray]:
    cache_path = Path(cache_path) if cache_path is not None else None
    cached: dict[str, np.ndarray] = {}
    if cache_path is not None and cache_path.exists() and not force:
        with np.load(cache_path) as z:
            cached = {k: z[k] for k in z.files}
        if all(d in cached for d in date_ids):
            return {d: cached[d] for d in date_ids}
    observations = index_hls_directory(hls_dir, lon=lon, lat=lat, patch_size=patch_size)
    if not observations:
        return cached
    first_bands = next(iter(observations.values()))
    row, col, _, _ = lonlat_to_pixel(next(iter(first_bands.values())), lon, lat)
    out = dict(cached)
    for date_id in date_ids:
        if date_id in out and not force:
            continue
        bands = observations.get(date_id)
        if not bands:
            continue
        try:
            out[date_id] = index_cell_grid(
                index_patch(bands, row, col, kind=kind, patch_size=patch_size), grid=grid
            )
        except (FileNotFoundError, ValueError):
            continue
    if cache_path is not None and out:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **out)
    return out


def ndvi_grids_for_location(
    lon: float,
    lat: float,
    hls_dir: Path,
    date_ids: list[str],
    **kwargs,
) -> dict[str, np.ndarray]:
    return index_grids_for_location(lon, lat, hls_dir, date_ids, kind="ndvi", **kwargs)


def token_change_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - cosine_similarity_map(a, b)


def ndvi_change_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32))


def ndvi_signed_change_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """To minus From: positive = greening, negative = vegetation loss."""
    return np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)


def map_corr(x: np.ndarray, y: np.ndarray) -> float:
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def skill_score(model_mse: float, climatology_mse: float) -> float:
    if climatology_mse <= 0:
        return float("nan")
    return 1.0 - (model_mse / climatology_mse)
