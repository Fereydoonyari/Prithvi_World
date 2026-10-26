"""PNG helpers for change maps, vegetation heatmaps, and decoder RGB."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def _magma_lut() -> np.ndarray:
    """256×3 magma-like LUT so overlays work without matplotlib."""
    t = np.linspace(0, 1, 256, dtype=np.float32)
    r = np.clip(1.7 * t - 0.15 * np.sin(2 * np.pi * t), 0, 1)
    g = np.clip(1.4 * (t**1.4) * (1.1 - t), 0, 1)
    b = np.clip(0.35 + 1.4 * t * (1 - t) ** 0.4, 0, 1)
    rgb = np.stack([r, g * 0.55 + 0.05, b * (0.6 + 0.4 * t)], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def _diverging_lut() -> np.ndarray:
    """Brown → cream → green. Negative = vegetation loss, positive = gain."""
    t = np.linspace(0, 1, 256, dtype=np.float32)
    brown = np.array([0.48, 0.22, 0.10], dtype=np.float32)
    cream = np.array([0.94, 0.91, 0.84], dtype=np.float32)
    green = np.array([0.18, 0.46, 0.22], dtype=np.float32)
    rgb = np.empty((256, 3), dtype=np.float32)
    lo = t < 0.5
    u = np.where(lo, t * 2.0, (t - 0.5) * 2.0)
    rgb[lo] = (1.0 - u[lo, None]) * brown + u[lo, None] * cream
    rgb[~lo] = (1.0 - u[~lo, None]) * cream + u[~lo, None] * green
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _ndvi_lut() -> np.ndarray:
    """Brown → yellow → green for absolute NDVI in [-1, 1] (shown clipped to [0, 1])."""
    t = np.linspace(0, 1, 256, dtype=np.float32)
    brown = np.array([0.42, 0.24, 0.12], dtype=np.float32)
    yellow = np.array([0.90, 0.82, 0.35], dtype=np.float32)
    green = np.array([0.12, 0.48, 0.20], dtype=np.float32)
    rgb = np.empty((256, 3), dtype=np.float32)
    lo = t < 0.5
    u = np.where(lo, t * 2.0, (t - 0.5) * 2.0)
    rgb[lo] = (1.0 - u[lo, None]) * brown + u[lo, None] * yellow
    rgb[~lo] = (1.0 - u[~lo, None]) * yellow + u[~lo, None] * green
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


_LUT = _magma_lut()
_DIV = _diverging_lut()
_NDVI = _ndvi_lut()


def array_to_rgba(grid: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32)
    finite = np.isfinite(arr)
    if vmin is None:
        vmin = float(np.nanpercentile(arr[finite], 2)) if finite.any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(arr[finite], 98)) if finite.any() else 1.0
    vmax = max(vmax, vmin + 1e-6)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    idx = (norm * 255).astype(np.uint8)
    rgb = _LUT[idx]
    alpha = np.where(finite, 210, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def array_to_rgba_diverging(grid: np.ndarray, vmax: float | None = None) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32)
    finite = np.isfinite(arr)
    if vmax is None:
        if finite.any():
            vmax = float(
                max(
                    abs(np.nanpercentile(arr[finite], 2)),
                    abs(np.nanpercentile(arr[finite], 98)),
                    0.02,
                )
            )
        else:
            vmax = 0.02
    vmax = max(vmax, 1e-6)
    norm = np.clip((arr + vmax) / (2.0 * vmax), 0, 1)
    idx = (norm * 255).astype(np.uint8)
    rgb = _DIV[idx]
    alpha = np.where(finite, 215, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def array_to_rgba_ndvi(grid: np.ndarray, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32)
    finite = np.isfinite(arr)
    vmax = max(float(vmax), float(vmin) + 1e-6)
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    idx = (norm * 255).astype(np.uint8)
    rgb = _NDVI[idx]
    alpha = np.where(finite, 220, 0).astype(np.uint8)
    return np.dstack([rgb, alpha])


def save_overlay_png(
    grid: np.ndarray,
    path: Path,
    upsample: int = 16,
    cmap: str = "magma",
    resample: str = "bilinear",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cmap == "diverging":
        rgba = array_to_rgba_diverging(grid)
    elif cmap == "ndvi":
        rgba = array_to_rgba_ndvi(grid)
    else:
        rgba = array_to_rgba(grid)
    if Image is None:
        raise RuntimeError("Pillow is required to write overlay PNGs")
    img = Image.fromarray(rgba, mode="RGBA")
    if upsample > 1:
        mode = Image.NEAREST if resample == "nearest" else Image.BILINEAR
        img = img.resize(
            (rgba.shape[1] * upsample, rgba.shape[0] * upsample),
            resample=mode,
        )
    img.save(path)
    return path


def save_rgb_png(rgb: np.ndarray, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if Image is None:
        raise RuntimeError("Pillow is required to write RGB PNGs")
    arr = np.clip(np.asarray(rgb, dtype=np.float32), 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
    img.save(path)
    return path
