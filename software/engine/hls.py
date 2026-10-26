"""HLS → 224×224 six-band patch extraction and Fmask scoring."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

PRITHVI_BANDS_HLSS30 = ["B02", "B03", "B04", "B8A", "B11", "B12"]

PRITHVI_MEAN = np.array(
    [1087.0, 1342.0, 1433.0, 2734.0, 1958.0, 1363.0],
    dtype=np.float32,
)
PRITHVI_STD = np.array(
    [2248.0, 2179.0, 2178.0, 1850.0, 1242.0, 1049.0],
    dtype=np.float32,
)

FMASK_ALL_QA_BITS = 0b00111111
# Water is not masked: burn scars are often false-flagged as water.
FMASK_CLOUD_BITS = 0b00011111
FMASK_CLEAR_BITS = FMASK_CLOUD_BITS

HLS_FILENAME_PATTERN = re.compile(
    r"HLS\.S30\.(T\d{2}[A-Z]{3})\.(\d{7})T(\d{6})\.v[\d.]+\.([A-Za-z0-9]+)\.tif$"
)

GSD_METERS = 30.0
PATCH_SIZE = 224


def parse_hls_filename(path: str | Path) -> dict:
    name = Path(path).name
    match = HLS_FILENAME_PATTERN.match(name)
    if match is None:
        raise ValueError(f"Could not parse HLS filename: {name}")
    return {
        "tile": match.group(1),
        "date": match.group(2),
        "time": match.group(3),
        "band": match.group(4),
        "path": str(path),
    }


def parse_hls_date(date_id: str) -> tuple[int, int]:
    return int(date_id[:4]), int(date_id[4:])


def date_id_to_datetime(date_id: str) -> datetime:
    year, doy = parse_hls_date(date_id)
    return datetime(year, 1, 1) + timedelta(days=doy - 1)


def date_id_to_month_key(date_id: str) -> str:
    return date_id_to_datetime(date_id).strftime("%Y-%m")


def aoi_pixel_placement(
    raster_path: str | Path,
    lon: float,
    lat: float,
    patch_size: int = PATCH_SIZE,
) -> tuple[int, int, int] | None:
    with rasterio.open(raster_path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        if not (0 <= row < src.height and 0 <= col < src.width):
            return None
        margin = min(row, col, src.height - 1 - row, src.width - 1 - col)
        return int(row), int(col), int(margin)


def index_hls_directory(
    hls_dir: str | Path,
    lon: float | None = None,
    lat: float | None = None,
    patch_size: int = PATCH_SIZE,
) -> dict[str, dict[str, str]]:
    nested: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for path in sorted(Path(hls_dir).glob("*.tif")):
        try:
            info = parse_hls_filename(path)
        except ValueError:
            continue
        nested[info["date"]][info["tile"]][info["band"]] = info["path"]

    observations: dict[str, dict[str, str]] = {}
    half = patch_size // 2
    for date_id, tiles in nested.items():
        if lon is None or lat is None:
            tile = sorted(tiles.keys())[0]
            observations[date_id] = tiles[tile]
            continue
        best_bands: dict[str, str] | None = None
        best_key: tuple[int, int] | None = None
        for _tile, bands in tiles.items():
            ref = bands.get("B02") or next(iter(bands.values()), None)
            if ref is None:
                continue
            place = aoi_pixel_placement(ref, lon, lat, patch_size=patch_size)
            if place is None:
                continue
            _row, _col, margin = place
            key = (1 if margin >= half else 0, margin)
            if best_key is None or key > best_key:
                best_key = key
                best_bands = bands
        if best_bands is not None:
            observations[date_id] = best_bands
    return observations


def resolve_band_path(bands: dict[str, str], band: str) -> str | None:
    if band in bands:
        return bands[band]
    if band == "B8A" and "B08" in bands:
        return bands["B08"]
    return None


def has_required_bands(
    bands: dict[str, str],
    required: Iterable[str] = PRITHVI_BANDS_HLSS30,
    need_fmask: bool = True,
) -> bool:
    required_list = list(required)
    if need_fmask and resolve_band_path(bands, "Fmask") is None:
        return False
    return all(resolve_band_path(bands, band) is not None for band in required_list)


def get_patch_window(row: int, col: int, patch_size: int = PATCH_SIZE) -> Window:
    half = patch_size // 2
    return Window(col - half, row - half, patch_size, patch_size)


def lonlat_to_pixel(
    raster_path: str | Path,
    lon: float,
    lat: float,
) -> tuple[int, int, float, float]:
    with rasterio.open(raster_path) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
        row, col = src.index(xs[0], ys[0])
        if not (0 <= row < src.height and 0 <= col < src.width):
            raise ValueError(
                f"AOI ({lon}, {lat}) falls outside raster {Path(raster_path).name}"
            )
        return int(row), int(col), float(lon), float(lat)


def patch_bounds_wgs84(
    raster_path: str | Path,
    lon: float,
    lat: float,
    patch_size: int = PATCH_SIZE,
) -> tuple[float, float, float, float]:
    """Return (west, south, east, north) of the 224×224 window in WGS84."""
    row, col, _, _ = lonlat_to_pixel(raster_path, lon, lat)
    window = get_patch_window(row, col, patch_size)
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = rasterio.windows.bounds(window, src.transform)
        xs, ys = warp_transform(src.crs, "EPSG:4326", [left, right], [bottom, top])
    west, east = min(xs), max(xs)
    south, north = min(ys), max(ys)
    return west, south, east, north


def approximate_bounds_wgs84(
    lon: float,
    lat: float,
    patch_size: int = PATCH_SIZE,
    gsd_m: float = GSD_METERS,
) -> tuple[float, float, float, float]:
    """Footprint when no GeoTIFF is on disk yet. 224 px × 30 m ≈ 6.7 km."""
    import math

    half_m = (patch_size / 2.0) * gsd_m
    dlat = half_m / 111_320.0
    dlon = half_m / (111_320.0 * max(math.cos(math.radians(lat)), 0.1))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def read_window(
    path: str | Path,
    window: Window,
    nodata_to_nan: bool = True,
) -> np.ndarray:
    with rasterio.open(path) as src:
        array = src.read(1, window=window).astype(np.float32)
        if nodata_to_nan and src.nodata is not None:
            array[array == src.nodata] = np.nan
    return array


def clear_fraction_fmask(
    fmask_path: str | Path,
    row: int,
    col: int,
    patch_size: int = PATCH_SIZE,
    fmask_bits: int = FMASK_CLEAR_BITS,
) -> float:
    window = get_patch_window(row, col, patch_size)
    with rasterio.open(fmask_path) as src:
        fmask = src.read(1, window=window)
    if fmask.size == 0:
        return 0.0
    clear = (fmask & fmask_bits) == 0
    return float(np.mean(clear))


def score_observation(
    bands: dict[str, str],
    row: int,
    col: int,
    patch_size: int = PATCH_SIZE,
    required: Iterable[str] = PRITHVI_BANDS_HLSS30,
    fmask_bits: int = FMASK_CLEAR_BITS,
) -> float:
    if not has_required_bands(bands, required=required, need_fmask=True):
        return -1.0
    fmask_path = resolve_band_path(bands, "Fmask")
    assert fmask_path is not None
    return clear_fraction_fmask(fmask_path, row, col, patch_size, fmask_bits=fmask_bits)


def select_best_per_month(
    observations: dict[str, dict[str, str]],
    row: int,
    col: int,
    patch_size: int = PATCH_SIZE,
    required: Iterable[str] = PRITHVI_BANDS_HLSS30,
    min_clear_fraction: float = 0.0,
    fmask_bits: int = FMASK_CLEAR_BITS,
) -> list[dict]:
    by_month: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for date_id, bands in observations.items():
        score = score_observation(
            bands, row, col, patch_size=patch_size, required=required, fmask_bits=fmask_bits
        )
        if score < min_clear_fraction:
            continue
        by_month[date_id_to_month_key(date_id)].append((date_id, score))
    selected: list[dict] = []
    for month in sorted(by_month):
        date_id, score = max(by_month[month], key=lambda item: item[1])
        selected.append(
            {
                "date_id": date_id,
                "month": month,
                "clear_fraction": score,
                "bands": observations[date_id],
            }
        )
    return selected


def load_observation(
    bands: dict[str, str],
    row: int,
    col: int,
    patch_size: int = PATCH_SIZE,
    band_names: Iterable[str] = PRITHVI_BANDS_HLSS30,
    apply_fmask: bool = True,
    fmask_bits: int = FMASK_CLEAR_BITS,
) -> np.ndarray:
    window = get_patch_window(row, col, patch_size)
    arrays = []
    for band in band_names:
        path = resolve_band_path(bands, band)
        if path is None:
            raise FileNotFoundError(f"Missing band {band} for observation")
        arrays.append(read_window(path, window, nodata_to_nan=True))
    image = np.stack(arrays, axis=0)
    fmask_path = resolve_band_path(bands, "Fmask")
    if apply_fmask and fmask_path is not None:
        with rasterio.open(fmask_path) as src:
            fmask = src.read(1, window=window)
        invalid = (fmask & fmask_bits) != 0
        image[:, invalid] = np.nan
    return image


def normalize_prithvi(patch: np.ndarray) -> np.ndarray:
    patch = np.nan_to_num(patch.astype(np.float32), nan=0.0001)
    normalized = (patch - PRITHVI_MEAN[:, None, None]) / PRITHVI_STD[:, None, None]
    return normalized.astype(np.float32)


def make_rgb_display(patch: np.ndarray) -> np.ndarray:
    rgb = np.stack([patch[2], patch[1], patch[0]], axis=-1)
    low = np.nanpercentile(rgb, 2)
    high = np.nanpercentile(rgb, 98)
    rgb = (rgb - low) / (high - low + 1e-6)
    return np.clip(rgb, 0, 1)
