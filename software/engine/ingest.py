"""HLS download via NASA Earthaccess and clear-scene listing."""

from __future__ import annotations

from pathlib import Path

from .hls import (
    FMASK_CLOUD_BITS,
    PRITHVI_BANDS_HLSS30,
    date_id_to_datetime,
    date_id_to_month_key,
    has_required_bands,
    index_hls_directory,
    score_observation,
    select_best_per_month,
)

HLS_DOWNLOAD_BANDS = list(PRITHVI_BANDS_HLSS30) + ["Fmask"]


def best_observation_for_month(
    observations: dict,
    month_key: str,
    row: int,
    col: int,
    patch_size: int = 224,
    min_clear_fraction: float = 0.3,
) -> dict | None:
    monthly = select_best_per_month(
        observations,
        row=row,
        col=col,
        patch_size=patch_size,
        required=PRITHVI_BANDS_HLSS30,
        min_clear_fraction=min_clear_fraction,
        fmask_bits=FMASK_CLOUD_BITS,
    )
    for item in monthly:
        if item["month"] == month_key:
            return item
    return None


def list_clear_observations(
    observations: dict,
    row: int,
    col: int,
    patch_size: int = 224,
    min_clear_fraction: float = 0.5,
) -> list[dict]:
    selected: list[dict] = []
    for date_id, bands in observations.items():
        score = score_observation(
            bands,
            row=row,
            col=col,
            patch_size=patch_size,
            required=PRITHVI_BANDS_HLSS30,
            fmask_bits=FMASK_CLOUD_BITS,
        )
        if score < min_clear_fraction:
            continue
        dt = date_id_to_datetime(date_id)
        selected.append(
            {
                "date_id": date_id,
                "month": date_id_to_month_key(date_id),
                "date_key": dt.strftime("%Y-%m-%d"),
                "clear_fraction": score,
                "bands": bands,
            }
        )
    selected.sort(key=lambda item: item["date_id"])
    return selected


def filter_hls_band_urls(granules, bands: list[str] | None = None) -> list[str]:
    bands = list(bands or HLS_DOWNLOAD_BANDS)
    patterns = [f".{band}.tif" for band in bands]
    urls: list[str] = []
    for granule in granules:
        for link in granule.data_links():
            if any(p in link for p in patterns):
                urls.append(link)
    return urls


def months_with_required_bands(
    hls_dir: Path,
    lon: float | None = None,
    lat: float | None = None,
    patch_size: int = 224,
) -> set[str]:
    observations = index_hls_directory(hls_dir, lon=lon, lat=lat, patch_size=patch_size)
    months: set[str] = set()
    for date_id, bands in observations.items():
        if has_required_bands(bands, required=PRITHVI_BANDS_HLSS30, need_fmask=True):
            months.add(date_id_to_month_key(date_id))
    return months


def _month_windows(temporal_start: str, temporal_end: str) -> list[tuple[str, str, str]]:
    from calendar import monthrange
    from datetime import date

    start = date.fromisoformat(temporal_start[:10])
    end = date.fromisoformat(temporal_end[:10])
    windows: list[tuple[str, str, str]] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        first = date(y, m, 1)
        last = date(y, m, monthrange(y, m)[1])
        w0 = max(first, start).isoformat()
        w1 = min(last, end).isoformat()
        windows.append((f"{y:04d}-{m:02d}", w0, w1))
        if m == 12:
            y, m = y + 1, 1
        else:
            m += 1
    return windows


def granule_cloud_cover(granule) -> float:
    try:
        value = granule["umm"]["CloudCover"]
        if value is not None:
            return float(value)
    except Exception:
        pass
    for attr in ("cloud_cover", "cloud_coverage"):
        try:
            value = getattr(granule, attr)
            if callable(value):
                value = value()
            if value is not None:
                return float(value)
        except Exception:
            continue
    return 100.0


def download_hls_for_location(
    lon: float,
    lat: float,
    hls_dir: Path,
    temporal_start: str,
    temporal_end: str,
    search_count: int = 20,
    bbox_pad_deg: float = 0.05,
    short_name: str = "HLSS30",
    download: bool = True,
    bands: list[str] | None = None,
    only_months: list[str] | None = None,
    max_per_month: int | None = None,
    log=print,
) -> list:
    import earthaccess

    bands = list(bands or HLS_DOWNLOAD_BANDS)
    hls_dir = Path(hls_dir)
    hls_dir.mkdir(parents=True, exist_ok=True)
    earthaccess.login()
    bbox = (
        lon - bbox_pad_deg,
        lat - bbox_pad_deg,
        lon + bbox_pad_deg,
        lat + bbox_pad_deg,
    )
    all_results: list = []
    windows = _month_windows(temporal_start, temporal_end)
    if only_months is not None:
        want = set(only_months)
        windows = [w for w in windows if w[0] in want]
        if not windows:
            log(f"  No month windows to search (only_months={sorted(want)})")
            return []
    for month_key, w0, w1 in windows:
        month_results = earthaccess.search_data(
            short_name=short_name,
            bounding_box=bbox,
            temporal=(w0, w1),
            count=search_count,
        )
        kept = list(month_results)
        if max_per_month is not None and len(kept) > max_per_month:
            kept = sorted(kept, key=granule_cloud_cover)[:max_per_month]
        log(f"  {month_key}: search granules={len(month_results)}")
        all_results.extend(kept)
    if download and all_results:
        urls = list(dict.fromkeys(filter_hls_band_urls(all_results, bands=bands)))
        log(f"  Total granules={len(all_results)} → download files={len(urls)}")
        if urls:
            earthaccess.download(urls, str(hls_dir))
    return all_results
