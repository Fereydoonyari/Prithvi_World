"""Pipeline jobs and spatial overlay composer (workbench services)."""

from __future__ import annotations

import json
import traceback
from datetime import date, datetime
from pathlib import Path
from threading import Event, Thread

import numpy as np
import torch
from sqlalchemy.orm import Session

from engine.dynamics import (
    ConvLSTMForecaster,
    LinearDOYForecaster,
    TokenLSTMForecaster,
    baseline_rows,
    build_forecast_samples,
    nuisance_floor,
    run_seeds,
    split_samples_by_year,
    summarize_seed_runs,
)
from engine.embeddings import (
    build_metadata,
    extract_single_date_embedding,
    list_embedding_keys,
    load_location_sequence,
    save_embedding,
)
from engine.hls import (
    FMASK_CLOUD_BITS,
    approximate_bounds_wgs84,
    date_id_to_datetime,
    index_hls_directory,
    load_observation,
    lonlat_to_pixel,
    make_rgb_display,
    parse_hls_date,
    parse_hls_filename,
    patch_bounds_wgs84,
    score_observation,
)
from engine.ingest import download_hls_for_location, list_clear_observations
from engine.ndvi import ndvi_from_sixband_patch, ndvi_patch, ndvi_signed_change_map, token_change_map
from engine.paths import (
    CHECKPOINTS_DIR,
    OVERLAYS_DIR,
    location_hls_dir,
)
from engine.prithvi import denormalize_prithvi, decode_tokens_to_patch, get_device, load_single_frame_prithvi
from engine.s4d import TokenS4DForecaster
from engine.similarity import cosine_similarity_map
from engine.visualization import save_overlay_png, save_rgb_png

from .db import SessionLocal
from .models import AOI, Embedding, Experiment, Job, Observation, ResultMetric

_STOP = Event()
_WORKER: Thread | None = None
_PRITHVI = None

AVAILABLE_MODELS = [
    {"id": "linear", "label": "Linear + day-of-year"},
    {"id": "token_lstm", "label": "Token LSTM"},
    {"id": "s4d", "label": "S4D"},
    {"id": "convlstm", "label": "ConvLSTM"},
]


def _cached_prithvi():
    global _PRITHVI
    if _PRITHVI is None:
        _PRITHVI = load_single_frame_prithvi()
    return _PRITHVI


def experiment_ckpt_path(exp_id: int, model_name: str) -> Path:
    return CHECKPOINTS_DIR / f"exp{exp_id}" / f"{model_name}_seed0.pt"


def enqueue(db: Session, *, kind: str, user_id: int | None, aoi_id=None, experiment_id=None, payload=None) -> Job:
    job = Job(
        kind=kind,
        status="queued",
        aoi_id=aoi_id,
        experiment_id=experiment_id,
        payload_json=json.dumps(payload or {}),
        created_by=user_id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def append_log(db: Session, job: Job, line: str) -> None:
    job.log_text = (job.log_text or "") + line + "\n"
    db.add(job)
    db.commit()


def _aoi(db: Session, aoi_id: int) -> AOI:
    aoi = db.get(AOI, aoi_id)
    if aoi is None:
        raise RuntimeError(f"AOI {aoi_id} not found")
    return aoi


def run_index_observations(db: Session, job: Job) -> None:
    aoi = _aoi(db, job.aoi_id)
    payload = json.loads(job.payload_json or "{}")
    min_clear = float(payload.get("min_clear", 0.0))
    hls_dir = location_hls_dir(aoi.slug)
    observations = index_hls_directory(hls_dir, lon=aoi.longitude, lat=aoi.latitude)
    if not observations:
        append_log(db, job, f"No HLS GeoTIFFs in {hls_dir}")
        return
    first = next(iter(observations.values()))
    ref = next(iter(first.values()))
    row, col, _, _ = lonlat_to_pixel(ref, aoi.longitude, aoi.latitude)
    n = 0
    for date_id, bands in observations.items():
        score = score_observation(bands, row, col, fmask_bits=FMASK_CLOUD_BITS)
        if score < min_clear:
            continue
        dt = date_id_to_datetime(date_id)
        date_key = dt.strftime("%Y-%m-%d")
        existing = (
            db.query(Observation)
            .filter(Observation.aoi_id == aoi.id, Observation.date_key == date_key)
            .one_or_none()
        )
        granule = Path(next(iter(bands.values()))).name
        if existing is None:
            db.add(
                Observation(
                    aoi_id=aoi.id,
                    date_id=date_id,
                    date_key=date_key,
                    granule=granule,
                    clear_fraction=score,
                    status="accepted" if score >= 0.90 else "indexed",
                    hls_dir=str(hls_dir),
                    ref_band_path=str(bands.get("B02") or next(iter(bands.values()))),
                )
            )
            n += 1
        else:
            existing.clear_fraction = score
            existing.granule = granule
            existing.ref_band_path = str(bands.get("B02") or next(iter(bands.values())))
    db.commit()
    append_log(db, job, f"Indexed {n} new observations ({len(observations)} granules on disk)")


def run_ingest_hls(db: Session, job: Job) -> None:
    aoi = _aoi(db, job.aoi_id)
    payload = json.loads(job.payload_json or "{}")
    hls_dir = location_hls_dir(aoi.slug)
    download_hls_for_location(
        lon=aoi.longitude,
        lat=aoi.latitude,
        hls_dir=hls_dir,
        temporal_start=payload.get("start", "2022-01-01"),
        temporal_end=payload.get("end", "2024-12-31"),
        log=lambda msg: append_log(db, job, str(msg)),
    )
    run_index_observations(db, job)


def run_encode(db: Session, job: Job) -> None:
    aoi = _aoi(db, job.aoi_id)
    payload = json.loads(job.payload_json or "{}")
    min_clear = float(payload.get("min_clear", 0.90))
    hls_dir = location_hls_dir(aoi.slug)
    observations = index_hls_directory(hls_dir, lon=aoi.longitude, lat=aoi.latitude)
    if not observations:
        raise RuntimeError(f"No HLS in {hls_dir}")
    first = next(iter(observations.values()))
    row, col, _, _ = lonlat_to_pixel(next(iter(first.values())), aoi.longitude, aoi.latitude)
    clear = list_clear_observations(observations, row, col, min_clear_fraction=min_clear)
    bundle = load_single_frame_prithvi()
    model, device = bundle["model"], bundle["device"]
    n = 0
    for item in clear:
        date_key = item["date_key"]
        if date_key in list_embedding_keys(aoi.slug):
            continue
        tokens, cls, _raw = extract_single_date_embedding(
            model, device, item["bands"], row, col, aoi.latitude, aoi.longitude, item["date_id"]
        )
        meta = build_metadata(
            location_id=aoi.slug,
            location_name=aoi.name,
            latitude=aoi.latitude,
            longitude=aoi.longitude,
            month_key=date_key,
            date_id=item["date_id"],
            clear_fraction=item["clear_fraction"],
            model_name=str(bundle["repo_id"]),
            device=str(device),
            bands_dict=item["bands"],
            center_row=row,
            center_col=col,
        )
        paths = save_embedding(aoi.slug, date_key, tokens, cls, meta)
        obs = (
            db.query(Observation)
            .filter(Observation.aoi_id == aoi.id, Observation.date_key == date_key)
            .one_or_none()
        )
        db.add(
            Embedding(
                aoi_id=aoi.id,
                observation_id=None if obs is None else obs.id,
                date_key=date_key,
                tokens_path=str(paths["tokens"]),
                cls_path=str(paths["cls"]),
                metadata_path=str(paths["metadata"]),
            )
        )
        n += 1
        append_log(db, job, f"encoded {date_key}")
        db.commit()
    append_log(db, job, f"Encoded {n} new dates")


def _model_factory(name: str):
    name = name.strip().lower()
    if name in ("linear", "linear_doy"):
        return lambda: LinearDOYForecaster()
    if name in ("token_lstm", "lstm"):
        return lambda: TokenLSTMForecaster()
    if name in ("s4d",):
        return lambda: TokenS4DForecaster()
    if name in ("convlstm", "conv_lstm"):
        return lambda: ConvLSTMForecaster()
    raise ValueError(f"Unknown model {name!r}")


def prepare_experiment_splits(exp: Experiment, aoi: AOI):
    seq = load_location_sequence(aoi.slug)
    if len(seq) < exp.history_k + 1:
        raise RuntimeError(f"Need more embeddings (have {len(seq)})")
    samples = build_forecast_samples({aoi.slug: seq}, history=exp.history_k, min_delta_days=exp.min_delta_days)
    train_years = tuple(int(x) for x in exp.train_years.split(",") if x.strip())
    train, val, test = split_samples_by_year(samples, train_years=train_years, test_years=(exp.test_year,))
    return seq, train, val, test


def run_evaluate(db: Session, job: Job) -> None:
    exp = db.get(Experiment, job.experiment_id)
    if exp is None:
        raise RuntimeError("experiment missing")
    aoi = _aoi(db, exp.aoi_id)
    seq, train, val, test = prepare_experiment_splits(exp, aoi)
    device = get_device()
    floor = nuisance_floor({aoi.slug: seq})
    floor_mse = float(sum(r["residual_mse"] for r in floor) / max(len(floor), 1))
    append_log(
        db,
        job,
        f"windows train={len(train)} val={len(val)} test={len(test)} floor={floor_mse:.4f}",
    )

    for row in baseline_rows(test, device):
        for metric in ("residual_mse", "aligned_cosine", "change_map_corr"):
            db.add(
                ResultMetric(
                    experiment_id=exp.id,
                    job_id=job.id,
                    model_name=row["model"],
                    seed=None,
                    metric_name=metric,
                    metric_value=float(row[metric]),
                )
            )
    names = [m.strip() for m in exp.models_csv.split(",") if m.strip()]
    factories = {name: _model_factory(name) for name in names}
    ckpt_dir = CHECKPOINTS_DIR / f"exp{exp.id}"
    rows = run_seeds(
        factories,
        train,
        val,
        test,
        device,
        seeds=tuple(range(exp.seeds)),
        epochs=exp.epochs,
        ckpt_dir=ckpt_dir,
        log=lambda msg: append_log(db, job, msg),
    )
    for row in rows:
        for metric in ("residual_mse", "aligned_cosine", "change_map_corr"):
            db.add(
                ResultMetric(
                    experiment_id=exp.id,
                    job_id=job.id,
                    model_name=row["model"],
                    seed=int(row["seed"]),
                    metric_name=metric,
                    metric_value=float(row[metric]),
                )
            )
    db.add(
        ResultMetric(
            experiment_id=exp.id,
            job_id=job.id,
            model_name="nuisance_floor",
            seed=None,
            metric_name="residual_mse",
            metric_value=floor_mse,
        )
    )
    exp.status = "completed"
    db.commit()
    summary = summarize_seed_runs(rows)
    append_log(db, job, json.dumps(summary, indent=2))


HANDLERS = {
    "index_observations": run_index_observations,
    "ingest_hls": run_ingest_hls,
    "encode": run_encode,
    "evaluate": run_evaluate,
}


def _worker_loop() -> None:
    while not _STOP.is_set():
        db = SessionLocal()
        try:
            job = (
                db.query(Job)
                .filter(Job.status == "queued")
                .order_by(Job.id.asc())
                .first()
            )
            if job is None:
                db.close()
                _STOP.wait(1.0)
                continue
            job.status = "running"
            job.started_at = datetime.utcnow()
            db.commit()
            handler = HANDLERS.get(job.kind)
            try:
                if handler is None:
                    raise RuntimeError(f"Unknown job kind {job.kind}")
                handler(db, job)
                job.status = "done"
            except Exception as exc:
                job.status = "failed"
                job.error = f"{exc}\n{traceback.format_exc()}"
                append_log(db, job, job.error)
            job.finished_at = datetime.utcnow()
            db.commit()
        finally:
            db.close()


def start_worker() -> None:
    global _WORKER
    if _WORKER is not None and _WORKER.is_alive():
        return
    _STOP.clear()
    _WORKER = Thread(target=_worker_loop, name="workbench-jobs", daemon=True)
    _WORKER.start()


def stop_worker() -> None:
    _STOP.set()


def aoi_bounds(aoi: AOI) -> tuple[float, float, float, float]:
    obs = next((o for o in aoi.observations if o.ref_band_path), None)
    if obs and Path(obs.ref_band_path).exists():
        try:
            return patch_bounds_wgs84(obs.ref_band_path, aoi.longitude, aoi.latitude)
        except Exception:
            pass
    return approximate_bounds_wgs84(aoi.longitude, aoi.latitude)


_OVERLAY_SNAP_DAYS = 10


def _snap_encoded_date(requested: str, keys: list[str]) -> tuple[str, int]:
    if requested in keys:
        return requested, 0
    target = date.fromisoformat(requested)
    best = min(keys, key=lambda k: abs((date.fromisoformat(k) - target).days))
    delta = abs((date.fromisoformat(best) - target).days)
    if delta > _OVERLAY_SNAP_DAYS:
        raise ValueError(
            f"No encoding within {_OVERLAY_SNAP_DAYS} days of {requested} "
            f"(nearest is {best}, {delta} days away)."
        )
    return best, delta


def compose_overlay(
    db: Session,
    aoi: AOI,
    date_a: str,
    date_b: str,
    kind: str = "token",
) -> dict:
    """Georeference a 14×14 change map onto the HLS window."""
    west, south, east, north = aoi_bounds(aoi)
    seq = {item["key"]: item for item in load_location_sequence(aoi.slug)}
    have = sorted(seq)
    if not have:
        raise ValueError("No embeddings on disk. Run Encode first.")
    used_a, da = _snap_encoded_date(date_a, have)
    used_b, db_delta = _snap_encoded_date(date_b, have)
    notes: list[str] = []
    if da:
        notes.append(f"{date_a} → {used_a}")
    if db_delta:
        notes.append(f"{date_b} → {used_b}")
    if used_a == used_b:
        raise ValueError(f"Pick two different encoded dates (both resolved to {used_a}).")

    cmap = "magma"
    extra: dict = {}
    upsample = 16
    resample = "bilinear"
    if kind in ("token", "change"):
        kind = "token"
        grid = token_change_map(seq[used_a]["tokens"], seq[used_b]["tokens"])
        upsample = 16
        resample = "bilinear"
    elif kind in ("vegetation", "ndvi", "ndvi_signed"):
        kind = "vegetation"
        hls_dir = location_hls_dir(aoi.slug)
        id_a = str(seq[used_a]["metadata"].get("acquisition_date_id") or "")
        id_b = str(seq[used_b]["metadata"].get("acquisition_date_id") or "")
        if not id_a or not id_b:
            raise FileNotFoundError("Embeddings are missing HLS date ids for NDVI")
        observations = index_hls_directory(hls_dir, lon=aoi.longitude, lat=aoi.latitude)
        bands_a = observations.get(id_a)
        bands_b = observations.get(id_b)
        if not bands_a or not bands_b:
            raise FileNotFoundError(
                "Could not compute NDVI for those dates. HLS GeoTIFFs must still be on disk."
            )
        ref = next(iter(bands_a.values()))
        row, col, _, _ = lonlat_to_pixel(ref, aoi.longitude, aoi.latitude)
        grid = ndvi_signed_change_map(
            ndvi_patch(bands_a, row, col),
            ndvi_patch(bands_b, row, col),
        )
        cmap = "diverging"
        upsample = 2
        resample = "bilinear"
        extra = {
            "frac_gain": float(np.nanmean(grid > 0.02)),
            "frac_loss": float(np.nanmean(grid < -0.02)),
        }
    else:
        raise ValueError(f"Unknown overlay kind {kind}")
    out = OVERLAYS_DIR / f"{aoi.slug}_{used_a}_{used_b}_{kind}.png"
    save_overlay_png(grid, out, upsample=upsample, cmap=cmap, resample=resample)
    return {
        "url": f"/media/overlays/{out.name}",
        "bounds": [[south, west], [north, east]],
        "kind": kind,
        "date_a": used_a,
        "date_b": used_b,
        "requested_a": date_a,
        "requested_b": date_b,
        "snapped": notes,
        "mean": float(np.nanmean(grid)),
        "p98": float(np.nanpercentile(np.abs(grid), 98)),
        **extra,
    }


def compose_vegetation_trend(
    aoi: AOI,
    date_from: str,
    date_to: str,
) -> dict:
    """Mean NDVI per cloud-clear HLS scene inside a date window (a vegetation time series)."""
    d0 = date.fromisoformat(date_from)
    d1 = date.fromisoformat(date_to)
    if d1 < d0:
        d0, d1 = d1, d0

    hls_dir = location_hls_dir(aoi.slug)
    observations = index_hls_directory(hls_dir, lon=aoi.longitude, lat=aoi.latitude)
    if not observations:
        raise FileNotFoundError(f"No HLS GeoTIFFs on disk for {aoi.name}. Run Ingest first.")

    ref = next(iter(next(iter(observations.values())).values()))
    row, col, _, _ = lonlat_to_pixel(ref, aoi.longitude, aoi.latitude)

    points: list[dict] = []
    for date_id, bands in observations.items():
        try:
            dt = date_id_to_datetime(date_id).date()
        except Exception:
            continue
        if dt < d0 or dt > d1:
            continue
        try:
            patch = ndvi_patch(bands, row, col)
        except (FileNotFoundError, ValueError):
            continue
        valid = float(np.mean(np.isfinite(patch)))
        mean = float(np.nanmean(patch)) if valid > 0 else float("nan")
        if not np.isfinite(mean):
            continue
        points.append({"date": dt.isoformat(), "ndvi": round(mean, 4), "valid": round(valid, 3)})

    points.sort(key=lambda p: p["date"])
    if not points:
        raise ValueError("No cloud-clear NDVI scenes in that date range.")

    vals = [p["ndvi"] for p in points]
    return {
        "aoi_id": aoi.id,
        "index": "ndvi",
        "date_from": d0.isoformat(),
        "date_to": d1.isoformat(),
        "count": len(points),
        "first": vals[0],
        "last": vals[-1],
        "delta": round(vals[-1] - vals[0], 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
        "points": points,
    }


def _hls_bands_from_item(aoi: AOI, item: dict) -> dict[str, str] | None:
    hls_dir = location_hls_dir(aoi.slug)
    bands: dict[str, str] = {}
    for name in item.get("metadata", {}).get("source_files") or []:
        path = hls_dir / name
        if not path.exists():
            continue
        try:
            bands[parse_hls_filename(path)["band"]] = str(path)
        except ValueError:
            continue
    return bands or None


def _hls_rgb_from_item(aoi: AOI, item: dict) -> np.ndarray | None:
    bands = _hls_bands_from_item(aoi, item)
    if bands is None or "B02" not in bands or "B03" not in bands or "B04" not in bands:
        return None
    ref = bands.get("B02") or next(iter(bands.values()))
    row, col, _, _ = lonlat_to_pixel(ref, aoi.longitude, aoi.latitude)
    patch = load_observation(bands, row, col, apply_fmask=False)
    return make_rgb_display(patch)


def _hls_ndvi_from_item(aoi: AOI, item: dict) -> np.ndarray | None:
    bands = _hls_bands_from_item(aoi, item)
    if bands is None or "B04" not in bands or "B8A" not in bands:
        return None
    ref = bands.get("B04") or next(iter(bands.values()))
    row, col, _, _ = lonlat_to_pixel(ref, aoi.longitude, aoi.latitude)
    try:
        return ndvi_patch(bands, row, col, apply_fmask=True)
    except (FileNotFoundError, ValueError):
        return None


def _ndvi_mean(grid: np.ndarray | None) -> float | None:
    if grid is None:
        return None
    val = float(np.nanmean(grid))
    return round(val, 4) if np.isfinite(val) else None


def _decode_patch(bundle, tokens, cls, date_id: str, lat: float, lon: float) -> np.ndarray:
    year, doy = parse_hls_date(date_id)
    return decode_tokens_to_patch(
        bundle["model"], bundle["device"], tokens, cls, year, doy, lat, lon
    )


def _decode_rgb(bundle, tokens, cls, date_id: str, lat: float, lon: float) -> np.ndarray:
    return make_rgb_display(denormalize_prithvi(_decode_patch(bundle, tokens, cls, date_id, lat, lon)))


def _decode_ndvi(bundle, tokens, cls, date_id: str, lat: float, lon: float) -> np.ndarray:
    return ndvi_from_sixband_patch(denormalize_prithvi(_decode_patch(bundle, tokens, cls, date_id, lat, lon)))


def _pick_decode_sample(test):
    if not test:
        raise ValueError("No test windows. Need encodings in the experiment test year.")

    def _change(sample) -> float:
        cmap = 1.0 - cosine_similarity_map(sample.past[-1], sample.target)
        return float(np.mean(cmap))

    return max(test, key=_change)


def compose_decode_preview(db: Session, exp_id: int, model_name: str) -> dict:
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise FileNotFoundError("Experiment not found")
    aoi = _aoi(db, exp.aoi_id)
    model_name = model_name.strip().lower()
    ckpt = experiment_ckpt_path(exp.id, model_name)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"No checkpoint for {model_name}. Train that model in this experiment first."
        )
    seq, train, val, test = prepare_experiment_splits(exp, aoi)
    sample = _pick_decode_sample(test)
    by_key = {item["key"]: item for item in seq}
    last_key = sample.past_keys[-1]
    target_key = sample.target_key
    last_item = by_key[last_key]
    target_item = by_key[target_key]
    device = get_device()
    model = _model_factory(model_name)()
    try:
        payload = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    with torch.no_grad():
        pred = model(
            torch.from_numpy(sample.past).unsqueeze(0).to(device),
            torch.from_numpy(sample.delta_days).unsqueeze(0).to(device),
            torch.from_numpy(sample.doy_sincos).unsqueeze(0).to(device),
        )
    pred_tokens = pred.squeeze(0).detach().cpu().numpy()
    bundle = _cached_prithvi()
    lat, lon = aoi.latitude, aoi.longitude
    last_id = str(last_item["metadata"]["acquisition_date_id"])
    target_id = str(target_item["metadata"]["acquisition_date_id"])
    rgb_real = _decode_rgb(bundle, target_item["tokens"], target_item["cls"], target_id, lat, lon)
    rgb_pred = _decode_rgb(bundle, pred_tokens, last_item["cls"], target_id, lat, lon)
    rgb_persist = _decode_rgb(bundle, last_item["tokens"], last_item["cls"], last_id, lat, lon)
    rgb_sat_last = _hls_rgb_from_item(aoi, last_item)
    rgb_sat_target = _hls_rgb_from_item(aoi, target_item)

    # NDVI strip: same Prithvi-decode path for last / target / forecast so means are comparable.
    # HLS NDVI means are returned separately as satellite ground truth (different absolute scale).
    ndvi_last = _decode_ndvi(bundle, last_item["tokens"], last_item["cls"], last_id, lat, lon)
    ndvi_target = _decode_ndvi(bundle, target_item["tokens"], target_item["cls"], target_id, lat, lon)
    ndvi_pred = _decode_ndvi(bundle, pred_tokens, last_item["cls"], target_id, lat, lon)
    hls_ndvi_last = _hls_ndvi_from_item(aoi, last_item)
    hls_ndvi_target = _hls_ndvi_from_item(aoi, target_item)

    stem = f"exp{exp.id}_{model_name}"
    paths = {
        "real": OVERLAYS_DIR / f"{stem}_real.png",
        "pred": OVERLAYS_DIR / f"{stem}_pred.png",
        "persist": OVERLAYS_DIR / f"{stem}_persist.png",
        "sat_last": OVERLAYS_DIR / f"{stem}_sat_last.png",
        "sat_target": OVERLAYS_DIR / f"{stem}_sat_target.png",
        "ndvi_last": OVERLAYS_DIR / f"{stem}_ndvi_last.png",
        "ndvi_target": OVERLAYS_DIR / f"{stem}_ndvi_target.png",
        "ndvi_pred": OVERLAYS_DIR / f"{stem}_ndvi_pred.png",
    }
    save_rgb_png(rgb_real, paths["real"])
    save_rgb_png(rgb_pred, paths["pred"])
    save_rgb_png(rgb_persist, paths["persist"])
    save_overlay_png(ndvi_last, paths["ndvi_last"], upsample=2, cmap="ndvi", resample="bilinear")
    save_overlay_png(ndvi_target, paths["ndvi_target"], upsample=2, cmap="ndvi", resample="bilinear")
    save_overlay_png(ndvi_pred, paths["ndvi_pred"], upsample=2, cmap="ndvi", resample="bilinear")
    sat_last_url = None
    sat_target_url = None
    if rgb_sat_last is not None:
        save_rgb_png(rgb_sat_last, paths["sat_last"])
        sat_last_url = f"/media/overlays/{paths['sat_last'].name}"
    if rgb_sat_target is not None:
        save_rgb_png(rgb_sat_target, paths["sat_target"])
        sat_target_url = f"/media/overlays/{paths['sat_target'].name}"

    try:
        delta_days = abs((date.fromisoformat(target_key) - date.fromisoformat(last_key)).days)
    except ValueError:
        delta_days = None

    return {
        "model": model_name,
        "last_key": last_key,
        "target_key": target_key,
        "delta_days": delta_days,
        "real_url": f"/media/overlays/{paths['real'].name}",
        "pred_url": f"/media/overlays/{paths['pred'].name}",
        "persist_url": f"/media/overlays/{paths['persist'].name}",
        "sat_last_url": sat_last_url,
        "sat_target_url": sat_target_url,
        "ndvi_last_url": f"/media/overlays/{paths['ndvi_last'].name}",
        "ndvi_target_url": f"/media/overlays/{paths['ndvi_target'].name}",
        "ndvi_pred_url": f"/media/overlays/{paths['ndvi_pred'].name}",
        "ndvi_last_mean": _ndvi_mean(ndvi_last),
        "ndvi_target_mean": _ndvi_mean(ndvi_target),
        "ndvi_pred_mean": _ndvi_mean(ndvi_pred),
        "hls_ndvi_last_mean": _ndvi_mean(hls_ndvi_last),
        "hls_ndvi_target_mean": _ndvi_mean(hls_ndvi_target),
    }
