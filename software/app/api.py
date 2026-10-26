"""HTTP API: auth, CRUD, jobs, overlays."""

from __future__ import annotations

import csv
import io
import os
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from engine.hls import approximate_bounds_wgs84
from engine.paths import load_config

from .auth import get_current_user, hash_password, is_admin, new_token, verify_password
from .db import get_db
from .models import (
    AOI,
    Embedding,
    Experiment,
    Job,
    Observation,
    ResultMetric,
    SessionToken,
    User,
)
from .explain import canned_answer, chat_with_openai, list_topics, openai_configured, openai_model
from .services import (
    AVAILABLE_MODELS,
    aoi_bounds,
    compose_decode_preview,
    compose_overlay,
    compose_vegetation_trend,
    enqueue,
    experiment_ckpt_path,
)

api = APIRouter(prefix="/api")


class LoginIn(BaseModel):
    email: str
    password: str


class RegisterIn(BaseModel):
    email: str
    password: str


class AOIIn(BaseModel):
    name: str
    latitude: float
    longitude: float
    slug: str | None = None
    biome: str = ""
    landcover: str = ""
    description: str = ""
    notes: str = ""
    shared: bool = False


class ExperimentIn(BaseModel):
    name: str
    aoi_id: int
    models_csv: str = "linear,token_lstm"
    seeds: int = 3
    min_clear: float = 0.90
    min_delta_days: int = 21
    history_k: int = 6
    train_years: str = "2022,2023"
    test_year: int = 2024
    epochs: int = 12


class ExplainCannedIn(BaseModel):
    topic: str


class ExplainChatIn(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    context: dict | None = None
    history: list[dict[str, str]] = Field(default_factory=list)


class JobIn(BaseModel):
    kind: str
    aoi_id: int | None = None
    experiment_id: int | None = None
    payload: dict = Field(default_factory=dict)


def _user_out(u: User) -> dict:
    return {"id": u.id, "email": u.email, "role": u.role}


def _norm_email(email: str) -> str:
    return (email or "").strip().lower()


def _validate_password(password: str) -> None:
    if len(password or "") < 6:
        raise HTTPException(400, "Password must be at least 6 characters")


def _aois_for(db: Session, user: User):
    q = db.query(AOI)
    if not is_admin(user):
        q = q.filter((AOI.owner_id == user.id) | (AOI.is_shared.is_(True)) | (AOI.owner_id.is_(None)))
    return q.order_by(AOI.id)


def _get_aoi(db: Session, aoi_id: int, user: User, *, mutate: bool = False) -> AOI:
    aoi = db.get(AOI, aoi_id)
    if aoi is None:
        raise HTTPException(404, "AOI not found")
    if is_admin(user):
        return aoi
    owned = aoi.owner_id == user.id
    shared = bool(aoi.is_shared) or aoi.owner_id is None
    if mutate:
        if not owned:
            raise HTTPException(403, "You can only edit study areas you created")
        return aoi
    if owned or shared:
        return aoi
    raise HTTPException(404, "AOI not found")


def _get_experiment(db: Session, exp_id: int, user: User) -> Experiment:
    exp = db.get(Experiment, exp_id)
    if exp is None:
        raise HTTPException(404, "Experiment not found")
    if is_admin(user) or exp.created_by == user.id:
        return exp
    raise HTTPException(404, "Experiment not found")


def _aoi_out(a: AOI) -> dict:
    west, south, east, north = aoi_bounds(a) if a.observations else approximate_bounds_wgs84(a.longitude, a.latitude)
    return {
        "id": a.id,
        "slug": a.slug,
        "name": a.name,
        "latitude": a.latitude,
        "longitude": a.longitude,
        "biome": a.biome,
        "landcover": a.landcover,
        "description": a.description,
        "notes": a.notes,
        "bounds": [[south, west], [north, east]],
        "n_observations": len(a.observations),
        "n_embeddings": len(a.embeddings),
        "owner_id": a.owner_id,
        "shared": bool(a.is_shared) or a.owner_id is None,
    }


@api.post("/auth/register")
def register(body: RegisterIn, db: Session = Depends(get_db)):
    email = _norm_email(body.email)
    if not email or "@" not in email:
        raise HTTPException(400, "Enter a valid email")
    _validate_password(body.password)
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(400, "Email already registered")
    user = User(email=email, hashed_password=hash_password(body.password), role="researcher")
    db.add(user)
    db.commit()
    token = SessionToken(user_id=user.id, token=new_token())
    db.add(token)
    db.commit()
    return {"token": token.token, "user": _user_out(user)}


@api.post("/auth/login")
def login(body: LoginIn, db: Session = Depends(get_db)):
    email = _norm_email(body.email)
    user = db.query(User).filter(User.email == email).one_or_none()
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(401, "Invalid credentials")
    token = SessionToken(user_id=user.id, token=new_token())
    db.add(token)
    db.commit()
    return {"token": token.token, "user": _user_out(user)}


@api.post("/auth/logout")
def logout(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    db.query(SessionToken).filter(SessionToken.user_id == user.id).delete()
    db.commit()
    return {"ok": True}


@api.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return _user_out(user)


@api.get("/config")
def client_config(user: User = Depends(get_current_user)):
    cfg = load_config()
    key = os.environ.get("GOOGLE_MAPS_API_KEY") or (cfg.get("map") or {}).get("google_maps_api_key") or ""
    return {"google_maps_api_key": key}


@api.get("/aois")
def list_aois(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return [_aoi_out(a) for a in _aois_for(db, user).all()]


@api.post("/aois")
def create_aoi(body: AOIIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    slug = (body.slug or body.name.lower().replace(" ", "_"))[:64]
    if db.query(AOI).filter(AOI.slug == slug).first():
        slug = f"{slug}_{int(datetime.utcnow().timestamp())}"
    aoi = AOI(
        slug=slug,
        name=body.name,
        latitude=body.latitude,
        longitude=body.longitude,
        biome=body.biome,
        landcover=body.landcover,
        description=body.description,
        notes=body.notes,
        owner_id=user.id,
        is_shared=body.shared,
    )
    db.add(aoi)
    db.commit()
    db.refresh(aoi)
    return _aoi_out(aoi)


@api.put("/aois/{aoi_id}")
def update_aoi(aoi_id: int, body: AOIIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    aoi = _get_aoi(db, aoi_id, user, mutate=True)
    aoi.name = body.name
    aoi.latitude = body.latitude
    aoi.longitude = body.longitude
    aoi.biome = body.biome
    aoi.landcover = body.landcover
    aoi.description = body.description
    aoi.notes = body.notes
    aoi.is_shared = body.shared
    db.commit()
    return _aoi_out(aoi)


@api.delete("/aois/{aoi_id}")
def delete_aoi(aoi_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    aoi = _get_aoi(db, aoi_id, user, mutate=True)
    exp_ids = [eid for (eid,) in db.query(Experiment.id).filter(Experiment.aoi_id == aoi_id).all()]
    job_ids = [jid for (jid,) in db.query(Job.id).filter(Job.aoi_id == aoi_id).all()]
    if exp_ids:
        extra = [jid for (jid,) in db.query(Job.id).filter(Job.experiment_id.in_(exp_ids)).all()]
        job_ids = list({*job_ids, *extra})
        db.query(ResultMetric).filter(ResultMetric.experiment_id.in_(exp_ids)).delete(synchronize_session=False)
    if job_ids:
        db.query(ResultMetric).filter(ResultMetric.job_id.in_(job_ids)).update(
            {ResultMetric.job_id: None}, synchronize_session=False
        )
        # Keep a running ingest row (worker still holds it) but detach the FK.
        db.query(Job).filter(Job.id.in_(job_ids), Job.status == "running").update(
            {Job.aoi_id: None, Job.experiment_id: None}, synchronize_session=False
        )
        db.query(Job).filter(Job.id.in_(job_ids), Job.status != "running").delete(synchronize_session=False)
    if exp_ids:
        db.query(Experiment).filter(Experiment.id.in_(exp_ids)).delete(synchronize_session=False)
    db.query(Embedding).filter(Embedding.aoi_id == aoi_id).delete(synchronize_session=False)
    db.query(Observation).filter(Observation.aoi_id == aoi_id).delete(synchronize_session=False)
    db.delete(aoi)
    db.commit()
    return {"ok": True}


@api.get("/aois/{aoi_id}/observations")
def list_obs(aoi_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _get_aoi(db, aoi_id, user)
    rows = db.query(Observation).filter(Observation.aoi_id == aoi_id).order_by(Observation.date_key).all()
    return [
        {
            "id": r.id,
            "date_id": r.date_id,
            "date_key": r.date_key,
            "clear_fraction": r.clear_fraction,
            "status": r.status,
            "granule": r.granule,
        }
        for r in rows
    ]


@api.delete("/observations/{obs_id}")
def delete_obs(obs_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    row = db.get(Observation, obs_id)
    if row is None:
        raise HTTPException(404, "Not found")
    _get_aoi(db, row.aoi_id, user, mutate=True)
    db.delete(row)
    db.commit()
    return {"ok": True}


@api.get("/aois/{aoi_id}/embeddings")
def list_emb(aoi_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _get_aoi(db, aoi_id, user)
    rows = db.query(Embedding).filter(Embedding.aoi_id == aoi_id).order_by(Embedding.date_key).all()
    return [
        {
            "id": r.id,
            "date_key": r.date_key,
            "encoder_version": r.encoder_version,
            "tokens_path": r.tokens_path,
        }
        for r in rows
    ]


@api.get("/experiments")
def list_exp(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    q = db.query(Experiment).order_by(Experiment.id.desc())
    if not is_admin(user):
        q = q.filter(Experiment.created_by == user.id)
    rows = q.all()
    return [
        {
            "id": e.id,
            "name": e.name,
            "aoi_id": e.aoi_id,
            "models_csv": e.models_csv,
            "seeds": e.seeds,
            "min_clear": e.min_clear,
            "min_delta_days": e.min_delta_days,
            "status": e.status,
            "epochs": e.epochs,
            "models": [m.strip() for m in e.models_csv.split(",") if m.strip()],
            "checkpoints": [
                m.strip()
                for m in e.models_csv.split(",")
                if m.strip() and experiment_ckpt_path(e.id, m.strip()).exists()
            ],
        }
        for e in rows
    ]


@api.post("/experiments")
def create_exp(body: ExperimentIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _get_aoi(db, body.aoi_id, user)
    e = Experiment(**body.model_dump(), created_by=user.id)
    db.add(e)
    db.commit()
    db.refresh(e)
    return {"id": e.id, **body.model_dump(), "status": e.status}


@api.delete("/experiments/{exp_id}")
def delete_exp(exp_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    e = _get_experiment(db, exp_id, user)
    # Remove dependent rows first so FK constraints don't block the delete.
    db.query(ResultMetric).filter(ResultMetric.experiment_id == exp_id).delete(
        synchronize_session=False
    )
    db.query(Job).filter(Job.experiment_id == exp_id).delete(synchronize_session=False)
    db.delete(e)
    db.commit()
    return {"ok": True}


@api.get("/models")
def list_models(user: User = Depends(get_current_user)):
    return AVAILABLE_MODELS


@api.get("/experiments/{exp_id}/decode")
def decode_preview(
    exp_id: int,
    model: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    _get_experiment(db, exp_id, user)
    try:
        return compose_decode_preview(db, exp_id, model)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.get("/experiments/{exp_id}/results")
def exp_results(exp_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _get_experiment(db, exp_id, user)
    rows = db.query(ResultMetric).filter(ResultMetric.experiment_id == exp_id).all()
    return [
        {
            "model_name": r.model_name,
            "seed": r.seed,
            "metric_name": r.metric_name,
            "metric_value": r.metric_value,
            "split": r.split,
        }
        for r in rows
    ]


@api.get("/experiments/{exp_id}/results.csv")
def exp_results_csv(exp_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    _get_experiment(db, exp_id, user)
    rows = db.query(ResultMetric).filter(ResultMetric.experiment_id == exp_id).all()
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["model_name", "seed", "metric_name", "metric_value", "split"])
    for r in rows:
        w.writerow([r.model_name, r.seed, r.metric_name, r.metric_value, r.split])
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=experiment_{exp_id}.csv"},
    )


@api.get("/jobs")
def list_jobs(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    q = db.query(Job).order_by(Job.id.desc())
    if not is_admin(user):
        q = q.filter(Job.created_by == user.id)
    rows = q.limit(80).all()
    return [
        {
            "id": j.id,
            "kind": j.kind,
            "status": j.status,
            "aoi_id": j.aoi_id,
            "experiment_id": j.experiment_id,
            "error": j.error,
            "created_at": j.created_at.isoformat() if j.created_at else None,
            "log_tail": (j.log_text or "")[-1200:],
        }
        for j in rows
    ]


@api.post("/jobs")
def create_job(body: JobIn, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    if body.aoi_id is not None:
        _get_aoi(db, body.aoi_id, user)
    if body.experiment_id is not None:
        _get_experiment(db, body.experiment_id, user)
    job = enqueue(
        db,
        kind=body.kind,
        user_id=user.id,
        aoi_id=body.aoi_id,
        experiment_id=body.experiment_id,
        payload=body.payload,
    )
    return {"id": job.id, "status": job.status, "kind": job.kind}


@api.delete("/jobs")
def clear_jobs(db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """Remove finished and failed jobs. Queued and running jobs are kept."""
    q = db.query(Job).filter(Job.status.in_(("done", "failed")))
    if not is_admin(user):
        q = q.filter(Job.created_by == user.id)
    ids = [jid for (jid,) in q.with_entities(Job.id).all()]
    if not ids:
        return {"ok": True, "deleted": 0}
    # Detach result metrics so the FK constraint doesn't block the delete (keep the metrics).
    db.query(ResultMetric).filter(ResultMetric.job_id.in_(ids)).update(
        {ResultMetric.job_id: None}, synchronize_session=False
    )
    n = db.query(Job).filter(Job.id.in_(ids)).delete(synchronize_session=False)
    db.commit()
    return {"ok": True, "deleted": int(n)}


@api.delete("/jobs/{job_id}")
def delete_job(job_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    job = db.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "Not found")
    if not is_admin(user) and job.created_by != user.id:
        raise HTTPException(404, "Not found")
    if job.status == "running":
        raise HTTPException(400, "Cannot delete a running job")
    db.query(ResultMetric).filter(ResultMetric.job_id == job.id).update(
        {ResultMetric.job_id: None}, synchronize_session=False
    )
    db.delete(job)
    db.commit()
    return {"ok": True}


@api.get("/overlays/change")
def overlay_change(
    aoi_id: int,
    date_a: str,
    date_b: str,
    kind: str = "token",
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    aoi = _get_aoi(db, aoi_id, user)
    try:
        return compose_overlay(db, aoi, date_a, date_b, kind=kind)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.get("/analysis/vegetation")
def vegetation_trend(
    aoi_id: int,
    date_from: str,
    date_to: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    aoi = _get_aoi(db, aoi_id, user)
    try:
        return compose_vegetation_trend(aoi, date_from, date_to)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@api.get("/explain/status")
def explain_status(user: User = Depends(get_current_user)):
    return {
        "openai_configured": openai_configured(),
        "model": openai_model() if openai_configured() else None,
        "topics": list_topics(),
    }


@api.post("/explain/canned")
def explain_canned(body: ExplainCannedIn, user: User = Depends(get_current_user)):
    try:
        return canned_answer(body.topic.strip())
    except KeyError as exc:
        raise HTTPException(404, f"Unknown topic: {body.topic}") from exc


@api.post("/explain/chat")
def explain_chat(body: ExplainChatIn, user: User = Depends(get_current_user)):
    try:
        return chat_with_openai(body.message, context=body.context, history=body.history)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

