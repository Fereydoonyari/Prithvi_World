"""Relational models for the workbench."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(32), default="researcher")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    sessions: Mapped[list[SessionToken]] = relationship(back_populates="user")
    aois: Mapped[list[AOI]] = relationship(back_populates="owner")


class SessionToken(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    user: Mapped[User] = relationship(back_populates="sessions")


class AOI(Base):
    __tablename__ = "aois"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    latitude: Mapped[float] = mapped_column(Float)
    longitude: Mapped[float] = mapped_column(Float)
    biome: Mapped[str] = mapped_column(String(64), default="")
    landcover: Mapped[str] = mapped_column(String(64), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    notes: Mapped[str] = mapped_column(Text, default="")
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    is_shared: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    owner: Mapped[User | None] = relationship(back_populates="aois")
    observations: Mapped[list[Observation]] = relationship(back_populates="aoi", cascade="all, delete-orphan")
    embeddings: Mapped[list[Embedding]] = relationship(back_populates="aoi", cascade="all, delete-orphan")
    experiments: Mapped[list[Experiment]] = relationship(back_populates="aoi")


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (UniqueConstraint("aoi_id", "date_key", name="uq_obs_aoi_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    aoi_id: Mapped[int] = mapped_column(ForeignKey("aois.id"), index=True)
    date_id: Mapped[str] = mapped_column(String(16))
    date_key: Mapped[str] = mapped_column(String(16), index=True)
    granule: Mapped[str] = mapped_column(String(128), default="")
    clear_fraction: Mapped[float] = mapped_column(Float, default=-1.0)
    status: Mapped[str] = mapped_column(String(32), default="indexed")
    hls_dir: Mapped[str] = mapped_column(String(512), default="")
    ref_band_path: Mapped[str] = mapped_column(String(512), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    aoi: Mapped[AOI] = relationship(back_populates="observations")


class Embedding(Base):
    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint("aoi_id", "date_key", name="uq_emb_aoi_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    aoi_id: Mapped[int] = mapped_column(ForeignKey("aois.id"), index=True)
    observation_id: Mapped[int | None] = mapped_column(ForeignKey("observations.id"), nullable=True)
    date_key: Mapped[str] = mapped_column(String(16), index=True)
    encoder_version: Mapped[str] = mapped_column(String(128), default="Prithvi-EO-2.0-100M-TL")
    tokens_path: Mapped[str] = mapped_column(String(512))
    cls_path: Mapped[str] = mapped_column(String(512))
    metadata_path: Mapped[str] = mapped_column(String(512))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    aoi: Mapped[AOI] = relationship(back_populates="embeddings")


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    aoi_id: Mapped[int] = mapped_column(ForeignKey("aois.id"), index=True)
    models_csv: Mapped[str] = mapped_column(String(255), default="linear,token_lstm")
    seeds: Mapped[int] = mapped_column(Integer, default=3)
    min_clear: Mapped[float] = mapped_column(Float, default=0.90)
    min_delta_days: Mapped[int] = mapped_column(Integer, default=21)
    history_k: Mapped[int] = mapped_column(Integer, default=6)
    train_years: Mapped[str] = mapped_column(String(32), default="2022,2023")
    test_year: Mapped[int] = mapped_column(Integer, default=2024)
    epochs: Mapped[int] = mapped_column(Integer, default=20)
    status: Mapped[str] = mapped_column(String(32), default="draft")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    aoi: Mapped[AOI] = relationship(back_populates="experiments")
    results: Mapped[list[ResultMetric]] = relationship(back_populates="experiment", cascade="all, delete-orphan")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    aoi_id: Mapped[int | None] = mapped_column(ForeignKey("aois.id"), nullable=True)
    experiment_id: Mapped[int | None] = mapped_column(ForeignKey("experiments.id"), nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    log_text: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ResultMetric(Base):
    __tablename__ = "result_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    job_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    model_name: Mapped[str] = mapped_column(String(64))
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metric_name: Mapped[str] = mapped_column(String(64))
    metric_value: Mapped[float] = mapped_column(Float)
    split: Mapped[str] = mapped_column(String(16), default="test")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    experiment: Mapped[Experiment] = relationship(back_populates="results")

