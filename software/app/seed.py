"""Seed admin user, a demo researcher, and default AOIs on first launch."""

from __future__ import annotations

from sqlalchemy.orm import Session

from engine.paths import load_config

from .auth import hash_password
from .models import AOI, User


def _ensure_user(db: Session, email: str, password: str, role: str) -> None:
    if db.query(User).filter(User.email == email).first():
        return
    db.add(User(email=email, hashed_password=hash_password(password), role=role))
    db.commit()


def seed_if_empty(db: Session) -> None:
    _ensure_user(db, "admin@workbench.local", "admin", "admin")
    _ensure_user(db, "researcher@workbench.local", "researcher", "researcher")
    cfg = load_config()
    added = False
    for slug, loc in cfg.get("locations", {}).items():
        if db.query(AOI).filter(AOI.slug == slug).first():
            continue
        db.add(
            AOI(
                slug=slug,
                name=loc.get("name", slug),
                latitude=float(loc["latitude"]),
                longitude=float(loc["longitude"]),
                biome=loc.get("environment", ""),
                landcover=loc.get("landcover", ""),
                description=loc.get("description", ""),
                owner_id=None,
                is_shared=True,
            )
        )
        added = True
    if added:
        db.commit()
