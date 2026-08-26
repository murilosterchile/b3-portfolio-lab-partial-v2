from __future__ import annotations

import hashlib
import secrets
from datetime import UTC, datetime, timedelta

from argon2 import PasswordHasher
from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .models import User, UserSession

_ph = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=2, hash_len=32, salt_len=16)


def hash_password(password: str) -> str:
    if len(password) < 10:
        raise ValueError("Password must have at least 10 characters")
    return _ph.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _ph.verify(password_hash, password)
    except Exception:
        return False


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_session(db: Session, user: User) -> tuple[str, str, datetime]:
    token = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(24)
    expires = datetime.now(UTC) + timedelta(days=7)
    db.add(UserSession(user_id=user.id, token_hash=_token_hash(token), csrf_token=csrf, expires_at=expires))
    db.commit()
    return token, csrf, expires


def current_session(
    session: str | None = Cookie(default=None),
    db: Session = Depends(get_db),
) -> UserSession:
    if not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    stmt = select(UserSession).where(
        UserSession.token_hash == _token_hash(session),
        UserSession.expires_at > datetime.now(UTC),
    )
    record = db.scalar(stmt)
    if record is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid session")
    return record


def current_user(record: UserSession = Depends(current_session), db: Session = Depends(get_db)) -> User:
    user = db.get(User, record.user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def require_csrf(
    request: Request,
    record: UserSession = Depends(current_session),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
    csrf_cookie: str | None = Cookie(default=None, alias="csrf_token"),
) -> None:
    if request.method in {"GET", "HEAD", "OPTIONS"}:
        return
    if not csrf_header or not csrf_cookie or not secrets.compare_digest(csrf_header, csrf_cookie):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    if not secrets.compare_digest(csrf_header, record.csrf_token):
        raise HTTPException(status_code=403, detail="CSRF validation failed")


def revoke_session(db: Session, token: str | None) -> None:
    if token:
        db.execute(delete(UserSession).where(UserSession.token_hash == _token_hash(token)))
        db.commit()


def require_optimizer_auth(
    request: Request,
    session: str | None = Cookie(default=None),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
    csrf_cookie: str | None = Cookie(default=None, alias="csrf_token"),
    db: Session = Depends(get_db),
) -> None:
    """Require session + double-submit CSRF outside local demo mode."""
    if get_settings().demo_mode:
        return
    if not session:
        raise HTTPException(status_code=401, detail="Not authenticated")
    record = db.scalar(
        select(UserSession).where(
            UserSession.token_hash == _token_hash(session),
            UserSession.expires_at > datetime.now(UTC),
        )
    )
    if record is None:
        raise HTTPException(status_code=401, detail="Invalid session")
    if not csrf_header or not csrf_cookie:
        raise HTTPException(status_code=403, detail="CSRF validation failed")
    if not (
        secrets.compare_digest(csrf_header, csrf_cookie)
        and secrets.compare_digest(csrf_header, record.csrf_token)
    ):
        raise HTTPException(status_code=403, detail="CSRF validation failed")
