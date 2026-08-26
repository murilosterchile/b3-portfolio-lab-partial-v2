from __future__ import annotations

from fastapi import APIRouter, Cookie, Depends, HTTPException, Response
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..models import User
from ..rate_limit import auth_rate_limit
from ..schemas import LoginRequest, RegisterRequest
from ..security import create_session, current_user, hash_password, require_csrf, revoke_session, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


def _set_auth_cookies(response: Response, token: str, csrf: str) -> None:
    settings = get_settings()
    response.set_cookie(
        "session", token, httponly=True, secure=settings.secure_cookies,
        samesite="lax", max_age=7 * 24 * 3600, path="/",
    )
    response.set_cookie(
        "csrf_token", csrf, httponly=False, secure=settings.secure_cookies,
        samesite="lax", max_age=7 * 24 * 3600, path="/",
    )


@router.post("/register", dependencies=[Depends(auth_rate_limit)])
def register(payload: RegisterRequest, response: Response, db: Session = Depends(get_db)) -> dict[str, str]:
    email = payload.email.lower().strip()
    if db.scalar(select(User).where(User.email == email)) is not None:
        raise HTTPException(status_code=409, detail="Email already registered")
    try:
        password_hash = hash_password(payload.password)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    user = User(email=email, password_hash=password_hash)
    db.add(user)
    db.commit()
    db.refresh(user)
    token, csrf, _ = create_session(db, user)
    _set_auth_cookies(response, token, csrf)
    return {"email": user.email}


@router.post("/login", dependencies=[Depends(auth_rate_limit)])
def login(payload: LoginRequest, response: Response, db: Session = Depends(get_db)) -> dict[str, str]:
    user = db.scalar(select(User).where(User.email == payload.email.lower().strip()))
    if user is None or not verify_password(user.password_hash, payload.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token, csrf, _ = create_session(db, user)
    _set_auth_cookies(response, token, csrf)
    return {"email": user.email}


@router.post("/logout", dependencies=[Depends(require_csrf)])
def logout(response: Response, session: str | None = Cookie(default=None), db: Session = Depends(get_db)) -> dict[str, bool]:
    revoke_session(db, session)
    response.delete_cookie("session", path="/")
    response.delete_cookie("csrf_token", path="/")
    return {"ok": True}


@router.get("/me")
def me(user: User = Depends(current_user)) -> dict[str, str]:
    return {"id": user.id, "email": user.email}
