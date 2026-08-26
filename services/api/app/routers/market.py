from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import AssetSnapshot
from ..schemas import AssetOut

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/ranking", response_model=list[AssetOut])
def ranking(limit: int = Query(default=20, ge=1, le=100), db: Session = Depends(get_db)) -> list[AssetSnapshot]:
    latest = db.scalar(select(func.max(AssetSnapshot.as_of)))
    if latest is None:
        return []
    stmt = (
        select(AssetSnapshot)
        .where(AssetSnapshot.as_of == latest)
        .order_by(desc(AssetSnapshot.ml_score))
        .limit(limit)
    )
    return list(db.scalars(stmt))
