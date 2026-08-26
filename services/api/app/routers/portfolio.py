from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import desc, func, select
from sqlalchemy.orm import Session

from portfolio_core.optimizer import Candidate, build_portfolio_qkp, solve_portfolio
from portfolio_core.quant import hierarchical_risk_parity, inverse_volatility, minimum_variance

from ..config import get_settings
from ..db import get_db
from ..models import AssetSnapshot
from ..rate_limit import optimizer_rate_limit
from ..schemas import OptimizeRequest, OptimizeResponse, PositionOut
from ..security import require_optimizer_auth
from ..research import historical_correlation

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


def _correlation(rows: list[AssetSnapshot]) -> np.ndarray:
    n = len(rows)
    corr = np.full((n, n), 0.12, dtype=float)
    np.fill_diagonal(corr, 1.0)
    for i in range(n):
        for j in range(i + 1, n):
            same_sector = rows[i].sector == rows[j].sector
            value = 0.52 if same_sector else 0.12
            if {rows[i].sector, rows[j].sector} == {"Energy", "Materials"}:
                value = 0.36
            corr[i, j] = corr[j, i] = value
    return corr


@router.post(
    "/optimize",
    response_model=OptimizeResponse,
    dependencies=[Depends(optimizer_rate_limit), Depends(require_optimizer_auth)],
)
def optimize(payload: OptimizeRequest, db: Session = Depends(get_db)) -> OptimizeResponse:
    if payload.min_positions > payload.max_positions:
        raise HTTPException(status_code=422, detail="min_positions must be <= max_positions")
    latest = db.scalar(select(func.max(AssetSnapshot.as_of)))
    if latest is None:
        raise HTTPException(status_code=409, detail="No model snapshot available")
    rows = list(
        db.scalars(
            select(AssetSnapshot)
            .where(AssetSnapshot.as_of == latest)
            .order_by(
                desc(
                    0.80 * AssetSnapshot.ml_score
                    + 0.15 * AssetSnapshot.quant_score
                    + 0.05 * AssetSnapshot.liquidity_score
                )
            )
            .limit(payload.candidate_count)
        )
    )
    if len(rows) < payload.min_positions:
        raise HTTPException(status_code=409, detail="Insufficient candidate universe")

    corr = historical_correlation(tickers=[r.ticker for r in rows], data_dir=get_settings().data_dir)
    correlation_source = "B3 history + Ledoit-Wolf"
    if corr is None:
        corr = _correlation(rows)
        correlation_source = "synthetic demo fallback"
    candidates = [
        Candidate(
            ticker=r.ticker,
            price=r.price,
            predicted_excess_return=r.predicted_excess_return,
            uncertainty=r.prediction_uncertainty,
            sector=r.sector,
            quant_score=r.quant_score,
            liquidity_score=r.liquidity_score,
            volatility_annual=r.volatility_annual,
        )
        for r in rows
    ]
    instance = build_portfolio_qkp(
        candidates,
        correlation=corr,
        budget=payload.budget,
        min_positions=payload.min_positions,
        max_positions=payload.max_positions,
        risk_aversion=payload.risk_aversion,
        uncertainty_penalty=payload.uncertainty_penalty,
        min_position_fraction=1.0 / max(payload.max_positions * 1.8, 1.0),
        sector_max_count={"Financials": 3, "Energy": 2, "Materials": 2, "Utilities": 3, "Consumer": 3},
    )
    result = solve_portfolio(instance, backend=get_settings().qkp_solver)
    if not result.selected_indices:
        raise HTTPException(status_code=409, detail=f"Optimizer returned {result.status}")

    selected_rows = [rows[i] for i in result.selected_indices]
    selected_corr = corr[np.ix_(result.selected_indices, result.selected_indices)]
    vols = np.asarray([r.volatility_annual for r in selected_rows])
    cov = selected_corr * np.outer(vols, vols)
    try:
        if payload.allocation == "minvar":
            weights = minimum_variance(cov, max_weight=min(0.30, 1.0))
        elif payload.allocation == "inverse_vol":
            weights = inverse_volatility(cov)
        else:
            weights = hierarchical_risk_parity(cov)
    except Exception:
        weights = inverse_volatility(cov)

    mu = np.asarray([r.predicted_excess_return for r in selected_rows])
    expected = float(weights @ mu)
    vol = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    positions = [
        PositionOut(
            ticker=r.ticker, company=r.company, sector=r.sector, weight=float(w),
            amount=float(w * payload.budget), price=r.price,
            expected_excess_return=r.predicted_excess_return,
            volatility_annual=r.volatility_annual,
        )
        for r, w in zip(selected_rows, weights, strict=True)
    ]
    return OptimizeResponse(
        status=result.status,
        solver=result.solver,
        exact=result.exact,
        objective=result.objective,
        expected_excess_return=expected,
        expected_volatility=vol,
        positions=positions,
        notes=[
            "QKP selection is solved exactly; allocation is a separate risk-allocation stage.",
            f"Correlation source: {correlation_source}.",
            "This prototype is research software, not an investment recommendation.",
        ],
    )
