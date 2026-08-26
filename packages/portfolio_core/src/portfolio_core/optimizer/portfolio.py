from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from portfolio_core.data_quality import (
    DataQualityError,
    validate_correlation_matrix,
    validate_finite_array,
)

from .bnb import solve_exact_branch_and_bound
from .scip import solve_exact_scip
from .types import OptimizationResult, QKPInstance


@dataclass(frozen=True)
class Candidate:
    ticker: str
    price: float
    predicted_excess_return: float
    uncertainty: float
    sector: str = "Unknown"
    quant_score: float = 50.0
    liquidity_score: float = 50.0
    volatility_annual: float = 0.25


def build_portfolio_qkp(
    candidates: list[Candidate],
    *,
    correlation: np.ndarray,
    budget: float,
    min_positions: int,
    max_positions: int,
    risk_aversion: float,
    uncertainty_penalty: float = 0.5,
    quant_weight: float = 0.25,
    quant_return_scale: float = 0.02,
    liquidity_weight: float = 0.05,
    liquidity_return_scale: float = 0.01,
    min_position_fraction: float = 0.05,
    sector_max_count: dict[str, int] | None = None,
) -> QKPInstance:
    """Build a cardinality-constrained quadratic stock-selection problem.

    The linear term combines ML alpha, a small independent multifactor prior, liquidity and model
    uncertainty. The pair term is covariance-like rather than correlation-only: the observed
    correlation is scaled by each candidate's annualized volatility before normalization. This
    makes a pair of two highly volatile names carry more risk than an equally correlated low-vol
    pair while preserving the exact QKP formulation.
    """
    n = len(candidates)
    if correlation.shape != (n, n):
        raise ValueError("correlation shape does not match candidates")
    validate_correlation_matrix(correlation, context="QKP correlation")
    expected = np.asarray([c.predicted_excess_return for c in candidates], dtype=float)
    uncertainty = np.asarray([c.uncertainty for c in candidates], dtype=float)
    quant = np.asarray([c.quant_score for c in candidates], dtype=float)
    liquidity = np.asarray([c.liquidity_score for c in candidates], dtype=float)
    volatility = np.asarray([c.volatility_annual for c in candidates], dtype=float)
    prices = np.asarray([c.price for c in candidates], dtype=float)
    validate_finite_array(expected, context="QKP expected_return")
    validate_finite_array(uncertainty, context="QKP uncertainty")
    validate_finite_array(quant, context="QKP quant_score")
    validate_finite_array(liquidity, context="QKP liquidity_score")
    validate_finite_array(volatility, context="QKP volatility")
    validate_finite_array(prices, context="QKP prices")
    if np.any(uncertainty < 0):
        raise DataQualityError("QKP uncertainty must be non-negative")
    if np.any(volatility <= 0):
        raise DataQualityError("QKP volatility must be positive")
    if np.any(prices <= 0):
        raise DataQualityError("QKP prices must be positive")
    if not np.isfinite(budget) or budget <= 0:
        raise DataQualityError("QKP budget must be finite and positive")
    if min_positions < 0 or max_positions < min_positions or max_positions > n:
        raise DataQualityError("invalid QKP cardinality bounds")

    quant_alpha = quant_return_scale * (np.clip(quant, 0.0, 100.0) / 100.0 - 0.5)
    liquidity_alpha = liquidity_return_scale * (
        np.clip(liquidity, 0.0, 100.0) / 100.0 - 0.5
    )
    linear = 10_000.0 * (
        expected
        + quant_weight * quant_alpha
        + liquidity_weight * liquidity_alpha
        - uncertainty_penalty * uncertainty
    )

    # Convert correlation to a covariance-like pair-risk matrix and normalize to a stable scale.
    covariance_like = np.asarray(correlation, dtype=float) * np.outer(volatility, volatility)
    risk_reference = float(np.median(np.maximum(volatility**2, 1e-8)))
    normalized_pair_risk = covariance_like / max(risk_reference, 1e-8)
    scale = max(float(np.median(np.abs(linear))), 1.0)
    pair = -risk_aversion * scale * normalized_pair_risk
    np.fill_diagonal(pair, 0.0)

    # Selection cost is the minimum capital commitment. Allocation is solved in a second stage.
    minimum_commitment = max(budget * min_position_fraction, 1.0)
    costs = np.maximum(minimum_commitment, prices)
    return QKPInstance(
        names=tuple(c.ticker for c in candidates),
        linear_values=linear,
        pair_values=pair,
        costs=costs,
        capacity=budget,
        min_cardinality=min_positions,
        max_cardinality=max_positions,
        sectors=tuple(c.sector for c in candidates),
        sector_max_count=sector_max_count or {},
    )


def solve_portfolio(instance: QKPInstance, *, backend: str = "auto") -> OptimizationResult:
    selected_backend = backend.lower()
    if selected_backend == "auto":
        selected_backend = "scip" if importlib.util.find_spec("pyscipopt") else "bnb"
    if selected_backend == "scip":
        return solve_exact_scip(instance)
    if selected_backend == "bnb":
        return solve_exact_branch_and_bound(instance)
    raise ValueError("backend must be auto, scip or bnb")
