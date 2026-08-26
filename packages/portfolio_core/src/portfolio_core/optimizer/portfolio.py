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
    issuer_id: str | None = None


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
    fixed_k: int | None = None,
    previous_selected: set[str] | None = None,
    turnover_selection_penalty: float = 0.0,
) -> QKPInstance:
    """Build a cardinality-constrained quadratic stock-selection problem.

    The linear term combines ML alpha, a small independent multifactor prior, liquidity and model
    uncertainty. The pair term is covariance-like rather than correlation-only: the observed
    correlation is scaled by each candidate's annualized volatility before normalization. This
    makes a pair of two highly volatile names carry more risk than an equally correlated low-vol
    pair while preserving the exact QKP formulation.
    """
    n = len(candidates)
    if n == 0:
        raise DataQualityError("QKP requires at least one candidate")
    if correlation.shape != (n, n):
        raise ValueError("correlation shape does not match candidates")
    validate_correlation_matrix(correlation, context="QKP correlation")
    expected = np.asarray([c.predicted_excess_return for c in candidates], dtype=float)
    uncertainty = np.asarray([c.uncertainty for c in candidates], dtype=float)
    quant = np.asarray([c.quant_score for c in candidates], dtype=float)
    liquidity = np.asarray([c.liquidity_score for c in candidates], dtype=float)
    volatility = np.asarray([c.volatility_annual for c in candidates], dtype=float)
    validate_finite_array(expected, context="QKP expected_return")
    validate_finite_array(uncertainty, context="QKP uncertainty")
    validate_finite_array(quant, context="QKP quant_score")
    validate_finite_array(liquidity, context="QKP liquidity_score")
    validate_finite_array(volatility, context="QKP volatility")
    if np.any(uncertainty < 0):
        raise DataQualityError("QKP uncertainty must be non-negative")
    if np.any(volatility <= 0):
        raise DataQualityError("QKP volatility must be positive")
    if min_positions < 0 or max_positions < min_positions or max_positions > n:
        raise DataQualityError("invalid QKP cardinality bounds")
    k = max_positions if fixed_k is None else fixed_k
    if not min_positions <= k <= max_positions or k < 1:
        raise DataQualityError("fixed_k must be within the cardinality bounds")
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise DataQualityError("risk_aversion must be finite and non-negative")
    if turnover_selection_penalty < 0:
        raise DataQualityError("turnover_selection_penalty must be non-negative")

    quant_alpha = quant_return_scale * (np.clip(quant, 0.0, 100.0) / 100.0 - 0.5)
    liquidity_alpha = liquidity_return_scale * (
        np.clip(liquidity, 0.0, 100.0) / 100.0 - 0.5
    )
    alpha = (
        expected
        + quant_weight * quant_alpha
        + liquidity_weight * liquidity_alpha
        - uncertainty_penalty * uncertainty
    )

    covariance = np.asarray(correlation, dtype=float) * np.outer(volatility, volatility)
    risk_scale = risk_aversion / float(k**2)
    linear = alpha - risk_scale * np.diag(covariance)
    if previous_selected:
        linear -= np.asarray(
            [0.0 if candidate.ticker in previous_selected else turnover_selection_penalty for candidate in candidates]
        )
    # The solver sums each i,j pair once, hence the factor two from x' Sigma x.
    pair = -2.0 * risk_scale * covariance
    np.fill_diagonal(pair, 0.0)
    linear *= 10_000.0
    pair *= 10_000.0

    # Selection is economically invariant to nominal share price. Budget and
    # round lots belong exclusively to the downstream discrete allocator.
    costs = np.zeros(n, dtype=float)
    return QKPInstance(
        names=tuple(c.ticker for c in candidates),
        linear_values=linear,
        pair_values=pair,
        costs=costs,
        capacity=0.0,
        min_cardinality=k,
        max_cardinality=k,
        sectors=tuple(c.sector for c in candidates),
        sector_max_count=sector_max_count or {},
        issuer_ids=tuple(c.issuer_id or c.ticker[:4] for c in candidates),
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
