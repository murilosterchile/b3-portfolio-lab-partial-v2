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
    expected_return_horizon_bars: int,
    covariance_horizon_bars: int,
    uncertainty_penalty: float = 0.5,
    min_position_fraction: float = 0.05,
    sector_max_count: dict[str, int] | None = None,
    fixed_k: int | None = None,
) -> QKPInstance:
    """Build a cardinality-constrained quadratic stock-selection problem.

    For the fixed-cardinality portfolio ``w = x / k``, coefficients are the exact
    expansion of ``mu' w - risk_aversion * w' covariance w``. ``mu`` is the
    predicted excess return less its calibrated uncertainty penalty. Expected
    returns and covariance must use the same explicit horizon.
    """
    n = len(candidates)
    if n == 0:
        raise DataQualityError("QKP requires at least one candidate")
    if correlation.shape != (n, n):
        raise ValueError("correlation shape does not match candidates")
    validate_correlation_matrix(correlation, context="QKP correlation")
    expected = np.asarray([c.predicted_excess_return for c in candidates], dtype=float)
    uncertainty = np.asarray([c.uncertainty for c in candidates], dtype=float)
    volatility = np.asarray([c.volatility_annual for c in candidates], dtype=float)
    validate_finite_array(expected, context="QKP expected_return")
    validate_finite_array(uncertainty, context="QKP uncertainty")
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
    if not isinstance(expected_return_horizon_bars, int) or expected_return_horizon_bars < 1:
        raise DataQualityError("expected_return_horizon_bars must be a positive integer")
    if not isinstance(covariance_horizon_bars, int) or covariance_horizon_bars < 1:
        raise DataQualityError("covariance_horizon_bars must be a positive integer")
    if expected_return_horizon_bars != covariance_horizon_bars:
        raise DataQualityError("expected-return and covariance horizons must match")
    if not np.isfinite(uncertainty_penalty) or uncertainty_penalty < 0:
        raise DataQualityError("uncertainty_penalty must be finite and non-negative")

    alpha = expected - uncertainty_penalty * uncertainty
    covariance = (
        np.asarray(correlation, dtype=float)
        * np.outer(volatility, volatility)
        * (covariance_horizon_bars / 252.0)
    )
    risk_scale = risk_aversion / float(k**2)
    linear = alpha / float(k) - risk_scale * np.diag(covariance)
    # The solver sums each i,j pair once, hence the factor two from x' Sigma x.
    pair = -2.0 * risk_scale * covariance
    np.fill_diagonal(pair, 0.0)

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
        expected_return_horizon_bars=expected_return_horizon_bars,
        covariance_horizon_bars=covariance_horizon_bars,
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
