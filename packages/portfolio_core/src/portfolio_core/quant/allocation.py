from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

from .risk import covariance_to_correlation


def equal_weight(n: int) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    return np.full(n, 1.0 / n)


def inverse_volatility(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    inv = 1.0 / vol
    return inv / inv.sum()


def minimum_variance(cov: np.ndarray, *, max_weight: float = 1.0) -> np.ndarray:
    n = cov.shape[0]
    x0 = equal_weight(n)
    result = minimize(
        lambda w: float(w @ cov @ w),
        x0,
        method="SLSQP",
        constraints=[{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}],
        bounds=[(0.0, max_weight)] * n,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success:
        return inverse_volatility(cov)
    return np.asarray(result.x)


def cost_aware_allocation(
    cov: np.ndarray,
    *,
    previous_weights: np.ndarray | None = None,
    min_weight: float = 0.0,
    max_weight: float = 0.25,
    turnover_penalty: float = 0.01,
    sectors: list[str] | None = None,
    max_sector_weight: float | dict[str, float] = 0.40,
    issuer_ids: list[str] | None = None,
    max_issuer_weight: float = 0.25,
    liquidity_weight_caps: np.ndarray | None = None,
    benchmark_weights: np.ndarray | None = None,
    tracking_error_cap: float | None = None,
) -> np.ndarray:
    """Long-only risk allocator with explicit investability and turnover constraints."""
    covariance = np.asarray(cov, dtype=float)
    n = covariance.shape[0]
    if covariance.shape != (n, n) or n == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if min_weight < 0 or max_weight <= 0 or min_weight > max_weight:
        raise ValueError("invalid weight bounds")
    if n * min_weight > 1.0 + 1e-12 or n * max_weight < 1.0 - 1e-12:
        raise ValueError("weight bounds cannot form a fully invested portfolio")
    previous = (
        np.asarray(previous_weights, dtype=float)
        if previous_weights is not None
        else np.zeros(n, dtype=float)
    )
    if previous.shape != (n,):
        raise ValueError("previous_weights must match covariance")
    caps = np.full(n, max_weight)
    if liquidity_weight_caps is not None:
        liquidity_caps = np.asarray(liquidity_weight_caps, dtype=float)
        if liquidity_caps.shape != (n,) or np.any(liquidity_caps <= 0):
            raise ValueError("liquidity_weight_caps must be positive and match covariance")
        caps = np.minimum(caps, liquidity_caps)
    if float(caps.sum()) < 1.0 - 1e-12:
        raise ValueError("combined liquidity/weight caps cannot form a fully invested portfolio")
    bounds = [(min_weight, float(cap)) for cap in caps]
    constraints: list[dict[str, object]] = [
        {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    ]
    if sectors is not None:
        if len(sectors) != n:
            raise ValueError("sectors must match covariance")
        for sector in sorted(set(sectors)):
            index = np.asarray([value == sector for value in sectors])
            cap = (
                float(max_sector_weight.get(sector, 1.0))
                if isinstance(max_sector_weight, dict)
                else float(max_sector_weight)
            )
            constraints.append(
                {"type": "ineq", "fun": lambda w, idx=index, limit=cap: float(limit - w[idx].sum())}
            )
    if issuer_ids is not None:
        if len(issuer_ids) != n:
            raise ValueError("issuer_ids must match covariance")
        for issuer in sorted(set(issuer_ids)):
            index = np.asarray([value == issuer for value in issuer_ids])
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, idx=index: float(max_issuer_weight - w[idx].sum()),
                }
            )
    benchmark = None
    if tracking_error_cap is not None:
        benchmark = np.asarray(benchmark_weights, dtype=float)
        if benchmark.shape != (n,) or tracking_error_cap <= 0:
            raise ValueError("tracking-error cap requires matching benchmark weights")
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: float(
                    tracking_error_cap**2 - (w - benchmark) @ covariance @ (w - benchmark)
                ),
            }
        )
    linear_inequalities: list[np.ndarray] = []
    linear_limits: list[float] = []
    if sectors is not None:
        for sector in sorted(set(sectors)):
            linear_inequalities.append(
                np.asarray([value == sector for value in sectors], dtype=float)
            )
            linear_limits.append(
                float(max_sector_weight.get(sector, 1.0))
                if isinstance(max_sector_weight, dict)
                else float(max_sector_weight)
            )
    if issuer_ids is not None:
        for issuer in sorted(set(issuer_ids)):
            linear_inequalities.append(
                np.asarray([value == issuer for value in issuer_ids], dtype=float)
            )
            linear_limits.append(float(max_issuer_weight))
    feasible = linprog(
        np.zeros(n),
        A_ub=np.asarray(linear_inequalities) if linear_inequalities else None,
        b_ub=np.asarray(linear_limits) if linear_limits else None,
        A_eq=np.ones((1, n)),
        b_eq=np.ones(1),
        bounds=bounds,
        method="highs",
    )
    if not feasible.success:
        raise ValueError("allocation constraints cannot form a fully invested portfolio")
    x0 = np.asarray(feasible.x, dtype=float)
    result = minimize(
        lambda w: float(w @ covariance @ w + turnover_penalty * np.abs(w - previous).sum()),
        x0,
        method="SLSQP",
        constraints=constraints,
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    weights = x0 if not result.success else np.maximum(np.asarray(result.x, dtype=float), 0.0)
    weights = weights / weights.sum()
    if np.any(weights < min_weight - 1e-7) or np.any(weights > caps + 1e-7):
        raise ValueError("allocator could not satisfy weight/liquidity bounds")
    if sectors is not None:
        for sector in set(sectors):
            cap = (
                float(max_sector_weight.get(sector, 1.0))
                if isinstance(max_sector_weight, dict)
                else float(max_sector_weight)
            )
            if weights[np.asarray([value == sector for value in sectors])].sum() > cap + 1e-7:
                raise ValueError("allocator could not satisfy sector exposure")
    if issuer_ids is not None:
        for issuer in set(issuer_ids):
            if (
                weights[np.asarray([value == issuer for value in issuer_ids])].sum()
                > max_issuer_weight + 1e-7
            ):
                raise ValueError("allocator could not satisfy issuer exposure")
    return weights


def _cluster_variance(cov: np.ndarray, items: list[int]) -> float:
    sub = cov[np.ix_(items, items)]
    w = inverse_volatility(sub)
    return float(w @ sub @ w)


def hierarchical_risk_parity(cov: np.ndarray) -> np.ndarray:
    """HRP-style allocation using single-link hierarchical clustering and recursive bisection."""
    n = cov.shape[0]
    if n == 1:
        return np.array([1.0])
    corr = covariance_to_correlation(cov)
    distance = np.sqrt(np.maximum((1.0 - corr) / 2.0, 0.0))
    condensed = squareform(distance, checks=False)
    order = list(map(int, leaves_list(linkage(condensed, method="single"))))
    weights = np.ones(n)
    clusters: list[list[int]] = [order]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left, right = cluster[:split], cluster[split:]
        var_left = _cluster_variance(cov, left)
        var_right = _cluster_variance(cov, right)
        alpha = 1.0 - var_left / max(var_left + var_right, 1e-12)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        clusters.extend([left, right])
    return weights / weights.sum()
