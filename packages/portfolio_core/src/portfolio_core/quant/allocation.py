from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
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
