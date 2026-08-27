from __future__ import annotations

import numpy as np


def black_litterman_posterior(
    *,
    covariance: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float,
    views: np.ndarray,
    view_matrix: np.ndarray | None = None,
    view_uncertainty: np.ndarray | None = None,
    tau: float = 0.05,
) -> np.ndarray:
    """Return posterior expected excess returns.

    This low-level challenger deliberately requires explicit market weights and
    OOF-calibrated view error. It is not wired into the primary strategy until
    both inputs have governed point-in-time provenance.
    """
    n = covariance.shape[0]
    weights = np.asarray(market_weights, dtype=float)
    if weights.shape != (n,) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("market_weights must be point-in-time, non-negative and sum to one")
    if view_uncertainty is None:
        raise ValueError("Black-Litterman requires OOF-calibrated view_uncertainty")
    p = np.eye(n) if view_matrix is None else np.asarray(view_matrix, dtype=float)
    q = np.asarray(views, dtype=float)
    prior = risk_aversion * covariance @ weights
    omega = np.asarray(view_uncertainty, dtype=float)
    if omega.shape != (p.shape[0], p.shape[0]) or np.any(np.diag(omega) <= 0):
        raise ValueError("view_uncertainty must be a positive OOF error covariance matrix")
    tau_cov_inv = np.linalg.pinv(tau * covariance)
    omega_inv = np.linalg.pinv(omega)
    middle = np.linalg.pinv(tau_cov_inv + p.T @ omega_inv @ p)
    return middle @ (tau_cov_inv @ prior + p.T @ omega_inv @ q)
