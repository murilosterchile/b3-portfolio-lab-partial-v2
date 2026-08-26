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

    `views` can be ML expected excess returns. Lower diagonal values in `view_uncertainty`
    express higher confidence in those views.
    """
    n = covariance.shape[0]
    p = np.eye(n) if view_matrix is None else np.asarray(view_matrix, dtype=float)
    q = np.asarray(views, dtype=float)
    prior = risk_aversion * covariance @ np.asarray(market_weights, dtype=float)
    omega = (
        np.diag(np.maximum(np.diag(p @ (tau * covariance) @ p.T), 1e-8))
        if view_uncertainty is None
        else np.asarray(view_uncertainty, dtype=float)
    )
    tau_cov_inv = np.linalg.pinv(tau * covariance)
    omega_inv = np.linalg.pinv(omega)
    middle = np.linalg.pinv(tau_cov_inv + p.T @ omega_inv @ p)
    return middle @ (tau_cov_inv @ prior + p.T @ omega_inv @ q)
