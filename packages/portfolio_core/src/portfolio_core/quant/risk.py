from __future__ import annotations

import numpy as np
from sklearn.covariance import LedoitWolf

from portfolio_core.data_quality import DataQualityError, validate_finite_array


def ledoit_wolf_covariance(returns: np.ndarray) -> np.ndarray:
    """Shrinkage covariance; rows are observations, columns are assets."""
    clean = np.asarray(returns, dtype=float)
    if clean.ndim != 2:
        raise DataQualityError(f"returns must be a 2D matrix, got shape={clean.shape}")
    validate_finite_array(clean, context="Ledoit-Wolf returns")
    if clean.shape[0] < 3:
        return np.eye(clean.shape[1]) * 1e-6
    covariance = LedoitWolf().fit(clean).covariance_
    validate_finite_array(covariance, context="Ledoit-Wolf covariance")
    return covariance


def covariance_to_correlation(cov: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise DataQualityError(f"covariance must be square, got shape={cov.shape}")
    validate_finite_array(cov, context="covariance")
    if not np.allclose(cov, cov.T, atol=1e-10):
        raise DataQualityError("covariance must be symmetric")
    std = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)
