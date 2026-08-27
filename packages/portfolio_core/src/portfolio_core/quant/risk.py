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


def pairwise_price_covariance(
    prices: np.ndarray,
    asset_names: list[str],
    *,
    min_return_observations: int = 60,
    minimum_coverage: float = 0.80,
    annualization: float = 252.0,
) -> tuple[list[str], np.ndarray, dict[str, int]] | None:
    """Shrink pairwise returns from genuinely consecutive observed quotes.

    Missing prices never become zero returns. Assets first pass an individual
    coverage gate; pairwise estimation is then projected to PSD after diagonal
    shrinkage. Effective observations are returned for audit logs.
    """
    values = np.asarray(prices, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(asset_names):
        raise DataQualityError("price matrix shape does not match asset names")
    if not 0.0 < minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if values.shape[0] < 2:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = values[1:] / values[:-1] - 1.0
    consecutive = np.isfinite(values[1:]) & np.isfinite(values[:-1])
    returns[~consecutive] = np.nan
    counts = np.isfinite(returns).sum(axis=0)
    required = max(min_return_observations, int(np.ceil(minimum_coverage * returns.shape[0])))
    keep = np.flatnonzero(counts >= required)
    if len(keep) < 2:
        return None
    returns = returns[:, keep]
    names = [asset_names[index] for index in keep]
    observations = {name: int(counts[index]) for name, index in zip(names, keep, strict=True)}
    n = len(names)
    covariance = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            common = np.isfinite(returns[:, i]) & np.isfinite(returns[:, j])
            if int(common.sum()) < min_return_observations:
                return None
            value = float(np.cov(returns[common, i], returns[common, j], ddof=1)[0, 1])
            covariance[i, j] = covariance[j, i] = value
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.8 * covariance + 0.2 * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
    covariance *= annualization
    validate_finite_array(covariance, context="pairwise shrunk covariance")
    return names, covariance, observations
