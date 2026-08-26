from __future__ import annotations

from collections.abc import Callable

import numpy as np


def moving_block_bootstrap_ci(
    values: np.ndarray,
    metric: Callable[[np.ndarray], float],
    *,
    block_length: int = 3,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Moving-block bootstrap preserving local temporal dependence."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    if block_length <= 0 or n_bootstrap <= 0:
        raise ValueError("block_length and n_bootstrap must be positive")
    block_length = min(block_length, len(x))
    starts = np.arange(0, len(x) - block_length + 1)
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_bootstrap, dtype=float)
    blocks_needed = int(np.ceil(len(x) / block_length))
    for i in range(n_bootstrap):
        sampled_starts = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([x[s : s + block_length] for s in sampled_starts])[: len(x)]
        estimates[i] = metric(sample)
    alpha = 1.0 - confidence
    return (
        float(metric(x)),
        float(np.nanquantile(estimates, alpha / 2.0)),
        float(np.nanquantile(estimates, 1.0 - alpha / 2.0)),
    )


def sharpe_from_periodic_returns(returns: np.ndarray, *, periods_per_year: int = 12) -> float:
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or np.std(values, ddof=1) <= 1e-12:
        return 0.0
    return float(np.mean(values) * periods_per_year / (np.std(values, ddof=1) * np.sqrt(periods_per_year)))


def cagr_from_periodic_returns(returns: np.ndarray, *, periods_per_year: int = 12) -> float:
    values = np.asarray(returns, dtype=float)
    if len(values) == 0 or np.any(values <= -1.0):
        return float("nan")
    wealth = float(np.prod(1.0 + values))
    years = len(values) / periods_per_year
    return float(wealth ** (1.0 / years) - 1.0) if years > 0 else float("nan")
