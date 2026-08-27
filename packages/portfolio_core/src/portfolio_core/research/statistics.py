from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import norm


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


def probabilistic_sharpe_ratio(
    returns: np.ndarray,
    *,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 12,
) -> float:
    """Probability that annualized Sharpe exceeds ``benchmark_sharpe``.

    The standard error includes the observed skewness and excess kurtosis, as
    required for non-Gaussian strategy returns.
    """
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return float("nan")
    sigma = float(np.std(values, ddof=1))
    if sigma <= 1e-12:
        return float("nan")
    centered = values - float(np.mean(values))
    skewness = float(np.mean(centered**3) / (np.std(values) ** 3))
    kurtosis = float(np.mean(centered**4) / (np.std(values) ** 4))
    sharpe = float(np.mean(values) / sigma * np.sqrt(periods_per_year))
    variance = (
        1.0
        - skewness * sharpe / np.sqrt(periods_per_year)
        + ((kurtosis - 1.0) / 4.0) * (sharpe**2 / periods_per_year)
    ) / max(len(values) - 1, 1)
    if variance <= 0:
        return float("nan")
    return float(norm.cdf((sharpe - benchmark_sharpe) / np.sqrt(variance * periods_per_year)))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    *,
    number_of_trials: int,
    trials_sharpe_std: float,
    periods_per_year: int = 12,
) -> tuple[float, float]:
    """Return DSR probability and the multiple-testing Sharpe hurdle."""
    if number_of_trials <= 0:
        raise ValueError("number_of_trials must be positive")
    if trials_sharpe_std < 0:
        raise ValueError("trials_sharpe_std must be non-negative")
    if number_of_trials == 1 or trials_sharpe_std == 0:
        hurdle = 0.0
    else:
        euler_gamma = 0.5772156649015329
        n = float(number_of_trials)
        expected_max_standard_normal = (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / n)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (n * np.e))
        )
        hurdle = float(trials_sharpe_std * expected_max_standard_normal)
    probability = probabilistic_sharpe_ratio(
        returns,
        benchmark_sharpe=hurdle,
        periods_per_year=periods_per_year,
    )
    return probability, hurdle


def probability_backtest_overfitting(strategy_returns: np.ndarray) -> float:
    """CSCV estimate of PBO from a T x N matrix of aligned strategy returns."""
    matrix = np.asarray(strategy_returns, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 8 or matrix.shape[1] < 2:
        return float("nan")
    matrix = matrix[np.all(np.isfinite(matrix), axis=1)]
    if matrix.shape[0] < 8:
        return float("nan")
    # Eight contiguous blocks keep the combinatorics bounded and preserve time.
    blocks = [block for block in np.array_split(np.arange(matrix.shape[0]), 8) if len(block)]
    from itertools import combinations

    logits: list[float] = []
    for selected in combinations(range(len(blocks)), len(blocks) // 2):
        in_blocks = set(selected)
        train_idx = np.concatenate([block for i, block in enumerate(blocks) if i in in_blocks])
        test_idx = np.concatenate([block for i, block in enumerate(blocks) if i not in in_blocks])
        train_scores = np.asarray(
            [sharpe_from_periodic_returns(matrix[train_idx, j]) for j in range(matrix.shape[1])]
        )
        if not np.any(np.isfinite(train_scores)):
            continue
        winner = int(np.nanargmax(train_scores))
        test_scores = np.asarray(
            [sharpe_from_periodic_returns(matrix[test_idx, j]) for j in range(matrix.shape[1])]
        )
        rank = float(np.mean(test_scores <= test_scores[winner]))
        rank = min(max(rank, 1e-6), 1.0 - 1e-6)
        logits.append(float(np.log(rank / (1.0 - rank))))
    return float(np.mean(np.asarray(logits) <= 0.0)) if logits else float("nan")
