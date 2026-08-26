from __future__ import annotations

import numpy as np
import polars as pl

from portfolio_core.optimizer import Candidate, build_portfolio_qkp, solve_portfolio
from portfolio_core.quant import hierarchical_risk_parity, inverse_volatility, minimum_variance
from portfolio_core.quant.risk import covariance_to_correlation


def point_in_time_covariance(
    prices: pl.DataFrame,
    tickers: list[str],
    signal_date: object,
    *,
    lookback_observations: int = 252,
    min_return_observations: int = 60,
) -> tuple[list[str], np.ndarray] | None:
    """Estimate covariance from consecutive, genuinely observed quotes only."""
    history = prices.filter(
        (pl.col("trade_date") <= pl.lit(signal_date)) & pl.col("ticker").is_in(tickers)
    ).select("trade_date", "ticker", "adjusted_close")
    wide = history.pivot(index="trade_date", on="ticker", values="adjusted_close").sort("trade_date")
    available = [ticker for ticker in tickers if ticker in wide.columns]
    if len(available) < 2:
        return None
    values = wide.select(available).tail(lookback_observations + 1).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = values[1:] / values[:-1] - 1.0
    counts = np.isfinite(returns).sum(axis=0)
    keep = np.flatnonzero(counts >= min_return_observations)
    if len(keep) < 2:
        return None
    returns = returns[:, keep]
    available = [available[index] for index in keep]
    n = len(available)
    covariance = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            common = np.isfinite(returns[:, i]) & np.isfinite(returns[:, j])
            if int(common.sum()) < min_return_observations:
                return None
            value = float(np.cov(returns[common, i], returns[common, j], ddof=1)[0, 1])
            covariance[i, j] = value
            covariance[j, i] = value
    # Pairwise missingness can make the sample matrix indefinite. Shrink the
    # observed off-diagonal estimates and clip only numerical negative modes.
    diagonal = np.diag(np.diag(covariance))
    covariance = 0.8 * covariance + 0.2 * diagonal
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance = (eigenvectors * np.maximum(eigenvalues, 1e-12)) @ eigenvectors.T
    return available, covariance * 252.0


def build_monthly_qkp_weights(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    allocation: str,
    candidate_count: int = 18,
    k_values: tuple[int, ...] = (6, 7, 8, 9, 10),
    risk_aversion: float = 0.7,
    turnover_selection_penalty: float = 0.001,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Run rolling fixed-K QKP selection and a downstream price-free allocation."""
    if allocation not in {"equal_weight", "inverse_vol", "hrp", "cost_aware_minvar"}:
        raise ValueError("unsupported QKP allocation")
    weight_rows: list[dict[str, object]] = []
    solver_rows: list[dict[str, object]] = []
    previous_selected: set[str] = set()
    previous_weights: dict[str, float] = {}
    for signal_date in sorted(set(signals.get_column("trade_date").to_list())):
        cross = signals.filter(pl.col("trade_date") == signal_date).sort(
            "signal_research_v2", descending=True
        ).head(candidate_count)
        requested = cross.get_column("ticker").to_list()
        covariance_result = point_in_time_covariance(prices, requested, signal_date)
        if covariance_result is None:
            continue
        available, covariance = covariance_result
        cross = cross.filter(pl.col("ticker").is_in(available))
        order = {ticker: index for index, ticker in enumerate(available)}
        cross = cross.with_columns(pl.col("ticker").replace_strict(order).alias("__order")).sort(
            "__order"
        )
        available = cross.get_column("ticker").to_list()
        indices = [order[ticker] for ticker in available]
        covariance = covariance[np.ix_(indices, indices)]
        if len(available) < min(k_values):
            continue
        correlation = covariance_to_correlation(covariance)
        volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        candidates = [
            Candidate(
                ticker=str(row["ticker"]),
                price=float(row.get("close") or 1.0),
                predicted_excess_return=float(row["predicted_excess_return"]),
                uncertainty=float(row["prediction_uncertainty"]),
                sector=str(row.get("sector") or "Unknown"),
                quant_score=100.0 * float(row.get("quant_score") or 0.5),
                liquidity_score=100.0 * float(row.get("rank_log_volume_21d") or 0.5),
                volatility_annual=float(vol),
                issuer_id=str(row.get("CD_CVM") or row.get("issuer_identifier") or row["ticker"]),
            )
            for row, vol in zip(cross.iter_rows(named=True), volatility, strict=True)
        ]
        best = None
        for k in (value for value in k_values if value <= len(candidates)):
            instance = build_portfolio_qkp(
                candidates,
                correlation=correlation,
                budget=1.0,
                min_positions=k,
                max_positions=k,
                fixed_k=k,
                risk_aversion=risk_aversion,
                previous_selected=previous_selected,
                turnover_selection_penalty=turnover_selection_penalty,
                sector_max_count={"Financials": 3, "Energy": 2, "Materials": 2, "Utilities": 3, "Consumer": 3},
            )
            result = solve_portfolio(instance, backend="auto")
            normalized_objective = result.objective / k if result.selected_indices else -np.inf
            if best is None or normalized_objective > best[0]:
                best = (normalized_objective, k, result)
        if best is None or not best[2].selected_indices:
            continue
        normalized_objective, k, result = best
        selected = list(result.selected_indices)
        selected_names = [candidates[index].ticker for index in selected]
        selected_covariance = covariance[np.ix_(selected, selected)]
        if allocation == "equal_weight":
            weights = np.full(k, 1.0 / k)
        elif allocation == "inverse_vol":
            weights = inverse_volatility(selected_covariance)
        elif allocation == "hrp":
            weights = hierarchical_risk_parity(selected_covariance)
        else:
            optimized = minimum_variance(selected_covariance, max_weight=min(0.25, 1.0))
            prior = np.asarray([previous_weights.get(ticker, 0.0) for ticker in selected_names])
            if prior.sum() > 0:
                prior /= prior.sum()
                weights = 0.75 * optimized + 0.25 * prior
            else:
                weights = optimized
            weights /= weights.sum()
        for ticker, weight in zip(selected_names, weights, strict=True):
            weight_rows.append(
                {"trade_date": signal_date, "ticker": ticker, "target_weight": float(weight)}
            )
        solver_rows.append(
            {
                "trade_date": signal_date,
                "fixed_k": k,
                "solver": result.solver,
                "objective": result.objective,
                "objective_per_position": normalized_objective,
            }
        )
        previous_selected = set(selected_names)
        previous_weights = dict(zip(selected_names, map(float, weights), strict=True))
    weight_schema = {"trade_date": pl.Date, "ticker": pl.Utf8, "target_weight": pl.Float64}
    solver_schema = {"trade_date": pl.Date, "fixed_k": pl.Int64, "solver": pl.Utf8, "objective": pl.Float64, "objective_per_position": pl.Float64}
    return (
        pl.DataFrame(weight_rows).sort(["trade_date", "ticker"]) if weight_rows else pl.DataFrame(schema=weight_schema),
        pl.DataFrame(solver_rows).sort("trade_date") if solver_rows else pl.DataFrame(schema=solver_schema),
    )
