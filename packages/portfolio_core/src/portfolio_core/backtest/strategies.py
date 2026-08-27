from __future__ import annotations

import numpy as np
import polars as pl

from portfolio_core.optimizer import Candidate, build_portfolio_qkp, solve_portfolio
from portfolio_core.quant import (
    cost_aware_allocation,
    hierarchical_risk_parity,
    inverse_volatility,
)
from portfolio_core.quant.risk import covariance_to_correlation, pairwise_price_covariance


def point_in_time_covariance(
    prices: pl.DataFrame,
    tickers: list[str],
    signal_date: object,
    *,
    lookback_observations: int = 252,
    min_return_observations: int = 60,
    minimum_coverage: float = 0.80,
    include_observations: bool = False,
) -> tuple[list[str], np.ndarray] | tuple[list[str], np.ndarray, dict[str, int]] | None:
    """Estimate covariance from consecutive, genuinely observed quotes only."""
    history = prices.filter(
        (pl.col("trade_date") <= pl.lit(signal_date)) & pl.col("ticker").is_in(tickers)
    ).select("trade_date", "ticker", "adjusted_close")
    wide = history.pivot(index="trade_date", on="ticker", values="adjusted_close").sort("trade_date")
    available = [ticker for ticker in tickers if ticker in wide.columns]
    if len(available) < 2:
        return None
    estimate = pairwise_price_covariance(
        wide.select(available).tail(lookback_observations + 1).to_numpy(),
        available,
        min_return_observations=min_return_observations,
        minimum_coverage=minimum_coverage,
    )
    if estimate is None:
        return None
    eligible, covariance, observations = estimate
    return (eligible, covariance, observations) if include_observations else (eligible, covariance)


def build_monthly_qkp_weights(
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    *,
    allocation: str,
    candidate_count: int = 30,
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
        # Investability defines the candidate universe. Alpha ranks only inside
        # that liquid/tradable set, leaving QKP room to diversify.
        cross = signals.filter(pl.col("trade_date") == signal_date).sort(
            ["rank_log_volume_21d", "signal_research_v2"], descending=[True, True]
        ).head(candidate_count)
        requested = cross.get_column("ticker").to_list()
        covariance_result = point_in_time_covariance(
            prices, requested, signal_date, include_observations=True
        )
        if covariance_result is None:
            continue
        available, covariance, effective_observations = covariance_result
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
                sector_max_count={
                    sector: max(1, int(0.40 * k))
                    for sector in {candidate.sector for candidate in candidates}
                },
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
        selected_rows = [candidates[index] for index in selected]
        selected_covariance = covariance[np.ix_(selected, selected)]
        if allocation == "equal_weight":
            weights = np.full(k, 1.0 / k)
        elif allocation == "inverse_vol":
            weights = inverse_volatility(selected_covariance)
        elif allocation == "hrp":
            weights = hierarchical_risk_parity(selected_covariance)
        else:
            prior = np.asarray([previous_weights.get(ticker, 0.0) for ticker in selected_names])
            liquidity = np.asarray([candidate.liquidity_score for candidate in selected_rows])
            # The cap is deliberately loose; it prevents an illiquid name from
            # receiving the same capacity as the most liquid names.
            liquidity_caps = np.clip(0.10 + 0.20 * liquidity / 100.0, 0.10, 0.30)
            weights = cost_aware_allocation(
                selected_covariance,
                previous_weights=prior,
                max_weight=0.25,
                turnover_penalty=0.01,
                sectors=[candidate.sector for candidate in selected_rows],
                max_sector_weight=0.40,
                issuer_ids=[candidate.issuer_id or candidate.ticker for candidate in selected_rows],
                max_issuer_weight=0.25,
                liquidity_weight_caps=liquidity_caps,
            )
        for candidate, weight in zip(selected_rows, weights, strict=True):
            weight_rows.append(
                {
                    "trade_date": signal_date,
                    "ticker": candidate.ticker,
                    "issuer_id": candidate.issuer_id or candidate.ticker,
                    "sector": candidate.sector,
                    "target_weight": float(weight),
                }
            )
        solver_rows.append(
            {
                "trade_date": signal_date,
                "fixed_k": k,
                "solver": result.solver,
                "objective": result.objective,
                "objective_per_position": normalized_objective,
                "min_effective_observations": min(
                    effective_observations[ticker] for ticker in selected_names
                ),
                "effective_observations": ",".join(
                    f"{ticker}:{effective_observations[ticker]}" for ticker in selected_names
                ),
                "issuer_exposure": ",".join(
                    f"{candidate.issuer_id or candidate.ticker}:{weight:.8f}"
                    for candidate, weight in zip(selected_rows, weights, strict=True)
                ),
            }
        )
        previous_selected = set(selected_names)
        previous_weights = dict(zip(selected_names, map(float, weights), strict=True))
    weight_schema = {
        "trade_date": pl.Date,
        "ticker": pl.Utf8,
        "issuer_id": pl.Utf8,
        "sector": pl.Utf8,
        "target_weight": pl.Float64,
    }
    solver_schema = {
        "trade_date": pl.Date,
        "fixed_k": pl.Int64,
        "solver": pl.Utf8,
        "objective": pl.Float64,
        "objective_per_position": pl.Float64,
        "min_effective_observations": pl.Int64,
        "effective_observations": pl.Utf8,
        "issuer_exposure": pl.Utf8,
    }
    return (
        pl.DataFrame(weight_rows).sort(["trade_date", "ticker"]) if weight_rows else pl.DataFrame(schema=weight_schema),
        pl.DataFrame(solver_rows).sort("trade_date") if solver_rows else pl.DataFrame(schema=solver_schema),
    )
