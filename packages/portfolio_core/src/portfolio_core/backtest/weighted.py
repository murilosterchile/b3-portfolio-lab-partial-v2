from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from portfolio_core.data_quality import DataQualityError, validate_prices
from portfolio_core.quant import hierarchical_risk_parity, inverse_volatility, minimum_variance
from portfolio_core.quant.risk import ledoit_wolf_covariance

from .engine import BacktestSummary, _metrics, _one_way_turnover


@dataclass(frozen=True)
class WeightedBacktestConfig:
    transaction_cost_bps: float = 15.0
    initial_capital: float = 100_000.0
    execution_delay_bars: int = 1


_DEFAULT_WEIGHTED_CONFIG = WeightedBacktestConfig()


def build_monthly_risk_benchmark_weights(
    prices: pl.DataFrame,
    features: pl.DataFrame,
    *,
    allocation: str,
    estimator: str = "ledoit_wolf",
    candidate_count: int = 50,
    lookback_observations: int = 252,
) -> pl.DataFrame:
    """Build PIT min-var/inverse-vol/HRP weights from a liquid monthly universe."""
    if allocation not in {"minvar", "inverse_vol", "hrp"}:
        raise ValueError("allocation must be minvar, inverse_vol or hrp")
    if estimator not in {"ledoit_wolf", "sample"}:
        raise ValueError("estimator must be ledoit_wolf or sample")
    if "adjusted_close" not in prices.columns:
        raise DataQualityError("risk benchmarks require adjusted total-return prices")
    rows: list[dict[str, object]] = []
    for signal_date in sorted(set(features["trade_date"].to_list())):
        cross = features.filter(pl.col("trade_date") == signal_date).sort(
            "rank_log_volume_21d", descending=True
        ).head(candidate_count)
        tickers = cross["ticker"].to_list()
        if len(tickers) < 2:
            continue
        history = prices.filter(
            (pl.col("trade_date") <= pl.lit(signal_date)) & pl.col("ticker").is_in(tickers)
        ).select("trade_date", "ticker", "adjusted_close")
        wide = history.pivot(index="trade_date", on="ticker", values="adjusted_close").sort("trade_date")
        available = [ticker for ticker in tickers if ticker in wide.columns]
        if len(available) < 2:
            continue
        values = wide.select(available).fill_null(strategy="forward").tail(lookback_observations + 1).to_numpy()
        if values.shape[0] < 80:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            returns = values[1:] / values[:-1] - 1.0
        returns = returns[np.all(np.isfinite(returns), axis=1)]
        if returns.shape[0] < 60:
            continue
        covariance = (
            ledoit_wolf_covariance(returns)
            if estimator == "ledoit_wolf"
            else np.cov(returns, rowvar=False, ddof=1)
        )
        if allocation == "minvar":
            weights = minimum_variance(covariance, max_weight=min(0.20, 1.0))
        elif allocation == "inverse_vol":
            weights = inverse_volatility(covariance)
        else:
            weights = hierarchical_risk_parity(covariance)
        for ticker, weight in zip(available, weights, strict=True):
            rows.append(
                {"trade_date": signal_date, "ticker": ticker, "target_weight": float(weight)}
            )
    if not rows:
        return pl.DataFrame(
            schema={"trade_date": pl.Date, "ticker": pl.Utf8, "target_weight": pl.Float64}
        )
    return pl.DataFrame(rows).sort(["trade_date", "ticker"])


def run_monthly_weighted_backtest(
    prices: pl.DataFrame,
    target_weights: pl.DataFrame,
    *,
    config: WeightedBacktestConfig = _DEFAULT_WEIGHTED_CONFIG,
) -> tuple[pl.DataFrame, BacktestSummary]:
    """Execute precomputed weights at the next close with half-L1 trading costs."""
    if config.execution_delay_bars < 1:
        raise ValueError("execution_delay_bars must be >= 1")
    required = {"trade_date", "ticker", "target_weight"}
    if missing := required - set(target_weights.columns):
        raise DataQualityError(f"target weights missing columns: {sorted(missing)}")
    if "adjusted_close" not in prices.columns:
        raise DataQualityError("weighted research backtest requires adjusted_close")
    invalid_weights = target_weights.filter(
        pl.col("target_weight").is_null()
        | ~pl.col("target_weight").is_finite()
        | (pl.col("target_weight") < 0)
    )
    if not invalid_weights.is_empty():
        raise DataQualityError("target weights contain invalid values")
    sums = target_weights.group_by("trade_date").agg(pl.col("target_weight").sum().alias("total"))
    if not sums.filter((pl.col("total") > 1.000001) | (pl.col("total") <= 0)).is_empty():
        raise DataQualityError("target weights must sum to (0, 1] on every signal date")

    tickers = target_weights["ticker"].unique().to_list()
    used = prices.filter(pl.col("ticker").is_in(tickers))
    validate_prices(used, context="weighted backtest prices")
    wide = used.select("trade_date", "ticker", "adjusted_close").pivot(
        index="trade_date", on="ticker", values="adjusted_close"
    ).sort("trade_date")
    dates = wide["trade_date"].to_list()
    asset_names = [name for name in wide.columns if name != "trade_date"]
    values = wide.select(asset_names).fill_null(strategy="forward").to_numpy()
    date_to_index = {value: i for i, value in enumerate(dates)}
    ticker_to_index = {ticker: i for i, ticker in enumerate(asset_names)}
    schedule: dict[object, np.ndarray] = {}
    positions: list[int] = []
    for signal_date in sorted(set(target_weights["trade_date"].to_list())):
        idx = date_to_index.get(signal_date)
        if idx is None or idx + config.execution_delay_bars >= len(dates):
            continue
        execution_date = dates[idx + config.execution_delay_bars]
        cross = target_weights.filter(pl.col("trade_date") == signal_date)
        target = np.zeros(len(asset_names))
        for row in cross.iter_rows(named=True):
            if row["ticker"] in ticker_to_index:
                target[ticker_to_index[row["ticker"]]] = float(row["target_weight"])
        schedule[execution_date] = target
        positions.append(int(np.sum(target > 0)))
    if not schedule:
        raise DataQualityError("no executable weighted rebalance dates")

    first_signal = min(target_weights["trade_date"].to_list())
    start = date_to_index[first_signal]
    weights = np.zeros(len(asset_names))
    cash = 1.0
    capital = config.initial_capital
    equity = [capital]
    periodic: list[float] = []
    turnover_total = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = values[1:] / values[:-1] - 1.0
    for i in range(start + 1, len(dates)):
        current_date = dates[i]
        before = capital
        active = weights > 0
        asset_return = returns[i - 1]
        if np.any(active & ~np.isfinite(asset_return)):
            raise DataQualityError("held asset has non-finite adjusted return")
        day_return = float(weights[active] @ asset_return[active]) if np.any(active) else 0.0
        capital *= 1.0 + day_return
        growth = 1.0 + day_return
        drift = np.zeros_like(weights)
        drift[active] = weights[active] * (1.0 + asset_return[active]) / growth
        weights = drift
        cash /= growth
        if current_date in schedule:
            target = schedule[current_date]
            target_cash = max(0.0, 1.0 - float(target.sum()))
            turnover = _one_way_turnover(weights, target, current_cash=cash, target_cash=target_cash)
            turnover_total += turnover
            capital *= 1.0 - turnover * config.transaction_cost_bps / 10_000.0
            weights = target
            cash = target_cash
        periodic.append(capital / before - 1.0)
        equity.append(capital)
    equity_array = np.asarray(equity)
    return (
        pl.DataFrame({"trade_date": dates[start:], "portfolio_value": equity_array}).with_columns(
            pl.lit("adjusted_close").alias("price_basis")
        ),
        _metrics(
            equity_array,
            np.asarray(periodic),
            turnover_total,
            rebalance_periods=len(schedule),
            average_positions=float(np.mean(positions)) if positions else 0.0,
            execution_delay_bars=config.execution_delay_bars,
        ),
    )
