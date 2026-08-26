from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from portfolio_core.data_quality import DataQualityError, validate_prices
from portfolio_core.quant import hierarchical_risk_parity, inverse_volatility, minimum_variance

from .engine import BacktestSummary, _metrics, _one_way_turnover


@dataclass(frozen=True)
class WeightedBacktestConfig:
    transaction_cost_bps: float = 15.0
    initial_capital: float = 100_000.0
    execution_delay_bars: int = 1
    max_stale_valuation_sessions: int = 20


@dataclass(frozen=True)
class DetailedWeightedBacktest:
    curve: pl.DataFrame
    summary: BacktestSummary
    rejected_orders: pl.DataFrame


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
        )
        if "CD_CVM" in cross.columns:
            cross = cross.with_columns(pl.col("CD_CVM").cast(pl.Utf8).alias("__issuer"))
        elif "issuer_identifier" in cross.columns:
            cross = cross.with_columns(pl.col("issuer_identifier").cast(pl.Utf8).alias("__issuer"))
        else:
            cross = cross.with_columns(pl.col("ticker").str.slice(0, 4).alias("__issuer"))
        cross = cross.unique(subset="__issuer", keep="first", maintain_order=True).head(
            candidate_count
        )
        tickers = cross["ticker"].to_list()
        if len(tickers) < 2:
            continue
        if estimator == "ledoit_wolf":
            from .strategies import point_in_time_covariance

            covariance_result = point_in_time_covariance(
                prices,
                tickers,
                signal_date,
                lookback_observations=lookback_observations,
            )
            if covariance_result is None:
                continue
            available, covariance = covariance_result
        else:
            history = prices.filter(
                (pl.col("trade_date") <= pl.lit(signal_date)) & pl.col("ticker").is_in(tickers)
            ).select("trade_date", "ticker", "adjusted_close")
            wide = history.pivot(index="trade_date", on="ticker", values="adjusted_close").sort("trade_date")
            available = [ticker for ticker in tickers if ticker in wide.columns]
            if len(available) < 2:
                continue
            values = wide.select(available).tail(lookback_observations + 1).to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                returns = values[1:] / values[:-1] - 1.0
            returns = returns[np.all(np.isfinite(returns), axis=1)]
            if returns.shape[0] < 60:
                continue
            covariance = np.cov(returns, rowvar=False, ddof=1) * 252.0
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
    """Compatibility wrapper for the detailed weighted execution engine."""
    result = run_monthly_weighted_backtest_detailed(prices, target_weights, config=config)
    return result.curve, result.summary


def run_monthly_weighted_backtest_detailed(
    prices: pl.DataFrame,
    target_weights: pl.DataFrame,
    *,
    config: WeightedBacktestConfig = _DEFAULT_WEIGHTED_CONFIG,
) -> DetailedWeightedBacktest:
    """Execute weights only where the execution close is genuinely observed."""
    if config.execution_delay_bars < 1:
        raise ValueError("execution_delay_bars must be >= 1")
    if config.max_stale_valuation_sessions < 1:
        raise ValueError("max_stale_valuation_sessions must be positive")
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
    if used.is_empty():
        raise DataQualityError("target weights have no matching price history")
    validate_prices(used, context="weighted backtest prices")
    wide = used.select("trade_date", "ticker", "adjusted_close").pivot(
        index="trade_date", on="ticker", values="adjusted_close"
    )
    wide = prices.select("trade_date").unique().join(
        wide, on="trade_date", how="left", validate="1:1"
    ).sort("trade_date")
    dates = wide["trade_date"].to_list()
    asset_names = [name for name in wide.columns if name != "trade_date"]
    observed = wide.select(pl.col(asset_names).is_not_null()).to_numpy()
    values = wide.select(asset_names).fill_null(strategy="forward").to_numpy()
    date_to_index = {value: i for i, value in enumerate(dates)}
    ticker_to_index = {ticker: i for i, ticker in enumerate(asset_names)}
    schedule: dict[object, np.ndarray] = {}
    schedule_signal_dates: dict[object, object] = {}
    positions: list[int] = []
    rejected_rows: list[dict[str, object]] = []
    for signal_date in sorted(set(target_weights["trade_date"].to_list())):
        idx = date_to_index.get(signal_date)
        if idx is None or idx + config.execution_delay_bars >= len(dates):
            continue
        execution_date = dates[idx + config.execution_delay_bars]
        cross = target_weights.filter(pl.col("trade_date") == signal_date)
        target = np.zeros(len(asset_names))
        for row in cross.iter_rows(named=True):
            if row["ticker"] in ticker_to_index:
                column = ticker_to_index[row["ticker"]]
                if observed[idx + config.execution_delay_bars, column]:
                    target[column] = float(row["target_weight"])
                else:
                    rejected_rows.append(
                        {
                            "signal_date": signal_date,
                            "execution_date": execution_date,
                            "ticker": row["ticker"],
                            "target_weight": float(row["target_weight"]),
                            "reason": "no_observed_execution_quote",
                        }
                    )
            else:
                rejected_rows.append(
                    {
                        "signal_date": signal_date,
                        "execution_date": execution_date,
                        "ticker": row["ticker"],
                        "target_weight": float(row["target_weight"]),
                        "reason": "missing_price_history",
                    }
                )
        schedule[execution_date] = target
        schedule_signal_dates[execution_date] = signal_date
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
    stale_age = np.zeros(len(asset_names), dtype=int)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = values[1:] / values[:-1] - 1.0
    for i in range(start + 1, len(dates)):
        current_date = dates[i]
        before = capital
        active = weights > 0
        stale_age = np.where(observed[i], 0, stale_age + 1)
        asset_return = returns[i - 1].copy()
        asset_return[active & (stale_age == config.max_stale_valuation_sessions + 1)] = -0.999
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
            target = schedule[current_date].copy()
            blocked = active & ~observed[i]
            if np.any(blocked):
                target[blocked] = weights[blocked]
                residual = max(0.0, 1.0 - float(target[blocked].sum()))
                tradable_target = ~blocked
                requested = float(target[tradable_target].sum())
                if requested > residual and requested > 0:
                    target[tradable_target] *= residual / requested
                for column in np.flatnonzero(blocked):
                    rejected_rows.append(
                        {
                            "signal_date": schedule_signal_dates[current_date],
                            "execution_date": current_date,
                            "ticker": asset_names[column],
                            "target_weight": float(schedule[current_date][column]),
                            "reason": "no_observed_quote_for_exit",
                        }
                    )
            target_cash = max(0.0, 1.0 - float(target.sum()))
            turnover = _one_way_turnover(weights, target, current_cash=cash, target_cash=target_cash)
            turnover_total += turnover
            capital *= 1.0 - turnover * config.transaction_cost_bps / 10_000.0
            weights = target
            cash = target_cash
        periodic.append(capital / before - 1.0)
        equity.append(capital)
    equity_array = np.asarray(equity)
    curve = pl.DataFrame({"trade_date": dates[start:], "portfolio_value": equity_array}).with_columns(
            pl.lit("adjusted_close").alias("price_basis")
        )
    summary = _metrics(
            equity_array,
            np.asarray(periodic),
            turnover_total,
            rebalance_periods=len(schedule),
            average_positions=float(np.mean(positions)) if positions else 0.0,
            execution_delay_bars=config.execution_delay_bars,
    )
    rejected = pl.DataFrame(rejected_rows) if rejected_rows else pl.DataFrame(
        schema={"signal_date": pl.Date, "execution_date": pl.Date, "ticker": pl.Utf8, "target_weight": pl.Float64, "reason": pl.Utf8}
    )
    return DetailedWeightedBacktest(curve, summary, rejected)
