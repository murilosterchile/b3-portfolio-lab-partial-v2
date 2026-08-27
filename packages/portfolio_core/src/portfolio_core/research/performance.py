from __future__ import annotations

import numpy as np
import polars as pl


def equity_period_returns(curve: pl.DataFrame, *, period: str) -> pl.DataFrame:
    if period not in {"1mo", "1y"}:
        raise ValueError("period must be 1mo or 1y")
    return (
        curve.with_columns(pl.col("trade_date").dt.truncate(period).alias("period"))
        .group_by("period")
        .agg(
            pl.col("portfolio_value").first().alias("start_value"),
            pl.col("portfolio_value").last().alias("end_value"),
        )
        .sort("period")
        .with_columns((pl.col("end_value") / pl.col("start_value") - 1.0).alias("return"))
    )


def annual_performance(curve: pl.DataFrame) -> pl.DataFrame:
    daily = curve.sort("trade_date").with_columns(
        (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1.0).alias("daily_return")
    ).drop_nulls("daily_return")
    rows: list[dict[str, object]] = []
    for year in sorted(set(daily["trade_date"].dt.year().to_list())):
        group = daily.filter(pl.col("trade_date").dt.year() == year)
        returns = group["daily_return"].to_numpy()
        if len(returns) == 0:
            continue
        wealth = np.cumprod(1.0 + returns)
        running = np.maximum.accumulate(wealth)
        vol = float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0
        rows.append(
            {
                "year": year,
                "annual_return": float(np.prod(1.0 + returns) - 1.0),
                "sharpe": float(np.mean(returns) * 252 / vol) if vol > 1e-12 else 0.0,
                "max_drawdown": float(np.min(wealth / running - 1.0)),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def contribution_decomposition(
    contributions: pl.DataFrame,
    *,
    sector_map: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if contributions.is_empty():
        return pl.DataFrame(), pl.DataFrame()
    by_ticker = contributions.group_by("ticker").agg(
        pl.col("return_contribution").sum().alias("sum_daily_return_contribution"),
        pl.len().alias("held_days"),
    ).sort("sum_daily_return_contribution", descending=True)
    if sector_map is None or "sector" not in sector_map.columns:
        return by_ticker, pl.DataFrame()
    if "trade_date" in sector_map.columns:
        attributed = contributions.sort(["ticker", "trade_date"]).join_asof(
            sector_map.select("trade_date", "ticker", "sector")
            .drop_nulls("sector")
            .sort(["ticker", "trade_date"]),
            on="trade_date",
            by="ticker",
            strategy="backward",
        )
    else:
        sector_lookup = sector_map.select("ticker", "sector").unique(
            subset=["ticker"], keep="last"
        )
        attributed = contributions.join(sector_lookup, on="ticker", how="left")
    by_sector = attributed.with_columns(
        pl.col("sector").fill_null("Unknown")
    ).group_by("sector").agg(
        pl.col("return_contribution").sum().alias("sum_daily_return_contribution")
    ).sort("sum_daily_return_contribution", descending=True)
    return by_ticker, by_sector


def concentration_statistics(
    annual: pl.DataFrame,
    by_ticker: pl.DataFrame,
) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "fraction_positive_log_wealth_from_top_5_years": None,
        "fraction_positive_position_contribution_from_top_10_tickers": None,
    }
    if not annual.is_empty():
        positive = np.log1p(np.maximum(annual["annual_return"].to_numpy(), -0.999999))
        positive = positive[positive > 0]
        if positive.size:
            result["fraction_positive_log_wealth_from_top_5_years"] = float(
                np.sort(positive)[-5:].sum() / positive.sum()
            )
    if not by_ticker.is_empty():
        values = by_ticker["sum_daily_return_contribution"].to_numpy()
        values = values[values > 0]
        if values.size:
            result["fraction_positive_position_contribution_from_top_10_tickers"] = float(
                np.sort(values)[-10:].sum() / values.sum()
            )
    return result


def rolling_excess_performance(
    strategy_curve: pl.DataFrame,
    benchmark_curve: pl.DataFrame,
    *,
    windows: tuple[int, ...] = (12, 24, 36),
) -> pl.DataFrame:
    strategy = equity_period_returns(strategy_curve, period="1mo").select(
        pl.col("period").alias("month"), pl.col("return").alias("strategy_return")
    )
    benchmark = equity_period_returns(benchmark_curve, period="1mo").select(
        pl.col("period").alias("month"), pl.col("return").alias("benchmark_return")
    )
    aligned = strategy.join(benchmark, on="month", how="inner").sort("month")
    rows: list[dict[str, object]] = []
    strategy_values = aligned["strategy_return"].to_numpy()
    benchmark_values = aligned["benchmark_return"].to_numpy()
    months = aligned["month"].to_list()
    for window in windows:
        for end in range(window - 1, len(months)):
            start = end - window + 1
            strategy_return = float(np.prod(1.0 + strategy_values[start : end + 1]) - 1.0)
            benchmark_return = float(np.prod(1.0 + benchmark_values[start : end + 1]) - 1.0)
            rows.append(
                {
                    "month": months[end],
                    "window_months": window,
                    "strategy_return": strategy_return,
                    "benchmark_return": benchmark_return,
                    "excess_return": strategy_return - benchmark_return,
                }
            )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def annual_alpha_beta(strategy_curve: pl.DataFrame, benchmark_curve: pl.DataFrame) -> pl.DataFrame:
    strategy = strategy_curve.sort("trade_date").with_columns(
        (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1.0).alias("strategy")
    ).select("trade_date", "strategy")
    benchmark = benchmark_curve.sort("trade_date").with_columns(
        (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1.0).alias("benchmark")
    ).select("trade_date", "benchmark")
    aligned = strategy.join(benchmark, on="trade_date", how="inner").drop_nulls()
    rows: list[dict[str, object]] = []
    for year in sorted(set(aligned["trade_date"].dt.year().to_list())):
        group = aligned.filter(pl.col("trade_date").dt.year() == year)
        x = group["benchmark"].to_numpy()
        y = group["strategy"].to_numpy()
        variance = float(np.var(x, ddof=1)) if len(x) > 1 else 0.0
        beta = float(np.cov(y, x, ddof=1)[0, 1] / variance) if variance > 1e-15 else float("nan")
        alpha = float((np.mean(y) - beta * np.mean(x)) * 252.0) if np.isfinite(beta) else float("nan")
        rows.append({"year": year, "annualized_alpha": alpha, "beta_vs_ibov": beta, "observations": len(x)})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def drawdown_episodes(curve: pl.DataFrame) -> pl.DataFrame:
    ordered = curve.sort("trade_date")
    values = ordered["portfolio_value"].to_numpy()
    dates = ordered["trade_date"].to_list()
    rows: list[dict[str, object]] = []
    peak = 0
    trough = 0
    in_drawdown = False
    for index in range(1, len(values)):
        if values[index] >= values[peak]:
            if in_drawdown:
                rows.append(
                    {
                        "peak_date": dates[peak],
                        "trough_date": dates[trough],
                        "recovery_date": dates[index],
                        "max_drawdown": float(values[trough] / values[peak] - 1.0),
                        "duration_sessions": index - peak,
                        "recovered": True,
                    }
                )
            peak = index
            trough = index
            in_drawdown = False
        else:
            in_drawdown = True
            if values[index] < values[trough]:
                trough = index
    if in_drawdown:
        rows.append(
            {
                "peak_date": dates[peak],
                "trough_date": dates[trough],
                "recovery_date": None,
                "max_drawdown": float(values[trough] / values[peak] - 1.0),
                "duration_sessions": len(values) - 1 - peak,
                "recovered": False,
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def regime_performance(
    strategy_curve: pl.DataFrame,
    ibov_curve: pl.DataFrame,
    cdi_curve: pl.DataFrame,
) -> pl.DataFrame:
    strategy = equity_period_returns(strategy_curve, period="1mo").select(
        pl.col("period").alias("month"), pl.col("return").alias("strategy")
    )
    ibov = equity_period_returns(ibov_curve, period="1mo").select(
        pl.col("period").alias("month"), pl.col("return").alias("ibov")
    )
    cdi = equity_period_returns(cdi_curve, period="1mo").select(
        pl.col("period").alias("month"), pl.col("return").alias("cdi")
    )
    aligned = strategy.join(ibov, on="month", how="inner").join(cdi, on="month", how="inner").sort("month")
    values = aligned.select("strategy", "ibov", "cdi").to_numpy()
    labels: list[tuple[str, str, float]] = []
    observations: list[dict[str, object]] = []
    for index in range(12, len(values)):
        past_market = values[index - 12 : index, 1]
        past_vol = float(np.std(values[max(0, index - 3) : index, 1], ddof=1))
        past_rate = float(np.mean(values[max(0, index - 3) : index, 2]))
        prior_vols = [float(np.std(values[max(0, j - 3) : j, 1], ddof=1)) for j in range(3, index)]
        prior_rates = [float(np.mean(values[max(0, j - 3) : j, 2])) for j in range(3, index)]
        labels = [
            ("market", "bull" if np.prod(1.0 + past_market) - 1.0 >= 0 else "bear", values[index, 0] - values[index, 1]),
            ("volatility", "high_vol" if past_vol >= np.median(prior_vols) else "low_vol", values[index, 0] - values[index, 1]),
            ("rates", "high_rate" if past_rate >= np.median(prior_rates) else "low_rate", values[index, 0] - values[index, 2]),
        ]
        for regime_type, regime, excess in labels:
            observations.append({"regime_type": regime_type, "regime": regime, "monthly_excess": float(excess)})
    if not observations:
        return pl.DataFrame()
    return pl.DataFrame(observations).group_by("regime_type", "regime").agg(
        pl.len().alias("months"),
        pl.col("monthly_excess").mean().alias("mean_monthly_excess"),
        (pl.col("monthly_excess") > 0).mean().alias("excess_hit_rate"),
    ).sort("regime_type", "regime")


def issuer_contribution(
    contributions: pl.DataFrame, selections: pl.DataFrame
) -> pl.DataFrame:
    if contributions.is_empty() or selections.is_empty() or "issuer_id" not in selections.columns:
        return pl.DataFrame()
    lookup = selections.select(
        pl.col("execution_date").alias("trade_date"), "ticker", "issuer_id"
    ).sort(["ticker", "trade_date"])
    return contributions.sort(["ticker", "trade_date"]).join_asof(
        lookup, on="trade_date", by="ticker", strategy="backward"
    ).with_columns(
        pl.col("issuer_id").fill_null(pl.col("ticker"))
    ).group_by("issuer_id").agg(
        pl.col("return_contribution").sum().alias("sum_daily_return_contribution")
    ).sort("sum_daily_return_contribution", descending=True)
