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
    sector_lookup = sector_map.select("ticker", "sector").unique(subset=["ticker"], keep="last")
    by_sector = contributions.join(sector_lookup, on="ticker", how="left").with_columns(
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
