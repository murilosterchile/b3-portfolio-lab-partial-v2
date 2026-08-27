from __future__ import annotations

import polars as pl
from portfolio_core.data_quality import DataQualityError, validate_prices

REQUIRED_PRICE_COLUMNS = {"trade_date", "ticker", "close", "trades"}


def build_technical_features(
    prices: pl.DataFrame,
    *,
    horizon_days: int = 21,
    execution_lag_bars: int = 1,
) -> pl.DataFrame:
    """Build leakage-safe daily features and executable forward-return labels.

    A signal computed with the close at time t is not allowed to execute at that
    same close. The target therefore starts at t+execution_lag_bars and ends
    horizon_days later. ``target_end_date`` is persisted so every temporal split
    can purge labels that overlap validation/test periods.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    if execution_lag_bars < 1:
        raise ValueError("execution_lag_bars must be >= 1 to prevent same-close leakage")
    missing = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")
    if "traded_value_brl" not in prices.columns and "volume" not in prices.columns:
        raise ValueError("Missing required financial-liquidity column: traded_value_brl")

    # ``volume`` is the legacy name of B3 VOLTOT, already scaled from cents to
    # BRL by the COTAHIST parser. Prefer the explicit name while accepting old
    # silver partitions without a destructive rewrite.
    if "traded_value_brl" not in prices.columns:
        prices = prices.with_columns(pl.col("volume").alias("traded_value_brl"))
    elif "volume" in prices.columns:
        prices = prices.with_columns(
            pl.coalesce("traded_value_brl", "volume").alias("traded_value_brl")
        )

    validate_prices(prices, context="technical feature prices")
    price_column = "adjusted_close" if "adjusted_close" in prices.columns else "close"
    if price_column == "adjusted_close":
        invalid_adjusted = prices.filter(
            pl.col(price_column).is_null()
            | ~pl.col(price_column).is_finite()
            | (pl.col(price_column) <= 0)
        )
        if not invalid_adjusted.is_empty():
            raise DataQualityError(
                "adjusted analytical prices contain invalid values: "
                f"{invalid_adjusted.select('ticker', 'trade_date', price_column).head(10).to_dicts()}"
            )

    has_distribution = "distribution_number" in prices.columns and price_column == "close"

    def same_distribution(lag: int) -> pl.Expr:
        if not has_distribution:
            return pl.lit(True)
        return pl.col("distribution_number") == pl.col("distribution_number").shift(lag).over("ticker")

    px = pl.col(price_column)
    end_offset = execution_lag_bars + horizon_days
    frame = prices.sort(["ticker", "trade_date"]).with_columns(
        pl.when(same_distribution(1))
        .then(px.pct_change().over("ticker"))
        .otherwise(None)
        .alias("ret_1d"),
        px.shift(-execution_lag_bars).over("ticker").alias("target_start_close"),
        px.shift(-end_offset).over("ticker").alias("target_end_close"),
        pl.col("trade_date").shift(-execution_lag_bars).over("ticker").alias("target_start_date"),
        pl.col("trade_date").shift(-end_offset).over("ticker").alias("target_end_date"),
    )

    for days in (5, 21, 63, 126, 252):
        frame = frame.with_columns(
            pl.when(same_distribution(days))
            .then(px / px.shift(days).over("ticker") - 1.0)
            .otherwise(None)
            .alias(f"return_{days}d"),
            pl.when(same_distribution(days - 1))
            .then(pl.col("ret_1d").rolling_std(days).over("ticker"))
            .otherwise(None)
            .alias(f"volatility_{days}d"),
        )

    if has_distribution:
        target_same_distribution = (
            pl.col("distribution_number").shift(-execution_lag_bars).over("ticker")
            == pl.col("distribution_number").shift(-end_offset).over("ticker")
        )
    else:
        target_same_distribution = pl.lit(True)

    frame = frame.with_columns(
        pl.when(same_distribution(20))
        .then(px / px.rolling_mean(21).over("ticker") - 1.0)
        .otherwise(None)
        .alias("distance_sma_21"),
        pl.when(same_distribution(62))
        .then(px / px.rolling_mean(63).over("ticker") - 1.0)
        .otherwise(None)
        .alias("distance_sma_63"),
        pl.when(same_distribution(251))
        .then(px / px.rolling_mean(252).over("ticker") - 1.0)
        .otherwise(None)
        .alias("distance_sma_252"),
        pl.when(
            pl.lit(True)
            if not has_distribution
            else pl.col("distribution_number").shift(21).over("ticker")
            == pl.col("distribution_number").shift(252).over("ticker")
        )
        .then(px.shift(21).over("ticker") / px.shift(252).over("ticker") - 1.0)
        .otherwise(None)
        .alias("momentum_12_1"),
        pl.col("traded_value_brl").log1p().alias("log_traded_value"),
        pl.col("traded_value_brl").log1p().alias("log_volume"),  # legacy alias
        pl.col("traded_value_brl")
        .rolling_mean(21)
        .over("ticker")
        .log1p()
        .alias("log_volume_21d"),  # legacy alias
        pl.col("trades").rolling_mean(21).over("ticker").log1p().alias("log_trades_21d"),
        (-pl.col("return_5d")).alias("short_term_reversal_5d"),
        (0.5 * pl.col("return_63d") + 0.5 * pl.col("return_126d")).alias(
            "medium_term_momentum"
        ),
        pl.col("traded_value_brl")
        .rolling_mean(21)
        .over("ticker")
        .log1p()
        .alias("log_traded_value_21d"),
        (
            (pl.col("ret_1d").abs() / pl.col("traded_value_brl").clip(lower_bound=1e-12))
            .rolling_mean(21)
            .over("ticker")
        ).alias("amihud_illiquidity_21d"),
        pl.when(target_same_distribution)
        .then(pl.col("target_end_close") / pl.col("target_start_close") - 1.0)
        .otherwise(None)
        .alias("future_return"),
    )

    frame = frame.with_columns(
        pl.col("ret_1d").mean().over("trade_date").alias("market_return_1d"),
    ).with_columns(
        (pl.col("ret_1d") - pl.col("market_return_1d")).alias("market_residual_1d"),
        pl.lit(horizon_days).alias("target_horizon_bars"),
        pl.lit(execution_lag_bars).alias("execution_lag_bars"),
    )

    frame = frame.with_columns(
        pl.col("market_residual_1d").rolling_std(63).over("ticker").alias(
            "residual_volatility_63d"
        )
    )

    rank_features = [
        "return_21d",
        "return_63d",
        "return_126d",
        "momentum_12_1",
        "volatility_63d",
        "distance_sma_63",
        "log_volume_21d",
        "short_term_reversal_5d",
        "medium_term_momentum",
        "residual_volatility_63d",
        "log_traded_value_21d",
        "amihud_illiquidity_21d",
    ]
    for feature in rank_features:
        frame = frame.with_columns(
            (
                pl.col(feature).rank(method="average").over("trade_date")
                / (pl.col(feature).is_not_null() & pl.col(feature).is_finite()).sum().over("trade_date")
            ).alias(f"rank_{feature}")
        )

    return frame.drop("target_start_close", "target_end_close").with_columns(
        pl.lit(price_column == "adjusted_close").alias("uses_total_return_adjusted_price")
    )


def monthly_snapshots(features: pl.DataFrame) -> pl.DataFrame:
    """Use one common market date per month for every cross-section.

    The previous implementation picked each ticker's own last observation of the
    month, mixing different dates inside one ranking/backtest cross-section. This
    version first finds the common market month-end date and then keeps only rows
    observed on that date. Stale observations are not silently forward-filled.
    """
    if features.is_empty():
        return features
    with_month = features.with_columns(pl.col("trade_date").dt.truncate("1mo").alias("month"))
    month_ends = with_month.group_by("month").agg(
        pl.col("trade_date").max().alias("month_end_trade_date")
    )
    result = (
        with_month.join(month_ends, on="month", how="left")
        .filter(pl.col("trade_date") == pl.col("month_end_trade_date"))
        .drop("month", "month_end_trade_date")
        .sort(["trade_date", "ticker"])
    )
    duplicates = result.group_by(["trade_date", "ticker"]).len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise DataQualityError("monthly snapshots contain duplicate ticker/date rows")
    return result
