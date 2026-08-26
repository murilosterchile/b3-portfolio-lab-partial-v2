from __future__ import annotations

import polars as pl
from portfolio_core.data_quality import DataQualityError, validate_prices

REQUIRED_PRICE_COLUMNS = {"trade_date", "ticker", "close", "volume", "trades"}


def build_technical_features(prices: pl.DataFrame, *, horizon_days: int = 63) -> pl.DataFrame:
    """Build leakage-safe daily features and a forward relative-return label.

    If the governed corporate-action pipeline has produced ``adjusted_close``, all return/trend
    features and labels use it. Otherwise the function falls back to raw ``close`` and masks windows
    that cross a COTAHIST ``distribution_number`` transition. This fallback remains useful for data
    diagnostics but is not considered total-return research.
    """
    missing = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f"Missing required price columns: {sorted(missing)}")

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

    # Corporate-action windows only need masking when the analytical series is still raw.
    has_distribution = "distribution_number" in prices.columns and price_column == "close"

    def same_distribution(lag: int) -> pl.Expr:
        if not has_distribution:
            return pl.lit(True)
        return pl.col("distribution_number") == pl.col("distribution_number").shift(lag).over("ticker")

    px = pl.col(price_column)
    frame = prices.sort(["ticker", "trade_date"]).with_columns(
        pl.when(same_distribution(1))
        .then(px.pct_change().over("ticker"))
        .otherwise(None)
        .alias("ret_1d"),
        px.shift(-horizon_days).over("ticker").alias("future_close"),
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
        pl.col("volume").log1p().alias("log_volume"),
        pl.col("volume").rolling_mean(21).over("ticker").log1p().alias("log_volume_21d"),
        pl.col("trades").rolling_mean(21).over("ticker").log1p().alias("log_trades_21d"),
        pl.when(same_distribution(-horizon_days))
        .then(pl.col("future_close") / px - 1.0)
        .otherwise(None)
        .alias("future_return"),
    )

    frame = frame.with_columns(
        pl.col("future_return").mean().over("trade_date").alias("market_forward_return")
    ).with_columns(
        (pl.col("future_return") - pl.col("market_forward_return")).alias("target_excess_return")
    )

    # Cross-sectional percentile ranks are robust to scale/regime shifts and align with the
    # ranking nature of stock selection. They use only contemporaneous observations.
    rank_features = [
        "return_21d",
        "return_63d",
        "return_126d",
        "momentum_12_1",
        "volatility_63d",
        "distance_sma_63",
        "log_volume_21d",
    ]
    for feature in rank_features:
        frame = frame.with_columns(
            (
                pl.col(feature).rank(method="average").over("trade_date")
                / pl.len().over("trade_date")
            ).alias(f"rank_{feature}")
        )

    return frame.drop("future_close").with_columns(
        pl.lit(price_column == "adjusted_close").alias("uses_total_return_adjusted_price")
    )


def monthly_snapshots(features: pl.DataFrame) -> pl.DataFrame:
    """Keep the last available trading observation for each ticker/month."""
    return (
        features.with_columns(pl.col("trade_date").dt.truncate("1mo").alias("month"))
        .sort(["ticker", "trade_date"])
        .group_by(["ticker", "month"], maintain_order=True)
        .tail(1)
        .drop("month")
        .sort(["trade_date", "ticker"])
    )
