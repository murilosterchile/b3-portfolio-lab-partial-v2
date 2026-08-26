from __future__ import annotations

import polars as pl


def momentum_score(frame: pl.DataFrame) -> pl.Expr:
    """12-1 momentum with medium-horizon confirmation (Jegadeesh/Titman style signal)."""
    return (
        0.60 * pl.col("rank_momentum_12_1")
        + 0.25 * pl.col("rank_return_126d")
        + 0.15 * pl.col("rank_return_63d")
    )


def low_volatility_score(frame: pl.DataFrame) -> pl.Expr:
    return 1.0 - pl.col("rank_volatility_63d")


def quality_score(frame: pl.DataFrame) -> pl.Expr:
    """Point-in-time profitability/quality score when CVM features are available."""
    weighted: list[tuple[float, str]] = [
        (0.40, "rank_gross_profitability"),
        (0.25, "rank_roe_proxy"),
        (0.15, "rank_roa_proxy"),
        (0.10, "rank_equity_to_assets"),
        (0.10, "rank_asset_turnover"),
    ]
    available = [(weight, name) for weight, name in weighted if name in frame.columns]
    if not available:
        return pl.lit(0.5)
    total = sum(weight for weight, _ in available)
    expression = pl.lit(0.0)
    for weight, name in available:
        expression = expression + (weight / total) * pl.col(name).fill_null(0.5)
    return expression


def composite_quant_score(frame: pl.DataFrame) -> pl.DataFrame:
    """Robust multifactor score for screening/QKP, separate from the ML forecast.

    Momentum remains the largest component; low volatility controls risk, and point-in-time CVM
    profitability adds a quality dimension when available. Missing fundamentals do not receive a
    penalty; their quality component is neutral (0.5).
    """
    return frame.with_columns(
        momentum_score(frame).alias("momentum_score"),
        low_volatility_score(frame).alias("low_volatility_score"),
        quality_score(frame).alias("quality_score"),
    ).with_columns(
        (
            0.50 * pl.col("momentum_score")
            + 0.20 * pl.col("low_volatility_score")
            + 0.30 * pl.col("quality_score")
        ).alias("quant_score")
    )
