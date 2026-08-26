from __future__ import annotations

import polars as pl


REGIME_CANDIDATE_FEATURES = [
    "regime_market_return_21d",
    "regime_market_return_63d",
    "regime_market_realized_vol_21d",
    "regime_breadth_1d",
    "regime_cross_sectional_dispersion_1d",
    "regime_market_drawdown",
    "regime_selic_target",
    "regime_selic_change",
    "regime_momentum_x_market_vol",
    "regime_volatility_x_market_vol",
]


def build_market_regime_candidates(daily_features: pl.DataFrame) -> pl.DataFrame:
    """Small PIT market-regime candidate set; not enabled in production by default."""
    market = daily_features.group_by("trade_date").agg(
        pl.col("ret_1d").mean().alias("regime_market_return_1d"),
        pl.col("return_21d").mean().alias("regime_market_return_21d"),
        pl.col("return_63d").mean().alias("regime_market_return_63d"),
        (pl.col("ret_1d") > 0).mean().alias("regime_breadth_1d"),
        pl.col("ret_1d").std().alias("regime_cross_sectional_dispersion_1d"),
    ).sort("trade_date")
    market = market.with_columns(
        pl.col("regime_market_return_1d")
        .rolling_std(21)
        .alias("regime_market_realized_vol_21d"),
        (1.0 + pl.col("regime_market_return_1d"))
        .cum_prod()
        .alias("__market_index"),
    ).with_columns(
        (pl.col("__market_index") / pl.col("__market_index").cum_max() - 1.0).alias(
            "regime_market_drawdown"
        )
    ).drop("__market_index")
    return market


def attach_selic_candidate(regime: pl.DataFrame, selic_sgs: pl.DataFrame) -> pl.DataFrame:
    """Attach SELIC conservatively one calendar day after the SGS observation date.

    This avoids assuming that a same-date Copom/SGS update was known before the
    B3 signal close. IPCA is intentionally excluded because the current raw SGS
    materialization does not retain a release timestamp for the reference month.
    """
    safe = selic_sgs.select(
        (pl.col("date") + pl.duration(days=1)).alias("available_at"),
        pl.col("value").alias("regime_selic_target"),
    ).sort("available_at")
    joined = regime.sort("trade_date").join_asof(
        safe,
        left_on="trade_date",
        right_on="available_at",
        strategy="backward",
    ).drop("available_at")
    return joined.with_columns(
        pl.col("regime_selic_target").diff().fill_null(0.0).alias("regime_selic_change")
    )


def attach_regime_candidates(panel: pl.DataFrame, regime: pl.DataFrame) -> pl.DataFrame:
    result = panel.join(regime, on="trade_date", how="left")
    return result.with_columns(
        (
            pl.col("momentum_12_1") * pl.col("regime_market_realized_vol_21d")
        ).alias("regime_momentum_x_market_vol"),
        (
            pl.col("volatility_63d") * pl.col("regime_market_realized_vol_21d")
        ).alias("regime_volatility_x_market_vol"),
    )
