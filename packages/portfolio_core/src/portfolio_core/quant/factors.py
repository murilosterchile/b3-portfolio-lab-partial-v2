from __future__ import annotations

from math import isfinite

import polars as pl


def momentum_score(frame: pl.DataFrame) -> pl.Expr:
    """12-1 momentum with medium-horizon confirmation (Jegadeesh/Titman style signal)."""
    return (
        0.60 * pl.col("rank_momentum_12_1")
        + 0.25 * pl.col("rank_return_126d")
        + 0.15 * pl.col("rank_return_63d")
    )


def low_volatility_score(frame: pl.DataFrame) -> pl.Expr:
    residual = (
        pl.col("rank_residual_volatility_63d")
        if "rank_residual_volatility_63d" in frame.columns
        else pl.col("rank_volatility_63d")
    )
    return 0.5 * (1.0 - pl.col("rank_volatility_63d")) + 0.5 * (1.0 - residual)


def quality_score(frame: pl.DataFrame) -> pl.Expr:
    """Point-in-time profitability/quality score when CVM features are available."""
    weighted: list[tuple[float, str]] = [
        (0.40, "rank_gross_profitability"),
        (0.25, "rank_roe_proxy"),
        (0.15, "rank_roa_proxy"),
        (0.10, "rank_equity_to_assets"),
        (0.10, "rank_asset_turnover"),
    ]
    available = []
    for weight, name in weighted:
        sector_name = f"sector_{name}"
        if sector_name in frame.columns:
            available.append((weight, sector_name))
        elif name in frame.columns:
            available.append((weight, name))
    if not available:
        return pl.lit(0.5)
    total = sum(weight for weight, _ in available)
    expression = pl.lit(0.0)
    for weight, name in available:
        expression = expression + (weight / total) * pl.col(name).fill_null(0.5)
    return expression


def composite_quant_score(
    frame: pl.DataFrame,
    *,
    sleeve_weights: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Robust multifactor score for screening/QKP, separate from the ML forecast.

    The registered baseline is equal-weight across robust sleeves. Challenger
    weights may be supplied only after development-only selection and are
    normalized here with no access to future outcomes.
    """
    configured = sleeve_weights or {"momentum": 1.0, "low_volatility": 1.0, "quality": 1.0}
    required = {"momentum", "low_volatility", "quality"}
    if set(configured) != required:
        raise ValueError(f"sleeve_weights must contain exactly {sorted(required)}")
    if any(
        not isinstance(value, (int, float)) or not isfinite(float(value)) or value < 0
        for value in configured.values()
    ):
        raise ValueError("sleeve weights must be finite and non-negative")
    total = float(sum(configured.values()))
    if total <= 0:
        raise ValueError("at least one sleeve weight must be positive")
    weights = {name: float(value) / total for name, value in configured.items()}
    return frame.with_columns(
        momentum_score(frame).alias("momentum_score"),
        low_volatility_score(frame).alias("low_volatility_score"),
        quality_score(frame).alias("quality_score"),
    ).with_columns(
        (weights["momentum"] * pl.col("momentum_score")).alias(
            "quant_contribution_momentum"
        ),
        (weights["low_volatility"] * pl.col("low_volatility_score")).alias(
            "quant_contribution_low_volatility"
        ),
        (weights["quality"] * pl.col("quality_score")).alias(
            "quant_contribution_quality"
        ),
        (
            pl.col("quant_contribution_momentum")
            + pl.col("quant_contribution_low_volatility")
            + pl.col("quant_contribution_quality")
        ).alias("quant_score")
    )


def learn_factor_sleeve_weights(
    development: pl.DataFrame,
    *,
    target_column: str = "target_excess_return",
    shrinkage_to_equal: float = 0.80,
    turnover_penalty: float = 0.10,
) -> dict[str, float]:
    """Learn a strongly shrunk sleeve challenger from past cross-sections only."""
    if not 0.0 <= shrinkage_to_equal <= 1.0:
        raise ValueError("shrinkage_to_equal must be in [0, 1]")
    scored = composite_quant_score(development)
    columns = {
        "momentum": "momentum_score",
        "low_volatility": "low_volatility_score",
        "quality": "quality_score",
    }
    strength: dict[str, float] = {}
    for sleeve, column in columns.items():
        correlations: list[float] = []
        for cross in scored.partition_by("trade_date", maintain_order=True):
            value = cross.select(pl.corr(column, target_column, method="spearman")).item()
            if value is not None and isfinite(float(value)):
                correlations.append(float(value))
        turnover_proxy = (
            scored.sort(["ticker", "trade_date"])
            .select(pl.col(column).diff().over("ticker").abs().mean())
            .item()
        )
        penalty = float(turnover_proxy) if turnover_proxy is not None else 0.0
        strength[sleeve] = max(0.0, sum(correlations) / max(len(correlations), 1)) / (
            1.0 + turnover_penalty * penalty
        )
    total = sum(strength.values())
    learned = (
        {name: value / total for name, value in strength.items()}
        if total > 0
        else {name: 1.0 / 3.0 for name in columns}
    )
    return {
        name: shrinkage_to_equal / 3.0 + (1.0 - shrinkage_to_equal) * learned[name]
        for name in columns
    }
