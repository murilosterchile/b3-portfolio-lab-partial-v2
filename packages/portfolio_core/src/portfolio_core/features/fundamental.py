from __future__ import annotations

import polars as pl
from portfolio_core.data.issuer_bridge import validate_issuer_bridge
from portfolio_core.data_quality import (
    DataQualityError,
    validate_fundamentals_point_in_time,
    validate_unique_rows,
)

FUNDAMENTAL_LEVEL_FEATURES = [
    "net_margin",
    "roe_proxy",
    "roa_proxy",
    "asset_turnover",
    "gross_margin",
    "gross_profitability",
    "equity_to_assets",
    "log_revenue",
    "log_assets",
    "log_equity",
]
FUNDAMENTAL_RANK_FEATURES = [f"rank_{name}" for name in FUNDAMENTAL_LEVEL_FEATURES]
FUNDAMENTAL_FEATURES = FUNDAMENTAL_LEVEL_FEATURES + FUNDAMENTAL_RANK_FEATURES


def build_company_fundamental_snapshots(statements: pl.DataFrame) -> pl.DataFrame:
    """Create conservative point-in-time accounting features from standardized CVM accounts.

    The feature set intentionally favors scale-free quality/profitability descriptors. Gross
    profitability (gross profit/assets) follows the economic intuition documented by Novy-Marx;
    cross-sectional ranks are added only after the point-in-time ticker join.
    """
    required = {"CD_CVM", "DT_REFER", "DT_RECEB", "CD_CONTA", "VL_CONTA"}
    missing = required - set(statements.columns)
    if missing:
        raise ValueError(f"Missing CVM columns: {sorted(missing)}")
    frame = statements
    if "ORDEM_EXERC" in frame.columns:
        frame = frame.filter(
            pl.col("ORDEM_EXERC").cast(pl.Utf8).str.to_uppercase().str.contains("ULT|ÚLT")
        )
    if "VERSAO" in frame.columns:
        version_keys = ["CD_CVM", "DT_REFER", "DT_RECEB"]
        frame = frame.with_columns(
            pl.col("VERSAO").cast(pl.Int64, strict=False).fill_null(-1).alias("_version_order")
        ).filter(pl.col("_version_order") == pl.col("_version_order").max().over(version_keys))

    snapshots = frame.group_by(["CD_CVM", "DT_REFER", "DT_RECEB"]).agg(
        pl.when(pl.col("CD_CONTA") == "3.01").then(pl.col("VL_CONTA")).max().alias("revenue"),
        pl.when(pl.col("CD_CONTA") == "3.03").then(pl.col("VL_CONTA")).max().alias("gross_profit"),
        pl.when(pl.col("CD_CONTA") == "3.11").then(pl.col("VL_CONTA")).max().alias("net_income"),
        pl.when(pl.col("CD_CONTA") == "1").then(pl.col("VL_CONTA")).max().alias("assets"),
        pl.when(pl.col("CD_CONTA") == "2.03").then(pl.col("VL_CONTA")).max().alias("equity"),
    )

    def safe_ratio(numerator: str, denominator: str) -> pl.Expr:
        return (
            pl.when(pl.col(denominator).is_not_null() & (pl.col(denominator).abs() > 0))
            .then(pl.col(numerator) / pl.col(denominator).abs())
            .otherwise(None)
        )

    result = snapshots.with_columns(
        safe_ratio("net_income", "revenue").alias("net_margin"),
        safe_ratio("net_income", "equity").alias("roe_proxy"),
        safe_ratio("net_income", "assets").alias("roa_proxy"),
        safe_ratio("revenue", "assets").alias("asset_turnover"),
        safe_ratio("gross_profit", "revenue").alias("gross_margin"),
        safe_ratio("gross_profit", "assets").alias("gross_profitability"),
        safe_ratio("equity", "assets").alias("equity_to_assets"),
        pl.col("revenue").abs().log1p().alias("log_revenue"),
        pl.col("assets").abs().log1p().alias("log_assets"),
        pl.col("equity").abs().log1p().alias("log_equity"),
    ).sort(["CD_CVM", "DT_RECEB", "DT_REFER"])

    # A restatement batch can publish several historical reference periods on the same day.
    # join_asof requires one deterministic value for each receipt date; use the newest reference
    # period present in that filing batch.
    result = result.unique(subset=["CD_CVM", "DT_RECEB"], keep="last", maintain_order=True)
    validate_unique_rows(result, ["CD_CVM", "DT_RECEB"], context="fundamental snapshots")
    impossible = result.filter(pl.col("DT_REFER") > pl.col("DT_RECEB"))
    if not impossible.is_empty():
        raise DataQualityError(
            "fundamental snapshots contain reference dates after receipt dates: "
            f"{impossible.select('CD_CVM', 'DT_REFER', 'DT_RECEB').head(10).to_dicts()}"
        )
    return result


def attach_fundamentals_point_in_time(
    monthly_technical: pl.DataFrame,
    fundamentals: pl.DataFrame,
    bridge: pl.DataFrame,
) -> pl.DataFrame:
    """Attach only fundamentals public and mapped to the ticker at prediction time.

    After the as-of join, each accounting descriptor also receives a same-date cross-sectional
    percentile rank. This is scale-robust, uses no future observation and is better aligned with
    stock-selection models than relying only on raw accounting magnitudes.
    """
    validate_unique_rows(
        monthly_technical, ["ticker", "trade_date"], context="monthly technical features"
    )
    validate_issuer_bridge(bridge)
    technical = monthly_technical.join(bridge, on="ticker", how="inner").filter(
        (pl.col("trade_date") >= pl.col("valid_from"))
        & (pl.col("trade_date") <= pl.col("valid_to"))
    )
    validate_unique_rows(technical, ["ticker", "trade_date"], context="ticker/CD_CVM join")
    technical = technical.sort(["CD_CVM", "trade_date"])
    fundamentals = fundamentals.with_columns(pl.col("CD_CVM").cast(pl.Utf8)).sort(
        ["CD_CVM", "DT_RECEB"]
    )
    joined = technical.join_asof(
        fundamentals,
        left_on="trade_date",
        right_on="DT_RECEB",
        by="CD_CVM",
        strategy="backward",
        check_sortedness=False,
    )
    validate_unique_rows(joined, ["ticker", "trade_date"], context="fundamentals point-in-time join")
    validate_fundamentals_point_in_time(joined)

    available = [name for name in FUNDAMENTAL_LEVEL_FEATURES if name in joined.columns]
    for feature in available:
        joined = joined.with_columns(
            (
                pl.col(feature).rank(method="average").over("trade_date")
                / (pl.col(feature).is_not_null() & pl.col(feature).is_finite()).sum().over("trade_date")
            ).alias(f"rank_{feature}")
        )
    return joined
