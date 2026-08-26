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
    "operating_cash_flow_to_assets",
    "equity_to_assets",
    "log_revenue",
    "log_assets",
    "log_equity",
    "fundamental_age_days",
    "negative_equity",
    "stale_fundamental",
]
FUNDAMENTAL_RANK_FEATURES = [
    f"rank_{name}"
    for name in FUNDAMENTAL_LEVEL_FEATURES
    if name not in {"fundamental_age_days", "negative_equity", "stale_fundamental"}
]
FUNDAMENTAL_FEATURES = FUNDAMENTAL_LEVEL_FEATURES + FUNDAMENTAL_RANK_FEATURES


_FLOW_ACCOUNTS = {
    "3.01": "revenue",
    "3.03": "gross_profit",
    "3.11": "net_income",
    "6.01": "operating_cash_flow",
}


def _point_in_time_flows(frame: pl.DataFrame) -> pl.DataFrame:
    """Build comparable TTM flows, falling back to the latest known fiscal year."""
    if not {"DT_INI_EXERC", "DT_FIM_EXERC"}.issubset(frame.columns):
        return pl.DataFrame()
    source = frame.filter(pl.col("CD_CONTA").is_in(list(_FLOW_ACCOUNTS))).select(
        "CD_CVM", "DT_RECEB", "DT_INI_EXERC", "DT_FIM_EXERC", "CD_CONTA", "VL_CONTA"
    ).drop_nulls(["DT_RECEB", "CD_CONTA", "VL_CONTA"])
    if source.is_empty():
        return pl.DataFrame()

    output: list[dict[str, object]] = []
    for company in source.partition_by("CD_CVM", maintain_order=True):
        issuer = str(company.get_column("CD_CVM").item(0))
        known: dict[tuple[str, object, object], float] = {}
        for receipt_frame in company.sort(["DT_RECEB", "DT_FIM_EXERC"]).partition_by(
            "DT_RECEB", maintain_order=True
        ):
            receipt = receipt_frame.get_column("DT_RECEB").item(0)
            for row in receipt_frame.iter_rows(named=True):
                known[(str(row["CD_CONTA"]), row["DT_INI_EXERC"], row["DT_FIM_EXERC"])] = float(
                    row["VL_CONTA"]
                )
            result: dict[str, object] = {"CD_CVM": issuer, "DT_RECEB": receipt}
            for account, name in _FLOW_ACCOUNTS.items():
                periods = [
                    (start, end, value)
                    for (code, start, end), value in known.items()
                    if code == account and start is not None and end is not None
                ]
                quarters: dict[object, float] = {}
                fiscal_years: list[tuple[object, float]] = []
                by_start: dict[object, list[tuple[object, float]]] = {}
                for start, end, value in periods:
                    by_start.setdefault(start, []).append((end, value))
                for start, cumulative in by_start.items():
                    ordered = sorted(cumulative, key=lambda item: item[0])
                    previous_value = 0.0
                    previous_end = None
                    for end, value in ordered:
                        duration = (end - start).days + 1
                        if previous_end is None or (end - previous_end).days > 120:
                            discrete = value if duration <= 120 else None
                        else:
                            discrete = value - previous_value
                        if discrete is not None:
                            quarters[end] = discrete
                        if duration >= 330:
                            fiscal_years.append((end, value))
                        previous_value = value
                        previous_end = end
                recent_quarters = sorted(quarters.items(), key=lambda item: item[0])[-4:]
                if len(recent_quarters) == 4:
                    result[name] = float(sum(value for _, value in recent_quarters))
                    result[f"{name}_basis"] = "TTM"
                elif fiscal_years:
                    result[name] = float(max(fiscal_years, key=lambda item: item[0])[1])
                    result[f"{name}_basis"] = "FY"
                else:
                    result[name] = None
                    result[f"{name}_basis"] = None
            output.append(result)
    return pl.DataFrame(output) if output else pl.DataFrame()


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

    frame = frame.with_columns(pl.col("CD_CVM").cast(pl.Utf8))
    snapshots = frame.group_by(["CD_CVM", "DT_REFER", "DT_RECEB"]).agg(
        pl.when(pl.col("CD_CONTA") == "1").then(pl.col("VL_CONTA")).max().alias("assets"),
        pl.when(pl.col("CD_CONTA") == "2.03").then(pl.col("VL_CONTA")).max().alias("equity"),
    )
    snapshots = snapshots.sort(["CD_CVM", "DT_RECEB", "DT_REFER"]).unique(
        subset=["CD_CVM", "DT_RECEB"], keep="last", maintain_order=True
    )
    flows = _point_in_time_flows(frame)
    if flows.is_empty():
        # Compatibility for governed legacy extracts that lack exercise-period metadata.
        flows = frame.group_by(["CD_CVM", "DT_RECEB"]).agg(
            *[
                pl.when(pl.col("CD_CONTA") == account)
                .then(pl.col("VL_CONTA"))
                .max()
                .alias(name)
                for account, name in _FLOW_ACCOUNTS.items()
            ]
        )
    snapshots = snapshots.join(flows, on=["CD_CVM", "DT_RECEB"], how="left", validate="1:1")

    def safe_ratio(numerator: str, denominator: str) -> pl.Expr:
        return (
            pl.when(pl.col(denominator).is_not_null() & (pl.col(denominator) > 0))
            .then(pl.col(numerator) / pl.col(denominator))
            .otherwise(None)
        )

    result = snapshots.with_columns(
        safe_ratio("net_income", "revenue").alias("net_margin"),
        safe_ratio("net_income", "equity").alias("roe_proxy"),
        safe_ratio("net_income", "assets").alias("roa_proxy"),
        safe_ratio("revenue", "assets").alias("asset_turnover"),
        safe_ratio("gross_profit", "revenue").alias("gross_margin"),
        safe_ratio("gross_profit", "assets").alias("gross_profitability"),
        safe_ratio("operating_cash_flow", "assets").alias("operating_cash_flow_to_assets"),
        safe_ratio("equity", "assets").alias("equity_to_assets"),
        pl.when(pl.col("revenue") > 0).then(pl.col("revenue").log1p()).alias("log_revenue"),
        pl.when(pl.col("assets") > 0).then(pl.col("assets").log1p()).alias("log_assets"),
        pl.when(pl.col("equity") > 0).then(pl.col("equity").log1p()).alias("log_equity"),
        (pl.col("equity").is_not_null() & (pl.col("equity") <= 0)).alias("negative_equity"),
    ).sort(["CD_CVM", "DT_RECEB", "DT_REFER"])

    # A restatement batch can publish several historical reference periods on the same day.
    # join_asof requires one deterministic value for each receipt date; use the newest reference
    # period present in that filing batch.
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
    *,
    stale_after_days: int = 180,
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
    joined = joined.with_columns(
        (pl.col("trade_date") - pl.col("DT_REFER")).dt.total_days().alias("fundamental_age_days")
    ).with_columns(
        (
            pl.col("fundamental_age_days").is_null()
            | (pl.col("fundamental_age_days") > stale_after_days)
        ).alias("stale_fundamental")
    )

    rankable = {name.removeprefix("rank_") for name in FUNDAMENTAL_RANK_FEATURES}
    available = [name for name in FUNDAMENTAL_LEVEL_FEATURES if name in joined.columns and name in rankable]
    for feature in available:
        joined = joined.with_columns(
            (
                pl.col(feature).rank(method="average").over("trade_date")
                / (pl.col(feature).is_not_null() & pl.col(feature).is_finite()).sum().over("trade_date")
            ).alias(f"rank_{feature}")
        )
    return joined
