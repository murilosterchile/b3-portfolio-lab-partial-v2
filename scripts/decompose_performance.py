from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl
from portfolio_core.research.performance import (
    annual_alpha_beta,
    annual_performance,
    concentration_statistics,
    contribution_decomposition,
    drawdown_episodes,
    equity_period_returns,
    issuer_contribution,
    regime_performance,
    rolling_excess_performance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("strategy", nargs="?", default="development_research_v2")
    args = parser.parse_args()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    root = data_dir / "gold" / "backtests" / args.strategy
    curve_path = root / "equity_curve.parquet"
    contrib_path = root / "contributions.parquet"
    if not curve_path.exists() or not contrib_path.exists():
        raise SystemExit(f"Missing detailed backtest outputs under {root}")
    curve = pl.read_parquet(curve_path)
    contributions = pl.read_parquet(contrib_path)
    selections_path = root / "selections.parquet"
    costs_path = root / "costs.parquet"
    selections = pl.read_parquet(selections_path) if selections_path.exists() else pl.DataFrame()
    costs = pl.read_parquet(costs_path) if costs_path.exists() else pl.DataFrame()
    feature_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    sector_map = pl.read_parquet(feature_path) if feature_path.exists() else None
    annual = annual_performance(curve)
    monthly = equity_period_returns(curve, period="1mo")
    by_ticker, by_sector = contribution_decomposition(contributions, sector_map=sector_map)
    by_issuer = issuer_contribution(contributions, selections)
    annual.write_csv(root / "annual_performance.csv")
    monthly.write_csv(root / "monthly_performance.csv")
    by_ticker.write_csv(root / "ticker_contributions.csv")
    if not by_sector.is_empty():
        by_sector.write_csv(root / "sector_contributions.csv")
    if not by_issuer.is_empty():
        by_issuer.write_csv(root / "issuer_contributions.csv")
    best_months = monthly.sort("return", descending=True).head(10)
    worst_months = monthly.sort("return").head(10)
    best_months.write_csv(root / "best_10_months.csv")
    worst_months.write_csv(root / "worst_10_months.csv")
    by_ticker.head(10).write_csv(root / "best_10_stocks.csv")
    by_ticker.tail(10).sort("sum_daily_return_contribution").write_csv(root / "worst_10_stocks.csv")
    concentration = concentration_statistics(annual, by_ticker)
    concentration["top_5_years_majority_flag"] = (
        concentration.get("fraction_positive_log_wealth_from_top_5_years") is not None
        and float(concentration["fraction_positive_log_wealth_from_top_5_years"]) > 0.50
    )
    concentration["top_10_tickers_majority_flag"] = (
        concentration.get("fraction_positive_position_contribution_from_top_10_tickers") is not None
        and float(concentration["fraction_positive_position_contribution_from_top_10_tickers"]) > 0.50
    )
    if not by_issuer.is_empty():
        positive_issuer = by_issuer["sum_daily_return_contribution"].to_numpy()
        positive_issuer = positive_issuer[positive_issuer > 0]
        concentration["fraction_positive_contribution_from_top_10_issuers"] = (
            float(sum(sorted(positive_issuer, reverse=True)[:10]) / positive_issuer.sum())
            if positive_issuer.size
            else None
        )
        concentration["top_10_issuers_majority_flag"] = (
            concentration["fraction_positive_contribution_from_top_10_issuers"] is not None
            and float(concentration["fraction_positive_contribution_from_top_10_issuers"]) > 0.50
        )
    (root / "concentration_statistics.json").write_text(
        json.dumps(concentration, indent=2), encoding="utf-8"
    )
    drawdowns = drawdown_episodes(curve)
    if not drawdowns.is_empty():
        drawdowns.write_csv(root / "drawdown_episodes.csv")
    backtest_root = data_dir / "gold" / "backtests"
    prefix = args.strategy.removesuffix("_research_v2")
    ibov_path = backtest_root / f"{prefix}_ibov_total_return" / "equity_curve.parquet"
    cdi_path = backtest_root / f"{prefix}_cdi" / "equity_curve.parquet"
    if ibov_path.exists():
        ibov = pl.read_parquet(ibov_path)
        rolling_excess_performance(curve, ibov).write_csv(root / "rolling_12_24_36m_excess_vs_ibov.csv")
        annual_alpha_beta(curve, ibov).write_csv(root / "annual_alpha_beta_vs_ibov.csv")
        if cdi_path.exists():
            regime_performance(curve, ibov, pl.read_parquet(cdi_path)).write_csv(
                root / "regime_performance.csv"
            )
    cost_report: dict[str, object] = {"status": "missing_cost_decomposition"}
    if not costs.is_empty():
        by_component = costs.group_by("component").agg(
            pl.col("cost_amount").sum().alias("cost_amount"),
            pl.col("cost_fraction").sum().alias("sum_cost_fraction"),
        ).sort("component")
        by_component.write_csv(root / "cost_contribution.csv")
        cost_report = {
            "status": "available",
            "total_cost_amount": float(costs["cost_amount"].sum()),
            "components": by_component.to_dicts(),
        }
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    attribution = {
        "annualized_turnover": summary.get("annualized_turnover"),
        "costs": cost_report,
        "benchmark_coverage": {
            "ibov_available": ibov_path.exists(),
            "cdi_available": cdi_path.exists(),
            "note": "Missing external histories are reported and never backfilled with synthetic proxies.",
        },
    }
    (root / "turnover_cost_attribution.json").write_text(
        json.dumps(attribution, indent=2, default=str), encoding="utf-8"
    )
    print(f"decomposition={root}")


if __name__ == "__main__":
    main()
