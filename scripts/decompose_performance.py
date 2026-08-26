from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl
from portfolio_core.research.performance import (
    annual_performance,
    concentration_statistics,
    contribution_decomposition,
    equity_period_returns,
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
    feature_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    sector_map = pl.read_parquet(feature_path) if feature_path.exists() else None
    annual = annual_performance(curve)
    monthly = equity_period_returns(curve, period="1mo")
    by_ticker, by_sector = contribution_decomposition(contributions, sector_map=sector_map)
    annual.write_csv(root / "annual_performance.csv")
    monthly.write_csv(root / "monthly_performance.csv")
    by_ticker.write_csv(root / "ticker_contributions.csv")
    if not by_sector.is_empty():
        by_sector.write_csv(root / "sector_contributions.csv")
    best_months = monthly.sort("return", descending=True).head(10)
    worst_months = monthly.sort("return").head(10)
    best_months.write_csv(root / "best_10_months.csv")
    worst_months.write_csv(root / "worst_10_months.csv")
    by_ticker.head(10).write_csv(root / "best_10_stocks.csv")
    by_ticker.tail(10).sort("sum_daily_return_contribution").write_csv(root / "worst_10_stocks.csv")
    concentration = concentration_statistics(annual, by_ticker)
    (root / "concentration_statistics.json").write_text(
        json.dumps(concentration, indent=2), encoding="utf-8"
    )
    print(f"decomposition={root}")


if __name__ == "__main__":
    main()
