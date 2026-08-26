from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import polars as pl
from portfolio_core.backtest import (
    WeightedBacktestConfig,
    build_monthly_risk_benchmark_weights,
    run_monthly_weighted_backtest,
)
from portfolio_core.ml.protocol import ResearchProtocol


def main() -> None:
    protocol = ResearchProtocol()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    prices_parts = sorted(
        (data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet")
    )
    feature_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    if not feature_path.exists():
        feature_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    if not prices_parts or not feature_path.exists():
        raise SystemExit("Adjusted prices and monthly features are required")
    prices = pl.concat([pl.read_parquet(path) for path in prices_parts], how="vertical_relaxed")
    features = pl.read_parquet(feature_path).filter(
        pl.col("trade_date").dt.year() <= protocol.development_end_year
    )
    output: dict[str, object] = {
        "development_end_year": protocol.development_end_year,
        "diagnostic_year_excluded": protocol.diagnostic_year,
        "warning": protocol.warning,
        "benchmarks": {},
    }
    out_root = data_dir / "gold" / "backtests" / "risk_benchmarks"
    out_root.mkdir(parents=True, exist_ok=True)
    for estimator in ("sample", "ledoit_wolf"):
        for allocation in ("minvar", "inverse_vol", "hrp"):
            name = f"{allocation}_{estimator}"
            print(name)
            weights = build_monthly_risk_benchmark_weights(
                prices,
                features,
                allocation=allocation,
                estimator=estimator,
                candidate_count=50,
                lookback_observations=252,
            )
            if weights.is_empty():
                output["benchmarks"][name] = {"status": "insufficient_history"}
                continue
            curve, summary = run_monthly_weighted_backtest(
                prices,
                weights,
                config=WeightedBacktestConfig(transaction_cost_bps=15.0),
            )
            target = out_root / name
            target.mkdir(parents=True, exist_ok=True)
            weights.write_parquet(target / "target_weights.parquet", compression="zstd")
            curve.write_parquet(target / "equity_curve.parquet", compression="zstd")
            payload = asdict(summary)
            (target / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            output["benchmarks"][name] = payload
    (out_root / "comparison.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"comparison={out_root / 'comparison.json'}")


if __name__ == "__main__":
    main()
