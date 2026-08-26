from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
from portfolio_core.backtest import (
    WeightedBacktestConfig,
    point_in_time_covariance,
    run_monthly_weighted_backtest,
)
from portfolio_core.optimizer import Candidate, build_portfolio_qkp, solve_portfolio
from portfolio_core.quant import hierarchical_risk_parity
from portfolio_core.quant.risk import covariance_to_correlation, ledoit_wolf_covariance


def _pit_covariance(
    prices: pl.DataFrame, tickers: list[str], signal_date: object, lookback: int = 252
) -> tuple[list[str], np.ndarray] | None:
    result = point_in_time_covariance(
        prices, tickers, signal_date, lookback_observations=lookback
    )
    return result if result is not None and len(result[0]) >= 6 else None


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    signals_path = data_dir / "gold" / "backtests" / "development_signals.parquet"
    parts = sorted((data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet"))
    if not signals_path.exists() or not parts:
        raise SystemExit("Run the development backtest and build adjusted prices first")
    signals = pl.read_parquet(signals_path)
    prices = pl.concat([pl.read_parquet(path) for path in parts], how="vertical_relaxed")

    equal_rows: list[dict[str, object]] = []
    allocated_rows: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    for signal_date in sorted(set(signals["trade_date"].to_list())):
        cross = signals.filter(pl.col("trade_date") == signal_date).with_columns(
            (
                0.80 * pl.col("ml_percentile")
                + 0.15 * pl.col("quant_score").fill_null(0.5)
                + 0.05 * pl.col("rank_log_volume_21d").fill_null(0.5)
            ).alias("candidate_score")
        ).sort("candidate_score", descending=True).head(18)
        requested = cross["ticker"].to_list()
        covariance_result = _pit_covariance(prices, requested, signal_date)
        if covariance_result is None:
            continue
        available, covariance = covariance_result
        cross = cross.filter(pl.col("ticker").is_in(available))
        # Preserve covariance order exactly.
        order = {ticker: i for i, ticker in enumerate(available)}
        cross = cross.with_columns(pl.col("ticker").replace_strict(order).alias("__order")).sort("__order")
        available = cross["ticker"].to_list()
        if len(available) < 6:
            continue
        idx = [order[ticker] for ticker in available]
        covariance = covariance[np.ix_(idx, idx)]
        correlation = covariance_to_correlation(covariance)
        vol = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        candidates = []
        for row, volatility in zip(cross.iter_rows(named=True), vol, strict=True):
            candidates.append(
                Candidate(
                    ticker=str(row["ticker"]),
                    price=float(row["close"]),
                    predicted_excess_return=float(row["predicted_excess_return"]),
                    uncertainty=float(row["prediction_uncertainty"]),
                    sector=str(row.get("sector") or "Unknown"),
                    quant_score=100.0 * float(row.get("quant_score") or 0.5),
                    liquidity_score=100.0 * float(row.get("rank_log_volume_21d") or 0.5),
                    volatility_annual=float(volatility),
                )
            )
        instance = build_portfolio_qkp(
            candidates,
            correlation=correlation,
            budget=100_000.0,
            min_positions=6,
            max_positions=10,
            fixed_k=10,
            risk_aversion=0.7,
            uncertainty_penalty=0.5,
            min_position_fraction=1.0 / 18.0,
            sector_max_count={"Financials": 3, "Energy": 2, "Materials": 2, "Utilities": 3, "Consumer": 3},
        )
        result = solve_portfolio(instance, backend="auto")
        if not result.selected_indices:
            continue
        selected = list(result.selected_indices)
        selected_cov = covariance[np.ix_(selected, selected)]
        allocation = hierarchical_risk_parity(selected_cov)
        equal_weight = 1.0 / len(selected)
        for position, weight in zip(selected, allocation, strict=True):
            ticker = candidates[position].ticker
            equal_rows.append({"trade_date": signal_date, "ticker": ticker, "target_weight": equal_weight})
            allocated_rows.append({"trade_date": signal_date, "ticker": ticker, "target_weight": float(weight)})
        monthly.append(
            {
                "trade_date": signal_date,
                "solver": result.solver,
                "exact": result.exact,
                "objective": result.objective,
                "candidate_count": len(candidates),
                "selected_count": len(selected),
            }
        )

    if not equal_rows:
        raise SystemExit("No QKP month had sufficient PIT covariance history")
    equal_weights = pl.DataFrame(equal_rows)
    allocated_weights = pl.DataFrame(allocated_rows)
    _, qkp_equal = run_monthly_weighted_backtest(
        prices, equal_weights, config=WeightedBacktestConfig(transaction_cost_bps=15.0)
    )
    _, qkp_allocated = run_monthly_weighted_backtest(
        prices, allocated_weights, config=WeightedBacktestConfig(transaction_cost_bps=15.0)
    )
    output = data_dir / "gold" / "backtests" / "qkp_ablation"
    output.mkdir(parents=True, exist_ok=True)
    equal_weights.write_parquet(output / "qkp_equal_weights.parquet", compression="zstd")
    allocated_weights.write_parquet(output / "qkp_hrp_weights.parquet", compression="zstd")
    pl.DataFrame(monthly).write_parquet(output / "monthly_solver_results.parquet", compression="zstd")
    report = {
        "candidate_rule": "current API-style 80% ML percentile + 15% multifactor + 5% liquidity; top 18",
        "qkp_parameters": {
            "risk_aversion": 0.7,
            "uncertainty_penalty": 0.5,
            "min_positions": 6,
            "max_positions": 10,
        },
        "qkp_equal_weight": asdict(qkp_equal),
        "qkp_plus_hrp_allocation": asdict(qkp_allocated),
        "interpretation": (
            "Compare these with development_ml_top10/research_v2 to separate signal selection, "
            "QKP selection and allocation effects. No parameter is chosen with 2026."
        ),
    }
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"qkp_ablation={output}")


if __name__ == "__main__":
    main()
