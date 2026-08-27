from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
from portfolio_core.backtest import (
    WeightedBacktestConfig,
    build_monthly_qkp_weights,
    run_monthly_weighted_backtest,
)


def _alpha_prefilter_baseline(signals: pl.DataFrame, candidate_count: int) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date in sorted(set(signals["trade_date"].to_list())):
        cross = signals.filter(pl.col("trade_date") == signal_date).with_columns(
            (
                pl.col("predicted_excess_return")
                - 0.5 * pl.col("prediction_uncertainty")
            ).alias("__qkp_mu")
        ).sort(["__qkp_mu", "rank_log_traded_value_21d"], descending=[True, True])
        issuer = next(
            (name for name in ("CD_CVM", "issuer_id", "issuer_identifier") if name in cross.columns),
            None,
        )
        issuer_expr = (
            pl.coalesce(pl.col(issuer).cast(pl.Utf8), pl.col("ticker").str.slice(0, 4))
            if issuer is not None
            else pl.col("ticker").str.slice(0, 4)
        )
        cross = cross.with_columns(issuer_expr.alias("__issuer")).unique(
            subset="__issuer", keep="first", maintain_order=True
        )
        selected = cross.head(candidate_count).head(10)
        if selected.is_empty():
            continue
        weight = 1.0 / selected.height
        rows.extend(
            {"trade_date": signal_date, "ticker": ticker, "target_weight": weight}
            for ticker in selected["ticker"].to_list()
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _prefilter_recall(signals: pl.DataFrame, candidate_count: int) -> float:
    recalls: list[float] = []
    for cross in signals.partition_by("trade_date", maintain_order=True):
        if "target_excess_return" not in cross.columns or cross.height < 10:
            continue
        candidate = set(
            cross.with_columns(
                (
                    pl.col("predicted_excess_return")
                    - 0.5 * pl.col("prediction_uncertainty")
                ).alias("__qkp_mu")
            ).sort("__qkp_mu", descending=True)
            .head(candidate_count)["ticker"]
            .to_list()
        )
        ex_post = set(
            cross.sort("target_excess_return", descending=True).head(10)["ticker"].to_list()
        )
        recalls.append(len(candidate & ex_post) / len(ex_post))
    return float(np.mean(recalls)) if recalls else float("nan")


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    signals_path = data_dir / "gold" / "backtests" / "development_signals.parquet"
    parts = sorted((data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet"))
    if not signals_path.exists() or not parts:
        raise SystemExit("Run the development backtest and build adjusted prices first")
    signals = pl.read_parquet(signals_path)
    prices = pl.concat([pl.read_parquet(path) for path in parts], how="vertical_relaxed")

    output = data_dir / "gold" / "backtests" / "qkp_ablation"
    output.mkdir(parents=True, exist_ok=True)
    comparisons: dict[str, object] = {}
    for candidate_count in (20, 30, 50, 75):
        baseline = _alpha_prefilter_baseline(signals, candidate_count)
        equal_weights, diagnostics = build_monthly_qkp_weights(
            prices,
            signals,
            allocation="equal_weight",
            candidate_count=candidate_count,
            k_values=(10,),
        )
        hrp_weights, _ = build_monthly_qkp_weights(
            prices,
            signals,
            allocation="hrp",
            candidate_count=candidate_count,
            k_values=(10,),
        )
        if baseline.is_empty() or equal_weights.is_empty() or hrp_weights.is_empty():
            comparisons[str(candidate_count)] = {"status": "insufficient_pit_history"}
            continue
        _, baseline_summary = run_monthly_weighted_backtest(
            prices, baseline, config=WeightedBacktestConfig(transaction_cost_bps=15.0)
        )
        _, qkp_equal = run_monthly_weighted_backtest(
            prices, equal_weights, config=WeightedBacktestConfig(transaction_cost_bps=15.0)
        )
        _, qkp_hrp = run_monthly_weighted_backtest(
            prices, hrp_weights, config=WeightedBacktestConfig(transaction_cost_bps=15.0)
        )
        equal_weights.write_parquet(
            output / f"qkp_equal_weights_candidates_{candidate_count}.parquet", compression="zstd"
        )
        hrp_weights.write_parquet(
            output / f"qkp_hrp_weights_candidates_{candidate_count}.parquet", compression="zstd"
        )
        diagnostics.write_parquet(
            output / f"solver_candidates_{candidate_count}.parquet", compression="zstd"
        )
        comparisons[str(candidate_count)] = {
            "prefilter_top10_equal_weight": asdict(baseline_summary),
            "qkp_same_candidates_equal_weight": asdict(qkp_equal),
            "qkp_same_candidates_hrp": asdict(qkp_hrp),
            "prefilter_recall_top10_ex_post_diagnostic": _prefilter_recall(
                signals, candidate_count
            ),
        }
    report = {
        "sample": "development only",
        "candidate_rule": (
            "PIT investability gate, then calibrated economic alpha first; issuer-aware capacity"
        ),
        "candidate_counts": [20, 30, 50, 75],
        "qkp_parameters": {
            "risk_aversion": 0.7,
            "uncertainty_penalty": 0.5,
            "expected_return_horizon_bars": 21,
            "covariance_horizon_bars": 21,
            "min_positions": 6,
            "max_positions": 10,
        },
        "comparisons": comparisons,
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
