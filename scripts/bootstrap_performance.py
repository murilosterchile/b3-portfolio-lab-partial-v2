from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from portfolio_core.ml.evaluation import top_k_metrics
from portfolio_core.research.statistics import (
    cagr_from_periodic_returns,
    moving_block_bootstrap_ci,
    sharpe_from_periodic_returns,
)


def _monthly_portfolio_returns(curve: pl.DataFrame) -> np.ndarray:
    daily = curve.sort("trade_date").with_columns(
        (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1.0).alias("ret")
    ).drop_nulls("ret").with_columns(pl.col("trade_date").dt.truncate("1mo").alias("month"))
    monthly = daily.group_by("month").agg(
        ((1.0 + pl.col("ret")).product() - 1.0).alias("return")
    ).sort("month")
    return monthly["return"].to_numpy()


def _monthly_signal_statistics(signals: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    ics: list[float] = []
    spreads: list[float] = []
    top_hits: list[float] = []
    for group in signals.partition_by("trade_date", maintain_order=True):
        truth = group["target_excess_return"].to_numpy()
        pred = group["predicted_excess_return"].to_numpy()
        mask = np.isfinite(truth) & np.isfinite(pred)
        if mask.sum() < 10:
            continue
        stat = spearmanr(truth[mask], pred[mask]).statistic
        if stat is not None and np.isfinite(stat):
            ics.append(float(stat))
        top = top_k_metrics(truth[mask], pred[mask], k=10)
        if np.isfinite(top.top_minus_bottom):
            spreads.append(top.top_minus_bottom)
            top_hits.append(top.hit_rate)
    return np.asarray(ics), np.asarray(spreads), float(np.mean(top_hits)) if top_hits else float("nan")


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    curve_path = data_dir / "gold" / "backtests" / "development_research_v2" / "equity_curve.parquet"
    signals_path = data_dir / "gold" / "backtests" / "development_signals.parquet"
    if not curve_path.exists() or not signals_path.exists():
        raise SystemExit("Run the development backtest first")
    monthly_returns = _monthly_portfolio_returns(pl.read_parquet(curve_path))
    monthly_ic, monthly_spread, hit_rate = _monthly_signal_statistics(pl.read_parquet(signals_path))
    cagr_ci = moving_block_bootstrap_ci(
        monthly_returns, cagr_from_periodic_returns, block_length=3, n_bootstrap=2000
    )
    sharpe_ci = moving_block_bootstrap_ci(
        monthly_returns, sharpe_from_periodic_returns, block_length=3, n_bootstrap=2000
    )
    rank_ic_ci = moving_block_bootstrap_ci(
        monthly_ic, lambda x: float(np.mean(x)), block_length=3, n_bootstrap=2000
    )
    spread_ci = moving_block_bootstrap_ci(
        monthly_spread, lambda x: float(np.mean(x)), block_length=3, n_bootstrap=2000
    )
    ic_std = float(np.std(monthly_ic, ddof=1)) if len(monthly_ic) > 1 else float("nan")
    ic_se = ic_std / np.sqrt(len(monthly_ic)) if len(monthly_ic) else float("nan")
    report = {
        "bootstrap_method": "moving block bootstrap",
        "block_length_months": 3,
        "n_bootstrap": 2000,
        "cagr": {"estimate": cagr_ci[0], "ci_low": cagr_ci[1], "ci_high": cagr_ci[2]},
        "sharpe": {"estimate": sharpe_ci[0], "ci_low": sharpe_ci[1], "ci_high": sharpe_ci[2]},
        "rank_ic": {"estimate": rank_ic_ci[0], "ci_low": rank_ic_ci[1], "ci_high": rank_ic_ci[2]},
        "top10_spread": {"estimate": spread_ci[0], "ci_low": spread_ci[1], "ci_high": spread_ci[2]},
        "rank_ic_standard_error": ic_se,
        "rank_ic_t_stat": float(np.mean(monthly_ic) / ic_se) if np.isfinite(ic_se) and ic_se > 0 else None,
        "rank_ic_ir": float(np.mean(monthly_ic) / ic_std) if np.isfinite(ic_std) and ic_std > 0 else None,
        "top10_hit_rate": hit_rate,
        "note": "IID bootstrap is intentionally not used.",
    }
    out = data_dir / "gold" / "backtests" / "development_statistical_inference.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"statistics={out}")


if __name__ == "__main__":
    main()
