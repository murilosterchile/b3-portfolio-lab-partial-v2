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
    deflated_sharpe_ratio,
    moving_block_bootstrap_ci,
    probability_backtest_overfitting,
    sharpe_from_periodic_returns,
)


def _monthly_portfolio_returns(curve: pl.DataFrame) -> pl.DataFrame:
    daily = curve.sort("trade_date").with_columns(
        (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1.0).alias("ret")
    ).drop_nulls("ret").with_columns(pl.col("trade_date").dt.truncate("1mo").alias("month"))
    monthly = daily.group_by("month").agg(
        ((1.0 + pl.col("ret")).product() - 1.0).alias("return")
    ).sort("month")
    return monthly


def _benchmark_differential(
    strategy: pl.DataFrame, benchmark: pl.DataFrame, *, block_length: int
) -> dict[str, object]:
    aligned = strategy.rename({"return": "strategy"}).join(
        benchmark.rename({"return": "benchmark"}), on="month", how="inner"
    ).with_columns((pl.col("strategy") - pl.col("benchmark")).alias("excess"))
    values = aligned["excess"].to_numpy()
    cagr_ci = moving_block_bootstrap_ci(
        values, cagr_from_periodic_returns, block_length=block_length, n_bootstrap=2000
    )
    mean_ci = moving_block_bootstrap_ci(
        values, lambda x: float(np.mean(x)), block_length=block_length, n_bootstrap=2000
    )
    return {
        "overlap_months": aligned.height,
        "excess_cagr": {"estimate": cagr_ci[0], "ci_low": cagr_ci[1], "ci_high": cagr_ci[2]},
        "mean_monthly_excess": {"estimate": mean_ci[0], "ci_low": mean_ci[1], "ci_high": mean_ci[2]},
    }


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
    research_curve = pl.read_parquet(curve_path)
    monthly_frame = _monthly_portfolio_returns(research_curve)
    monthly_returns = monthly_frame["return"].to_numpy()
    monthly_excess_returns = (
        research_curve.drop_nulls("excess_return")
        .with_columns(pl.col("trade_date").dt.truncate("1mo").alias("month"))
        .group_by("month")
        .agg(((1.0 + pl.col("excess_return")).product() - 1.0).alias("return"))
        .sort("month")["return"]
        .to_numpy()
        if "excess_return" in research_curve.columns
        else monthly_returns
    )
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
    benchmark_results: dict[str, object] = {}
    for name in ("cdi", "ibov_total_return", "universe_1n"):
        path = data_dir / "gold" / "backtests" / f"development_{name}" / "equity_curve.parquet"
        benchmark_results[name] = (
            _benchmark_differential(
                monthly_frame, _monthly_portfolio_returns(pl.read_parquet(path)), block_length=3
            )
            if path.exists()
            else {"status": "missing; no series was synthesized"}
        )
    strategy_frames: list[pl.DataFrame] = []
    strategy_names: list[str] = []
    for path in sorted((data_dir / "gold" / "backtests").glob("development_*/equity_curve.parquet")):
        name = path.parent.name.removeprefix("development_")
        if name in {"cdi", "ibov_total_return"}:
            continue
        frame = _monthly_portfolio_returns(pl.read_parquet(path)).rename({"return": name})
        strategy_frames.append(frame)
        strategy_names.append(name)
    aligned_strategies = strategy_frames[0] if strategy_frames else pl.DataFrame()
    for frame in strategy_frames[1:]:
        aligned_strategies = aligned_strategies.join(frame, on="month", how="inner")
    pbo = (
        probability_backtest_overfitting(aligned_strategies.select(strategy_names).to_numpy())
        if strategy_names and not aligned_strategies.is_empty()
        else float("nan")
    )
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    tuning_path = model_dir / "tuned_hyperparameters.json"
    tuning = json.loads(tuning_path.read_text(encoding="utf-8")) if tuning_path.exists() else {}
    number_of_trials = max(int(tuning.get("number_of_experiments", tuning.get("n_trials", 1))) + len(strategy_names), 1)
    cdi_path = data_dir / "gold" / "backtests" / "development_cdi" / "equity_curve.parquet"
    if strategy_names and cdi_path.exists():
        cdi = _monthly_portfolio_returns(pl.read_parquet(cdi_path)).rename({"return": "cdi"})
        excess_trials = aligned_strategies.join(cdi, on="month", how="inner")
        trial_sharpes = [
            sharpe_from_periodic_returns(
                (excess_trials[name] - excess_trials["cdi"]).to_numpy()
            )
            for name in strategy_names
        ]
    else:
        trial_sharpes = []
    dsr, dsr_hurdle = deflated_sharpe_ratio(
        monthly_excess_returns,
        number_of_trials=number_of_trials,
        trials_sharpe_std=float(np.std(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0,
    )
    report = {
        "bootstrap_method": "moving block bootstrap",
        "block_length_months": 3,
        "n_bootstrap": 2000,
        "raw_cagr_secondary": {"estimate": cagr_ci[0], "ci_low": cagr_ci[1], "ci_high": cagr_ci[2]},
        "raw_sharpe_secondary": {"estimate": sharpe_ci[0], "ci_low": sharpe_ci[1], "ci_high": sharpe_ci[2]},
        "rank_ic": {"estimate": rank_ic_ci[0], "ci_low": rank_ic_ci[1], "ci_high": rank_ic_ci[2]},
        "top10_spread": {"estimate": spread_ci[0], "ci_low": spread_ci[1], "ci_high": spread_ci[2]},
        "rank_ic_standard_error": ic_se,
        "rank_ic_t_stat": float(np.mean(monthly_ic) / ic_se) if np.isfinite(ic_se) and ic_se > 0 else None,
        "rank_ic_ir": float(np.mean(monthly_ic) / ic_std) if np.isfinite(ic_std) and ic_std > 0 else None,
        "top10_hit_rate": hit_rate,
        "benchmark_differentials": benchmark_results,
        "deflated_sharpe_probability": dsr,
        "deflated_sharpe_hurdle": dsr_hurdle,
        "probability_backtest_overfitting": pbo,
        "number_of_trials": number_of_trials,
        "strategies_in_pbo": strategy_names,
        "block_length_rationale": "Three monthly blocks conservatively exceed the 21-bar target/holding overlap.",
        "note": "IID bootstrap is intentionally not used.",
    }
    out = data_dir / "gold" / "backtests" / "development_statistical_inference.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"statistics={out}")


if __name__ == "__main__":
    main()
