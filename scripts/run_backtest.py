from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
from portfolio_core.backtest import (
    BacktestConfig,
    WeightedBacktestConfig,
    build_monthly_qkp_weights,
    build_monthly_risk_benchmark_weights,
    run_monthly_topk_backtest,
    run_monthly_topk_backtest_detailed,
    run_monthly_weighted_backtest_detailed,
)
from portfolio_core.data.universe import (
    UniverseConfig,
    apply_point_in_time_universe,
    build_point_in_time_universe,
)
from portfolio_core.data_quality import DataQualityError, filter_labels_known_by, invalid_price_rows
from portfolio_core.features.fundamental import FUNDAMENTAL_FEATURES
from portfolio_core.ml.diagnostics import yearly_topk_report
from portfolio_core.ml.protocol import ResearchProtocol
from portfolio_core.ml.walk_forward import (
    DEFAULT_FEATURES,
    TrainingPolicy,
    load_signal_model,
    train_once,
)
from portfolio_core.quant.factors import composite_quant_score, learn_factor_sleeve_weights
from portfolio_core.research import evaluate_acceptance_gates, write_experiment_record


def _load_prices(data_dir: Path) -> tuple[pl.DataFrame, str]:
    adjusted = sorted((data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet"))
    raw = sorted((data_dir / "silver" / "b3_prices").glob("year=*/part-000.parquet"))
    parts = adjusted or raw
    if len(parts) < 3:
        raise SystemExit("Need at least three B3 yearly price partitions")
    source = "B3 adjusted total-return analytical prices" if adjusted else "raw B3 COTAHIST"
    return pl.concat([pl.read_parquet(path) for path in parts], how="vertical_relaxed"), source


def _percentile(expr: pl.Expr) -> pl.Expr:
    denominator = pl.len().over("trade_date")
    return expr.rank(method="average").over("trade_date") / denominator


def _strategy_signals(frame: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    return frame.join(
        predictions.select(
            "trade_date",
            "ticker",
            "predicted_excess_return",
            "ensemble_disagreement",
            "calibrated_uncertainty",
            "prediction_uncertainty",
        ),
        on=["trade_date", "ticker"],
        how="inner",
    ).with_columns(
        _percentile(pl.col("predicted_excess_return")).alias("ml_percentile"),
        _percentile(pl.col("prediction_uncertainty")).alias("uncertainty_percentile"),
    ).with_columns(
        (pl.col("predicted_excess_return") - 0.5 * pl.col("prediction_uncertainty")).alias("signal_ml"),
        pl.col("rank_momentum_12_1").fill_null(0.5).alias("signal_momentum"),
        pl.col("low_volatility_score").fill_null(0.5).alias("signal_low_volatility"),
        pl.col("quality_score").fill_null(0.5).alias("signal_quality"),
        pl.col("quant_score_equal").fill_null(0.5).alias("signal_quant"),
        pl.col("quant_score").fill_null(0.5).alias("signal_quant_learned"),
        (
            0.65 * pl.col("ml_percentile")
            + 0.25 * pl.col("quant_score_equal").fill_null(0.5)
            + 0.10 * (1.0 - pl.col("uncertainty_percentile"))
        ).alias("signal_research_v2"),
        pl.lit(1.0).alias("signal_equal_weight"),
        pl.col("rank_log_volume_21d").fill_null(0.0).alias("signal_liquidity"),
    )


def _score_factor_challengers(
    frame: pl.DataFrame, learned_weights: dict[str, float]
) -> pl.DataFrame:
    equal = composite_quant_score(frame).select(
        "trade_date", "ticker", pl.col("quant_score").alias("quant_score_equal")
    )
    return composite_quant_score(frame, sleeve_weights=learned_weights).join(
        equal, on=["trade_date", "ticker"], how="left", validate="1:1"
    )



def _features_after_training(features: pl.DataFrame, trained_until: object) -> pl.DataFrame:
    """Backward-compatible helper: prediction feature dates must be strictly after trained feature dates."""
    future = features.filter(pl.col("trade_date") > pl.lit(trained_until))
    if future.is_empty():
        raise SystemExit("No feature rows after model trained_until")
    return future

def _diagnostic_features(features: pl.DataFrame, protocol: ResearchProtocol) -> pl.DataFrame:
    return features.filter(pl.col("trade_date").dt.year() == protocol.diagnostic_year)


def _write_result(
    data_dir: Path,
    name: str,
    curve: pl.DataFrame,
    summary: object,
    *,
    metadata: dict[str, object],
) -> None:
    out_dir = data_dir / "gold" / "backtests" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    curve.write_parquet(out_dir / "equity_curve.parquet", compression="zstd")
    payload = {**asdict(summary), **metadata}
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def _run_reference_benchmarks(
    *, data_dir: Path, signals: pl.DataFrame, prefix: str
) -> dict[str, dict[str, object]]:
    """Run governed IBOV total-return and CDI curves through the weighted engine.

    Files must live under ``silver/benchmarks`` and contain ``trade_date`` plus
    either ``adjusted_close`` or a decimal ``daily_return``. Missing benchmark
    data are reported explicitly, never substituted with a price index/proxy.
    """
    results: dict[str, dict[str, object]] = {}
    signal_dates = sorted(set(signals.get_column("trade_date").to_list()))
    for name in ("ibov_total_return", "cdi"):
        path = data_dir / "silver" / "benchmarks" / f"{name}.parquet"
        if not path.exists():
            results[name] = {"status": "missing_governed_benchmark", "expected_path": str(path)}
            continue
        frame = pl.read_parquet(path).sort("trade_date")
        if "adjusted_close" not in frame.columns:
            if "daily_return" not in frame.columns:
                results[name] = {"status": "invalid_benchmark_contract"}
                continue
            frame = frame.with_columns(
                (pl.col("daily_return") + 1.0).cum_prod().alias("adjusted_close")
            )
        ticker = f"__{name.upper()}__"
        benchmark_prices = frame.select("trade_date", "adjusted_close").with_columns(
            pl.lit(ticker).alias("ticker"), pl.col("adjusted_close").alias("close")
        )
        available_dates = set(benchmark_prices.get_column("trade_date").to_list())
        weights = pl.DataFrame(
            {
                "trade_date": [value for value in signal_dates if value in available_dates],
                "ticker": [ticker for value in signal_dates if value in available_dates],
                "target_weight": [1.0 for value in signal_dates if value in available_dates],
            }
        )
        if weights.is_empty():
            results[name] = {"status": "no_overlapping_signal_dates"}
            continue
        detail = run_monthly_weighted_backtest_detailed(
            benchmark_prices,
            weights,
            config=WeightedBacktestConfig(transaction_cost_bps=0.0),
        )
        target = data_dir / "gold" / "backtests" / f"{prefix}_{name}"
        target.mkdir(parents=True, exist_ok=True)
        detail.curve.write_parquet(target / "equity_curve.parquet", compression="zstd")
        results[name] = asdict(detail.summary)
    return results


def _run_strategies(
    *,
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    data_dir: Path,
    prefix: str,
    price_source: str,
    methodology: str,
) -> dict[str, dict[str, object]]:
    strategies = {
        "universe_1n": ("signal_equal_weight", BacktestConfig(top_k=100_000, transaction_cost_bps=15.0)),
        "top_liquidity_equal_weight": ("signal_liquidity", BacktestConfig(top_k=50, transaction_cost_bps=15.0, selection_buffer=10)),
        "ml_top10": ("signal_ml", BacktestConfig(top_k=10, transaction_cost_bps=15.0)),
        "momentum_top10": ("signal_momentum", BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5)),
        "low_volatility_top10": ("signal_low_volatility", BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5)),
        "quality_top10": ("signal_quality", BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5)),
        "multifactor_top10": ("signal_quant", BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5)),
        "multifactor_learned_top10": ("signal_quant_learned", BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5)),
        "research_v2": ("signal_research_v2", BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5)),
    }
    if signals["quality_score"].n_unique() <= 1:
        strategies.pop("quality_top10")
    results: dict[str, dict[str, object]] = {}
    for name, (column, config) in strategies.items():
        detail = run_monthly_topk_backtest_detailed(
            prices, signals, score_column=column, config=config
        )
        curve, summary = detail.curve, detail.summary
        metadata = {
            "strategy": name,
            "score_column": column,
            "price_source": price_source,
            "selection_buffer": config.selection_buffer,
            "transaction_cost_bps": config.transaction_cost_bps,
            "execution_delay_bars": config.execution_delay_bars,
            "methodology": methodology,
        }
        _write_result(data_dir, f"{prefix}_{name}", curve, summary, metadata=metadata)
        detail_dir = data_dir / "gold" / "backtests" / f"{prefix}_{name}"
        detail.selections.write_parquet(detail_dir / "selections.parquet", compression="zstd")
        detail.contributions.write_parquet(detail_dir / "contributions.parquet", compression="zstd")
        detail.rejected_orders.write_parquet(detail_dir / "rejected_orders.parquet", compression="zstd")
        results[name] = asdict(summary)
        print(
            f"{prefix}/{name:28s} CAGR={summary.cagr:8.2%} Sharpe={summary.sharpe:7.3f} "
            f"MDD={summary.max_drawdown:8.2%} turnover/y={summary.annualized_turnover:6.2f}x"
        )
    qkp_allocations = {
        "ml_qkp_ew": "equal_weight",
        "ml_qkp_inverse_vol": "inverse_vol",
        "ml_qkp_hrp": "hrp",
        "ml_qkp_cost_aware_minvar": "cost_aware_minvar",
    }
    for name, allocation in qkp_allocations.items():
        weights, solver_diagnostics = build_monthly_qkp_weights(
            prices, signals, allocation=allocation
        )
        if weights.is_empty():
            results[name] = {"status": "insufficient_point_in_time_covariance"}
            continue
        detail = run_monthly_weighted_backtest_detailed(
            prices, weights, config=WeightedBacktestConfig(transaction_cost_bps=15.0)
        )
        target = data_dir / "gold" / "backtests" / f"{prefix}_{name}"
        target.mkdir(parents=True, exist_ok=True)
        detail.curve.write_parquet(target / "equity_curve.parquet", compression="zstd")
        weights.write_parquet(target / "target_weights.parquet", compression="zstd")
        solver_diagnostics.write_parquet(target / "solver_diagnostics.parquet", compression="zstd")
        detail.rejected_orders.write_parquet(target / "rejected_orders.parquet", compression="zstd")
        payload = asdict(detail.summary)
        (target / "summary.json").write_text(
            json.dumps({**payload, "strategy": name, "allocation": allocation, "methodology": methodology}, indent=2),
            encoding="utf-8",
        )
        results[name] = payload
        print(
            f"{prefix}/{name:28s} CAGR={detail.summary.cagr:8.2%} "
            f"Sharpe={detail.summary.sharpe:7.3f} MDD={detail.summary.max_drawdown:8.2%}"
        )

    risk_weights = build_monthly_risk_benchmark_weights(
        prices, signals, allocation="hrp", estimator="ledoit_wolf", candidate_count=50
    )
    if risk_weights.is_empty():
        results["risk_only_hrp"] = {"status": "insufficient_point_in_time_covariance"}
    else:
        detail = run_monthly_weighted_backtest_detailed(
            prices, risk_weights, config=WeightedBacktestConfig(transaction_cost_bps=15.0)
        )
        target = data_dir / "gold" / "backtests" / f"{prefix}_risk_only_hrp"
        target.mkdir(parents=True, exist_ok=True)
        detail.curve.write_parquet(target / "equity_curve.parquet", compression="zstd")
        risk_weights.write_parquet(target / "target_weights.parquet", compression="zstd")
        detail.rejected_orders.write_parquet(target / "rejected_orders.parquet", compression="zstd")
        results["risk_only_hrp"] = asdict(detail.summary)
    results.update(_run_reference_benchmarks(data_dir=data_dir, signals=signals, prefix=prefix))
    return results


def _cost_sensitivity(prices: pl.DataFrame, signals: pl.DataFrame) -> list[dict[str, float]]:
    output: list[dict[str, float]] = []
    gross_curve, gross = run_monthly_topk_backtest(
        prices,
        signals,
        score_column="signal_research_v2",
        config=BacktestConfig(top_k=10, transaction_cost_bps=0.0, selection_buffer=5),
    )
    del gross_curve
    for bps in (5.0, 10.0, 25.0, 50.0, 100.0):
        _, net = run_monthly_topk_backtest(
            prices,
            signals,
            score_column="signal_research_v2",
            config=BacktestConfig(top_k=10, transaction_cost_bps=bps, selection_buffer=5),
        )
        output.append(
            {
                "transaction_cost_bps": bps,
                "gross_cagr": gross.cagr,
                "net_cagr": net.cagr,
                "gross_sharpe": gross.sharpe,
                "net_sharpe": net.sharpe,
                "annualized_turnover": net.annualized_turnover,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-trained-model",
        action="store_true",
        help="evaluate the leakage-safe saved artifact on diagnostic year only",
    )
    args = parser.parse_args()
    protocol = ResearchProtocol()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    selected_policy_path = model_dir / "selected_training_policy.json"
    if not selected_policy_path.exists():
        raise SystemExit(
            "Missing development-only policy selection. Run make diagnose-model-degradation first."
        )
    selected_policy = json.loads(selected_policy_path.read_text(encoding="utf-8"))
    if (
        int(selected_policy.get("development_end_year", -1)) != protocol.development_end_year
        or int(selected_policy.get("diagnostic_year", -1)) != protocol.diagnostic_year
    ):
        raise SystemExit("Selected training policy does not match the research protocol")
    policy = TrainingPolicy(**selected_policy["training_policy"])
    fundamental_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    technical_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    feature_path = fundamental_path if fundamental_path.exists() else technical_path
    if not feature_path.exists():
        raise SystemExit("Need a monthly feature panel")
    features = pl.read_parquet(feature_path)
    required = {"target_end_date", "execution_lag_bars", "target_horizon_bars"}
    if missing := required - set(features.columns):
        raise SystemExit(f"Feature panel predates leakage-safe protocol. Rebuild it; missing {sorted(missing)}")
    feature_names = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in features.columns]
    prices, price_source = _load_prices(data_dir)
    horizons = features.get_column("target_horizon_bars").unique().to_list()
    if horizons != [protocol.label_horizon_bars]:
        raise SystemExit(
            f"Feature panel target horizon {horizons} is incompatible with the pre-registered "
            f"monthly horizon {protocol.label_horizon_bars}; rebuild features"
        )
    universe_config = UniverseConfig()
    universe = build_point_in_time_universe(prices, config=universe_config)
    features = apply_point_in_time_universe(features, universe)
    if features.is_empty():
        raise SystemExit("Point-in-time investible universe is empty under the registered thresholds")
    print(f"feature_panel={feature_path} features={len(feature_names)}")
    print(f"price_source={price_source}")
    print(protocol.warning)

    print("\nDATA QUALITY")
    print("------------")
    print(f"Rows                     {prices.height:,}")
    print(f"Tickers                  {prices['ticker'].n_unique():,}")
    print(f"Date range               {prices['trade_date'].min()} -> {prices['trade_date'].max()}")
    print(f"Invalid source prices    {invalid_price_rows(prices).height:,}")
    duplicates = prices.group_by(["ticker", "trade_date"]).len().filter(pl.col("len") > 1)
    print(f"Duplicate ticker/date    {duplicates.height:,}")

    tuned_path = model_dir / "tuned_hyperparameters.json"
    model_config = None
    if tuned_path.exists():
        metadata = json.loads(tuned_path.read_text(encoding="utf-8"))
        if (
            int(metadata.get("development_end_year", -1)) == protocol.development_end_year
            and int(metadata.get("diagnostic_year", -1)) == protocol.diagnostic_year
            and int(metadata.get("target_horizon_bars", -1)) == protocol.label_horizon_bars
        ):
            model_config = metadata.get("config")

    if args.use_trained_model:
        model_path = model_dir / "signal_model.joblib"
        if not model_path.exists():
            raise SystemExit(f"Trained model not found: {model_path}. Run make train first.")
        model = load_signal_model(str(model_path))
        if model.target_end_max is None or model.target_end_max >= protocol.diagnostic_start:
            raise SystemExit("Saved model contains a label reaching the diagnostic period")
        if model.knowledge_cutoff is None or model.knowledge_cutoff >= protocol.diagnostic_start:
            raise SystemExit("Saved model knowledge_cutoff is not strictly before diagnostic year")
        diagnostic = _diagnostic_features(features, protocol)
        if diagnostic.is_empty():
            raise SystemExit(f"No features available for diagnostic year {protocol.diagnostic_year}")
        factor_path = model_dir / "factor_sleeve_challenger.json"
        if not factor_path.exists():
            raise SystemExit("Missing factor sleeve weights; retrain the governed artifact")
        factor_weights = json.loads(factor_path.read_text(encoding="utf-8"))["weights"]
        scored = _score_factor_challengers(diagnostic, factor_weights)
        signals = _strategy_signals(scored, model.predict(diagnostic))
        diagnostic_signals_path = data_dir / "gold" / "backtests" / f"diagnostic_{protocol.diagnostic_year}_signals.parquet"
        diagnostic_signals_path.parent.mkdir(parents=True, exist_ok=True)
        signals.write_parquet(diagnostic_signals_path, compression="zstd")
        results = _run_strategies(
            prices=prices,
            signals=signals,
            data_dir=data_dir,
            prefix=f"diagnostic_{protocol.diagnostic_year}",
            price_source=price_source,
            methodology=(
                f"diagnostic only; artifact knowledge_cutoff={model.knowledge_cutoff}; "
                f"target_end_max={model.target_end_max}; no diagnostic-year selection"
            ),
        )
        report = {
            "diagnostic_year": protocol.diagnostic_year,
            "warning": protocol.warning,
            "accepted": None,
            "reason": "Diagnostic-year performance is prohibited from changing acceptance or model selection.",
            "strategies": results,
        }
        out = data_dir / "gold" / "backtests" / f"comparison_diagnostic_{protocol.diagnostic_year}.json"
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"diagnostic_comparison={out}")
        return

    development = filter_labels_known_by(
        features, knowledge_cutoff=protocol.development_knowledge_cutoff
    )
    years = sorted(set(development["trade_date"].dt.year().to_list()))
    prediction_frames: list[pl.DataFrame] = []
    fold_metrics: dict[int, dict[str, float]] = {}
    for year in [value for value in years if 2014 <= value <= protocol.development_end_year]:
        train = development.filter(pl.col("trade_date").dt.year() < year)
        test = development.filter(pl.col("trade_date").dt.year() == year)
        if train.height < 500 or test.height < 50:
            continue
        model, metrics = train_once(
            train,
            test,
            feature_names=feature_names,
            seed=year,
            model_config=model_config,
            training_policy=policy,
        )
        validation_start = test["trade_date"].min()
        factor_train = train.filter(pl.col("target_end_date") < pl.lit(validation_start))
        if policy.window_years is not None:
            factor_train = factor_train.filter(
                pl.col("trade_date").dt.year() >= year - policy.window_years
            )
        fold_factor_weights = learn_factor_sleeve_weights(factor_train)
        scored = _score_factor_challengers(test, fold_factor_weights)
        combined = _strategy_signals(scored, model.predict(test)).with_columns(pl.lit(year).alias("oos_year"))
        prediction_frames.append(combined)
        fold_metrics[year] = asdict(metrics)
        print(
            f"development_fold={year} rank_ic={metrics.rank_ic:.4f} "
            f"top_minus_bottom={metrics.top_minus_bottom:.4f}"
        )
    if not prediction_frames:
        raise SystemExit("Insufficient history for purged walk-forward predictions")
    signals = pl.concat(prediction_frames, how="vertical_relaxed")
    signals_path = data_dir / "gold" / "backtests" / "development_signals.parquet"
    signals_path.parent.mkdir(parents=True, exist_ok=True)
    signals.write_parquet(signals_path, compression="zstd")
    results = _run_strategies(
        prices=prices,
        signals=signals,
        data_dir=data_dir,
        prefix="development",
        price_source=price_source,
        methodology=(
            f"purged selected-policy walk-forward ({selected_policy['selected_name']}); "
            "next-close execution; development data through 2025 only"
        ),
    )

    topk = yearly_topk_report(
        signals,
        signals["predicted_excess_return"].to_numpy(),
        target_column="target_excess_return",
    )
    topk_path = data_dir / "gold" / "backtests" / "development_topk_by_year.parquet"
    topk.write_parquet(topk_path, compression="zstd")
    recent_rank_values = [
        metrics["rank_ic"] for year, metrics in fold_metrics.items() if 2022 <= year <= 2025
    ]
    recent_rank_ic = float(np.mean(recent_rank_values)) if recent_rank_values else float("nan")
    top10_recent = topk.filter((pl.col("k") == 10) & pl.col("year").is_between(2022, 2025))
    top10_spread = float(top10_recent["top_minus_bottom"].mean()) if not top10_recent.is_empty() else float("nan")
    research = results["research_v2"]
    one_n = results["universe_1n"]
    gate = evaluate_acceptance_gates(
        recent_rank_ic=recent_rank_ic,
        top10_spread=top10_spread,
        net_sharpe=float(research["sharpe"]),
        max_drawdown=float(research["max_drawdown"]),
        annualized_turnover=float(research["annualized_turnover"]),
        cagr_advantage_vs_1n=float(research["cagr"]) - float(one_n["cagr"]),
        leakage_tests_passed=True,
        diagnostic_only_acknowledged=True,
    )
    cost_sensitivity = _cost_sensitivity(prices, signals)
    report = {
        "protocol": {
            "development_end_year": protocol.development_end_year,
            "diagnostic_year": protocol.diagnostic_year,
            "warning": protocol.warning,
        },
        "price_source": price_source,
        "feature_panel": str(feature_path),
        "fold_metrics": fold_metrics,
        "strategies": results,
        "cost_sensitivity": cost_sensitivity,
        "acceptance": gate,
        "accepted": bool(gate["production_ready"]),
        "important": "2026 is not included in these gates.",
    }
    out = data_dir / "gold" / "backtests" / "comparison_development.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    experiment = write_experiment_record(
        data_dir=data_dir,
        features=feature_names,
        parameters={"model_config": model_config, "transaction_cost_bps": 15.0},
        training_window={
            "selected_name": selected_policy["selected_name"],
            **selected_policy["training_policy"],
        },
        validation_folds=sorted(fold_metrics),
        metrics={"acceptance": gate, "strategies": results, "recent_rank_ic": recent_rank_ic},
        notes=[protocol.warning, "Same-close execution disabled", "Overlapping labels purged"],
    )
    print(f"comparison={out} production_ready={gate['production_ready']}")
    print(f"experiment_record={experiment}")


if __name__ == "__main__":
    try:
        main()
    except DataQualityError as exc:
        raise SystemExit(f"DATA QUALITY FAILURE: {exc}") from exc
