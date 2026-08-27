from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from portfolio_core.backtest import (
    BacktestConfig,
    ExecutionCostModel,
    WeightedBacktestConfig,
    build_monthly_qkp_weights,
    build_monthly_risk_benchmark_weights,
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
from portfolio_core.ml.evaluation import top_k_metrics
from portfolio_core.ml.protocol import ResearchProtocol
from portfolio_core.ml.walk_forward import (
    DEFAULT_FEATURES,
    TrainingPolicy,
    load_signal_model,
    train_once,
)
from portfolio_core.quant.factors import composite_quant_score, learn_factor_sleeve_weights
from portfolio_core.research import evaluate_acceptance_gates, write_experiment_record
from portfolio_core.research.gates import evaluate_benchmark_fairness
from portfolio_core.research.statistics import (
    cagr_from_periodic_returns,
    deflated_sharpe_ratio,
    moving_block_bootstrap_ci,
    sharpe_from_periodic_returns,
)
from portfolio_core.research.verification import verify_test_manifest


INITIAL_CAPITAL = 100_000.0
EXECUTION_DELAY_BARS = 1
PARTICIPATION_CAP = 0.02
FILL_POLICY = "next_observed_close_with_participation_cap_and_partial_fills"
REGISTERED_COST_MODEL = ExecutionCostModel(
    mode="liquidity",
    fee_bps=3.0,
    participation_cap=PARTICIPATION_CAP,
    impact_bps=10.0,
    use_eod_quoted_spread=True,
)
VOLATILITY_SCALED_IMPACT_STRESS = ExecutionCostModel(
    mode="liquidity",
    fee_bps=3.0,
    participation_cap=PARTICIPATION_CAP,
    impact_model="volatility_scaled",
    volatility_impact_y=0.5,
    use_eod_quoted_spread=True,
)
COST_SCENARIOS = {
    "registered_liquidity": REGISTERED_COST_MODEL,
    "stress_volatility_scaled_impact_y_0_5": VOLATILITY_SCALED_IMPACT_STRESS,
}
STRATEGY_COST_TABLE_SEMANTICS = {
    "gross": "gross total return before execution costs on the realized fill path",
    "fees": "cumulative one-way fees in BRL divided by initial capital",
    "spread": "cumulative one-way half-spread cost in BRL divided by initial capital",
    "impact": "cumulative one-way market impact in BRL divided by initial capital",
    "net": "net total return after fees, spread, and impact",
    "turnover": "cumulative one-way turnover (half-L1 including cash)",
    "fill_ratio": "filled notional / requested notional",
    "rejected_notional": "requested minus filled notional in BRL",
}


def _load_prices(data_dir: Path) -> tuple[pl.DataFrame, str]:
    adjusted = sorted((data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet"))
    raw = sorted((data_dir / "silver" / "b3_prices").glob("year=*/part-000.parquet"))
    parts = adjusted or raw
    if len(parts) < 3:
        raise SystemExit("Need at least three B3 yearly price partitions")
    source = "B3 adjusted total-return analytical prices" if adjusted else "raw B3 COTAHIST"
    return pl.concat([pl.read_parquet(path) for path in parts], how="vertical_relaxed"), source


def _load_cdi_returns(data_dir: Path) -> tuple[pl.DataFrame | None, dict[str, object]]:
    governed = data_dir / "silver" / "benchmarks" / "cdi.parquet"
    bcb = data_dir / "silver" / "bcb" / "series=cdi_daily" / "part-000.parquet"
    if governed.exists():
        frame = pl.read_parquet(governed).sort("trade_date")
        if "daily_return" not in frame.columns:
            return None, {"status": "invalid_contract", "path": str(governed)}
        source = governed
    elif bcb.exists():
        frame = pl.read_parquet(bcb).select(
            pl.col("date").alias("trade_date"),
            (pl.col("value") / 100.0).alias("daily_return"),
        ).sort("trade_date")
        source = bcb
    else:
        return None, {
            "status": "missing",
            "expected_paths": [str(governed), str(bcb)],
            "note": "No proxy was synthesized; run ingest-macro or provide a governed CDI series.",
        }
    return frame, {
        "status": "available",
        "path": str(source),
        "start": frame["trade_date"].min(),
        "end": frame["trade_date"].max(),
        "observations": frame.height,
        "unit": "decimal return per business day",
    }


def _monthly_returns(curve: pl.DataFrame, column: str = "net_return") -> pl.DataFrame:
    if column not in curve.columns:
        curve = curve.sort("trade_date").with_columns(
            (pl.col("portfolio_value") / pl.col("portfolio_value").shift(1) - 1.0).alias(column)
        )
    return curve.drop_nulls(column).with_columns(
        pl.col("trade_date").dt.truncate("1mo").alias("month")
    ).group_by("month").agg(((1.0 + pl.col(column)).product() - 1.0).alias("return")).sort("month")


def _differential_ci(
    data_dir: Path, prefix: str, benchmark: str, *, block_length: int = 3
) -> tuple[float, float, float]:
    root = data_dir / "gold" / "backtests"
    strategy_path = root / f"{prefix}_research_v2" / "equity_curve.parquet"
    benchmark_path = root / f"{prefix}_{benchmark}" / "equity_curve.parquet"
    if not strategy_path.exists() or not benchmark_path.exists():
        return float("nan"), float("nan"), float("nan")
    strategy = _monthly_returns(pl.read_parquet(strategy_path)).rename({"return": "strategy"})
    reference = _monthly_returns(pl.read_parquet(benchmark_path)).rename({"return": "benchmark"})
    aligned = strategy.join(reference, on="month", how="inner").with_columns(
        ((1.0 + pl.col("strategy")) / (1.0 + pl.col("benchmark")) - 1.0).alias(
            "relative_return"
        )
    )
    return moving_block_bootstrap_ci(
        aligned["relative_return"].to_numpy(),
        cagr_from_periodic_returns,
        block_length=block_length,
        n_bootstrap=2000,
    )


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
        pl.col("rank_log_traded_value_21d").fill_null(0.0).alias("signal_liquidity"),
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
    *,
    data_dir: Path,
    signals: pl.DataFrame,
    prefix: str,
    risk_free_returns: pl.DataFrame | None,
) -> dict[str, dict[str, object]]:
    """Run governed IBOV total-return and CDI curves through the weighted engine.

    Files must live under ``silver/benchmarks`` and contain ``trade_date`` plus
    either ``adjusted_close`` or a decimal ``daily_return``. Missing benchmark
    data are reported explicitly, never substituted with a price index/proxy.
    """
    results: dict[str, dict[str, object]] = {}
    signal_dates = sorted(set(signals.get_column("trade_date").to_list()))
    evaluation_end = (
        signals["target_end_date"].max()
        if "target_end_date" in signals.columns
        else max(signal_dates)
    )
    for name in ("ibov_total_return", "cdi"):
        path = data_dir / "silver" / "benchmarks" / f"{name}.parquet"
        if name == "cdi" and risk_free_returns is not None:
            frame = risk_free_returns
        elif not path.exists():
            results[name] = {"status": "missing_governed_benchmark", "expected_path": str(path)}
            continue
        else:
            frame = pl.read_parquet(path).sort("trade_date")
        frame = frame.filter(pl.col("trade_date") <= pl.lit(evaluation_end))
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
            config=WeightedBacktestConfig(
                transaction_cost_bps=0.0,
                initial_capital=INITIAL_CAPITAL,
                execution_delay_bars=EXECUTION_DELAY_BARS,
            ),
            risk_free_returns=risk_free_returns if name == "ibov_total_return" else None,
        )
        target = data_dir / "gold" / "backtests" / f"{prefix}_{name}"
        target.mkdir(parents=True, exist_ok=True)
        detail.curve.write_parquet(target / "equity_curve.parquet", compression="zstd")
        results[name] = {
            **asdict(detail.summary),
            "coverage_start": frame["trade_date"].min(),
            "coverage_end": frame["trade_date"].max(),
            "coverage_observations": frame.height,
            "source": str(path) if path.exists() else "BCB SGS 12 materialization",
        }
    return results


def _run_strategies(
    *,
    prices: pl.DataFrame,
    signals: pl.DataFrame,
    data_dir: Path,
    prefix: str,
    price_source: str,
    methodology: str,
    risk_free_returns: pl.DataFrame | None,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], dict[str, object]]:
    if "target_end_date" not in signals.columns:
        raise DataQualityError("strategy signals require target_end_date to bound evaluation")
    evaluation_end = signals["target_end_date"].max()
    prices = prices.filter(pl.col("trade_date") <= pl.lit(evaluation_end))
    strategies = {
        "universe_1n": ("signal_equal_weight", 100_000, 0),
        "top_liquidity_equal_weight": ("signal_liquidity", 50, 10),
        "ml_top10": ("signal_ml", 10, 0),
        "momentum_top10": ("signal_momentum", 10, 5),
        "low_volatility_top10": ("signal_low_volatility", 10, 5),
        "quality_top10": ("signal_quality", 10, 5),
        "multifactor_top10": ("signal_quant", 10, 5),
        "multifactor_learned_top10": ("signal_quant_learned", 10, 5),
        "research_v2": ("signal_research_v2", 10, 5),
    }
    if signals["quality_score"].n_unique() <= 1:
        strategies.pop("quality_top10")

    weighted_targets: dict[str, pl.DataFrame] = {}
    weighted_metadata: dict[str, dict[str, object]] = {}
    qkp_allocations = {
        "ml_qkp_ew": "equal_weight",
        "ml_qkp_inverse_vol": "inverse_vol",
        "ml_qkp_hrp": "hrp",
        "ml_qkp_cost_aware_minvar": "cost_aware_minvar",
    }
    for name, allocation in qkp_allocations.items():
        weights, diagnostics = build_monthly_qkp_weights(prices, signals, allocation=allocation)
        if not weights.is_empty():
            weighted_targets[name] = weights
            weighted_metadata[name] = {
                "allocation": allocation,
                "solver_diagnostics": diagnostics,
            }
    for name, allocation in {
        "risk_only_hrp": "hrp",
        "risk_only_minvar": "minvar",
    }.items():
        risk_weights = build_monthly_risk_benchmark_weights(
            prices,
            signals,
            allocation=allocation,
            estimator="ledoit_wolf",
            candidate_count=50,
        )
        if not risk_weights.is_empty():
            weighted_targets[name] = risk_weights
            weighted_metadata[name] = {"allocation": allocation}
    required_weighted = set(qkp_allocations) | {"risk_only_hrp", "risk_only_minvar"}
    if missing_weighted := required_weighted - set(weighted_targets):
        raise DataQualityError(
            "cannot build every comparable QKP/HRP/minvar strategy: "
            f"{sorted(missing_weighted)}"
        )

    common_dates = set(signals.get_column("trade_date").to_list())
    for weights in weighted_targets.values():
        common_dates &= set(weights.get_column("trade_date").to_list())
    if not common_dates:
        raise DataQualityError("B3 strategies have no common rebalance dates")
    ordered_common_dates = sorted(common_dates)
    comparable_signals = signals.filter(pl.col("trade_date").is_in(ordered_common_dates))
    weighted_targets = {
        name: weights.filter(pl.col("trade_date").is_in(ordered_common_dates))
        for name, weights in weighted_targets.items()
    }

    results: dict[str, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []
    fairness_records: list[dict[str, object]] = []

    def record_result(name: str, scenario: str, detail: object) -> None:
        summary = detail.summary
        if scenario == "registered_liquidity":
            results[name] = asdict(summary)
        comparison_rows.append(
            {
                "strategy": name,
                "cost": scenario,
                "gross": summary.gross_total_return,
                "fees": summary.fees_cost / summary.start_value,
                "spread": summary.spread_cost / summary.start_value,
                "impact": summary.impact_cost / summary.start_value,
                "net": summary.total_return,
                "turnover": summary.turnover,
                "fill_ratio": summary.fill_ratio,
                "rejected_notional": summary.rejected_notional,
            }
        )
        fairness_records.append(
            {
                "strategy": name,
                "cost_scenario": scenario,
                "initial_capital": summary.start_value,
                "evaluation_start": detail.curve["trade_date"].min(),
                "evaluation_end": detail.curve["trade_date"].max(),
                "rebalance_dates": tuple(ordered_common_dates),
                "rebalance_periods": summary.rebalance_periods,
                "execution_delay_bars": summary.execution_delay_bars,
                "participation_cap": COST_SCENARIOS[scenario].participation_cap,
                "fill_policy": FILL_POLICY,
            }
        )

    for scenario, cost_model in COST_SCENARIOS.items():
        suffix = "" if scenario == "registered_liquidity" else f"__{scenario}"
        for name, (column, top_k, selection_buffer) in strategies.items():
            config = BacktestConfig(
                top_k=top_k,
                selection_buffer=selection_buffer,
                initial_capital=INITIAL_CAPITAL,
                execution_delay_bars=EXECUTION_DELAY_BARS,
                cost_model=cost_model,
            )
            detail = run_monthly_topk_backtest_detailed(
                prices,
                comparable_signals,
                score_column=column,
                config=config,
                risk_free_returns=risk_free_returns,
            )
            target_name = f"{prefix}_{name}{suffix}"
            metadata = {
                "strategy": name,
                "cost_scenario": scenario,
                "score_column": column,
                "price_source": price_source,
                "selection_buffer": selection_buffer,
                "cost_model": asdict(cost_model),
                "cost_semantics": {
                    "fee_bps": "one-way fee on executed notional",
                    "quoted_full_spread_bps": "full bid/ask spread",
                    "one_way_half_spread_bps": "quoted full spread / 2; ADV tiers are fallback half-spreads",
                    "impact_bps": "one-way impact excluding fees and spread",
                },
                "methodology": methodology,
            }
            _write_result(data_dir, target_name, detail.curve, detail.summary, metadata=metadata)
            detail_dir = data_dir / "gold" / "backtests" / target_name
            detail.selections.write_parquet(detail_dir / "selections.parquet", compression="zstd")
            detail.contributions.write_parquet(detail_dir / "contributions.parquet", compression="zstd")
            detail.rejected_orders.write_parquet(detail_dir / "rejected_orders.parquet", compression="zstd")
            detail.costs.write_parquet(detail_dir / "costs.parquet", compression="zstd")
            record_result(name, scenario, detail)

        for name, weights in weighted_targets.items():
            config = WeightedBacktestConfig(
                initial_capital=INITIAL_CAPITAL,
                execution_delay_bars=EXECUTION_DELAY_BARS,
                cost_model=cost_model,
            )
            detail = run_monthly_weighted_backtest_detailed(
                prices, weights, config=config, risk_free_returns=risk_free_returns
            )
            target = data_dir / "gold" / "backtests" / f"{prefix}_{name}{suffix}"
            target.mkdir(parents=True, exist_ok=True)
            detail.curve.write_parquet(target / "equity_curve.parquet", compression="zstd")
            weights.write_parquet(target / "target_weights.parquet", compression="zstd")
            diagnostics = weighted_metadata[name].get("solver_diagnostics")
            if isinstance(diagnostics, pl.DataFrame):
                diagnostics.filter(pl.col("trade_date").is_in(ordered_common_dates)).write_parquet(
                    target / "solver_diagnostics.parquet", compression="zstd"
                )
            detail.rejected_orders.write_parquet(target / "rejected_orders.parquet", compression="zstd")
            detail.costs.write_parquet(target / "costs.parquet", compression="zstd")
            payload = asdict(detail.summary)
            (target / "summary.json").write_text(
                json.dumps(
                    {
                        **payload,
                        "strategy": name,
                        "cost_scenario": scenario,
                        "cost_model": asdict(cost_model),
                        "allocation": weighted_metadata[name]["allocation"],
                        "methodology": methodology,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            record_result(name, scenario, detail)

    fairness = evaluate_benchmark_fairness(fairness_records)
    if not fairness["passed"]:
        raise DataQualityError(f"benchmark fairness failed: {fairness['failures']}")
    comparison = pl.DataFrame(comparison_rows).sort("cost", "strategy")
    comparison.write_parquet(
        data_dir / "gold" / "backtests" / f"{prefix}_strategy_by_cost.parquet",
        compression="zstd",
    )
    (data_dir / "gold" / "backtests" / f"{prefix}_strategy_by_cost.json").write_text(
        json.dumps(
            {"semantics": STRATEGY_COST_TABLE_SEMANTICS, "rows": comparison_rows},
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    results.update(
        _run_reference_benchmarks(
            data_dir=data_dir,
            signals=comparable_signals,
            prefix=prefix,
            risk_free_returns=risk_free_returns,
        )
    )
    return results, comparison_rows, fairness


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
    required = {
        "target_end_date",
        "target_excess_return",
        "target_cross_sectional_rank",
        "execution_lag_bars",
        "target_horizon_bars",
    }
    if missing := required - set(features.columns):
        raise SystemExit(f"Feature panel predates leakage-safe protocol. Rebuild it; missing {sorted(missing)}")
    feature_names = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in features.columns]
    prices, price_source = _load_prices(data_dir)
    risk_free_returns, cdi_status = _load_cdi_returns(data_dir)
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
    outer_fold_configs: dict[int, dict] = {}
    if not tuned_path.exists():
        raise SystemExit("Missing nested tuning artifact. Run make tune first.")
    if tuned_path.exists():
        metadata = json.loads(tuned_path.read_text(encoding="utf-8"))
        if (
            int(metadata.get("development_end_year", -1)) == protocol.development_end_year
            and int(metadata.get("diagnostic_year", -1)) == protocol.diagnostic_year
            and int(metadata.get("target_horizon_bars", -1)) == protocol.label_horizon_bars
        ):
            model_config = metadata.get("config")
            outer_fold_configs = {
                int(year): config
                for year, config in metadata.get("outer_fold_configs", {}).items()
            }
            if not outer_fold_configs:
                raise SystemExit(
                    "Tuned hyperparameters predate nested outer folds. Run make tune again."
                )
        else:
            raise SystemExit("Tuned hyperparameters do not match the registered research protocol")

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
        results, strategy_by_cost, benchmark_fairness = _run_strategies(
            prices=prices,
            signals=signals,
            data_dir=data_dir,
            prefix=f"diagnostic_{protocol.diagnostic_year}",
            price_source=price_source,
            methodology=(
                f"diagnostic only; artifact knowledge_cutoff={model.knowledge_cutoff}; "
                f"target_end_max={model.target_end_max}; no diagnostic-year selection"
            ),
            risk_free_returns=risk_free_returns,
        )
        report = {
            "diagnostic_year": protocol.diagnostic_year,
            "warning": protocol.warning,
            "accepted": None,
            "reason": "Diagnostic-year performance is prohibited from changing acceptance or model selection.",
            "strategies": results,
            "strategy_by_cost": strategy_by_cost,
            "strategy_by_cost_semantics": STRATEGY_COST_TABLE_SEMANTICS,
            "benchmark_fairness": benchmark_fairness,
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
    evaluation_years = sorted(outer_fold_configs)
    for year in [value for value in years if value in evaluation_years]:
        train = development.filter(pl.col("trade_date").dt.year() < year)
        test = development.filter(pl.col("trade_date").dt.year() == year)
        if train.height < 500 or test.height < 50:
            continue
        model, metrics = train_once(
            train,
            test,
            feature_names=feature_names,
            seed=year,
            model_config=outer_fold_configs[year],
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
    results, strategy_by_cost, benchmark_fairness = _run_strategies(
        prices=prices,
        signals=signals,
        data_dir=data_dir,
        prefix="development",
        price_source=price_source,
        methodology=(
            f"nested outer-fold walk-forward ({selected_policy['selected_name']}); "
            "per-fold hyperparameters selected on earlier inner folds only; next-close execution; "
            "development data through 2025 only"
        ),
        risk_free_returns=risk_free_returns,
    )

    topk = yearly_topk_report(
        signals,
        signals["predicted_excess_return"].to_numpy(),
        target_column="target_excess_return",
    )
    topk_path = data_dir / "gold" / "backtests" / "development_topk_by_year.parquet"
    topk.write_parquet(topk_path, compression="zstd")
    monthly_ics: list[float] = []
    monthly_spreads: list[float] = []
    recent_signals = signals.filter(pl.col("trade_date").dt.year().is_between(2022, 2025))
    for group in recent_signals.partition_by("trade_date", maintain_order=True):
        truth = group["target_excess_return"].to_numpy()
        prediction = group["predicted_excess_return"].to_numpy()
        mask = np.isfinite(truth) & np.isfinite(prediction)
        if mask.sum() < 10:
            continue
        statistic = spearmanr(truth[mask], prediction[mask]).statistic
        if statistic is not None and np.isfinite(statistic):
            monthly_ics.append(float(statistic))
        metric = top_k_metrics(truth[mask], prediction[mask], k=10)
        if np.isfinite(metric.top_minus_bottom):
            monthly_spreads.append(metric.top_minus_bottom)
    rank_ic_ci = moving_block_bootstrap_ci(
        np.asarray(monthly_ics), lambda values: float(np.mean(values)), block_length=3
    )
    spread_ci = moving_block_bootstrap_ci(
        np.asarray(monthly_spreads), lambda values: float(np.mean(values)), block_length=3
    )
    recent_rank_ic = rank_ic_ci[0]
    top10_spread = spread_ci[0]
    research = results["research_v2"]
    cdi_excess_ci = _differential_ci(data_dir, "development", "cdi")
    ibov_excess_ci = _differential_ci(data_dir, "development", "ibov_total_return")
    research_curve = pl.read_parquet(
        data_dir / "gold" / "backtests" / "development_research_v2" / "equity_curve.parquet"
    )
    monthly_excess = _monthly_returns(research_curve, column="excess_return")["return"].to_numpy()
    strategy_sharpes: list[float] = []
    cdi_curve_path = data_dir / "gold" / "backtests" / "development_cdi" / "equity_curve.parquet"
    cdi_monthly = (
        _monthly_returns(pl.read_parquet(cdi_curve_path)).rename({"return": "cdi"})
        if cdi_curve_path.exists()
        else None
    )
    for name, payload in results.items():
        curve_path = data_dir / "gold" / "backtests" / f"development_{name}" / "equity_curve.parquet"
        if "cagr" not in payload or not curve_path.exists():
            continue
        trial_monthly = _monthly_returns(pl.read_parquet(curve_path))
        if cdi_monthly is not None:
            trial_monthly = trial_monthly.join(cdi_monthly, on="month", how="inner").with_columns(
                (pl.col("return") - pl.col("cdi")).alias("trial_excess")
            )
            strategy_sharpes.append(
                sharpe_from_periodic_returns(trial_monthly["trial_excess"].to_numpy())
            )
    configured_trials = int(metadata.get("number_of_experiments", metadata.get("n_trials", 1))) if tuned_path.exists() else 1
    number_of_trials = max(configured_trials + len(strategy_sharpes), 1)
    trials_sharpe_std = float(np.std(strategy_sharpes, ddof=1)) if len(strategy_sharpes) > 1 else 0.0
    dsr_probability, dsr_hurdle = deflated_sharpe_ratio(
        monthly_excess,
        number_of_trials=number_of_trials,
        trials_sharpe_std=trials_sharpe_std,
    )
    repository_root = Path(__file__).resolve().parents[1]
    test_manifest = data_dir / "gold" / "test_manifests" / "research_tests.json"
    leakage_tests_passed, test_manifest_reason = verify_test_manifest(
        test_manifest, repository_root=repository_root
    )
    gate = evaluate_acceptance_gates(
        rank_ic_ci_low=rank_ic_ci[1],
        top10_spread_ci_low=spread_ci[1],
        excess_sharpe=float(research["sharpe"]) if risk_free_returns is not None else float("nan"),
        max_drawdown=float(research["max_drawdown"]),
        annualized_turnover=float(research["annualized_turnover"]),
        relative_cagr_ci_low_vs_cdi=cdi_excess_ci[1],
        relative_cagr_ci_low_vs_ibov=ibov_excess_ci[1],
        deflated_sharpe_probability=dsr_probability,
        number_of_trials=number_of_trials,
        leakage_tests_passed=leakage_tests_passed,
        diagnostic_only_acknowledged=(
            protocol.diagnostic_year > protocol.development_end_year
            and protocol.diagnostic_start > protocol.development_knowledge_cutoff
        ),
        benchmark_fairness_passed=bool(benchmark_fairness["passed"]),
    )
    report = {
        "protocol": {
            "development_end_year": protocol.development_end_year,
            "diagnostic_year": protocol.diagnostic_year,
            "warning": protocol.warning,
        },
        "price_source": price_source,
        "cdi": cdi_status,
        "feature_panel": str(feature_path),
        "fold_metrics": fold_metrics,
        "strategies": results,
        "strategy_by_cost": strategy_by_cost,
        "strategy_by_cost_semantics": STRATEGY_COST_TABLE_SEMANTICS,
        "benchmark_fairness": benchmark_fairness,
        "inference": {
            "block_length_months": 3,
            "block_length_rationale": "Conservative dependence allowance exceeding the 21-bar target/holding overlap.",
            "rank_ic": {"estimate": rank_ic_ci[0], "ci_low": rank_ic_ci[1], "ci_high": rank_ic_ci[2]},
            "top10_spread": {"estimate": spread_ci[0], "ci_low": spread_ci[1], "ci_high": spread_ci[2]},
            "relative_cagr_vs_cdi": {"estimate": cdi_excess_ci[0], "ci_low": cdi_excess_ci[1], "ci_high": cdi_excess_ci[2]},
            "relative_cagr_vs_ibov": {"estimate": ibov_excess_ci[0], "ci_low": ibov_excess_ci[1], "ci_high": ibov_excess_ci[2]},
            "deflated_sharpe_probability": dsr_probability,
            "deflated_sharpe_hurdle": dsr_hurdle,
            "number_of_trials": number_of_trials,
        },
        "test_manifest": {"path": str(test_manifest), "verified": leakage_tests_passed, "reason": test_manifest_reason},
        "acceptance": gate,
        "accepted": bool(gate["production_ready"]),
        "important": "2026 is not included in these gates.",
    }
    out = data_dir / "gold" / "backtests" / "comparison_development.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    experiment = write_experiment_record(
        data_dir=data_dir,
        features=feature_names,
        parameters={
            "outer_fold_configs": outer_fold_configs,
            "final_artifact_model_config_not_used_for_outer_scoring": model_config,
            "registered_cost_model": asdict(REGISTERED_COST_MODEL),
            "pre_registered_impact_stress": asdict(VOLATILITY_SCALED_IMPACT_STRESS),
        },
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
