from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import polars as pl
from portfolio_core.backtest import BacktestConfig, run_monthly_topk_backtest
from portfolio_core.data_quality import invalid_price_rows
from portfolio_core.features.fundamental import FUNDAMENTAL_FEATURES
from portfolio_core.ml.walk_forward import (
    DEFAULT_FEATURES,
    load_signal_model,
    train_once,
)
from portfolio_core.quant.factors import composite_quant_score


def _load_prices(data_dir: Path) -> tuple[pl.DataFrame, str]:
    adjusted = sorted(
        (data_dir / "silver" / "b3_prices_adjusted").glob("year=*/part-000.parquet")
    )
    raw = sorted((data_dir / "silver" / "b3_prices").glob("year=*/part-000.parquet"))
    parts = adjusted or raw
    if len(parts) < 3:
        raise SystemExit("Need at least three B3 yearly price partitions")
    source = "B3 adjusted total-return analytical prices" if adjusted else "raw B3 COTAHIST"
    return pl.concat([pl.read_parquet(p) for p in parts], how="vertical_relaxed"), source


def _percentile(expr: pl.Expr) -> pl.Expr:
    return expr.rank(method="average").over("trade_date") / pl.len().over("trade_date")


def _strategy_signals(frame: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    return frame.join(
        predictions.select(
            "trade_date", "ticker", "predicted_excess_return", "prediction_uncertainty"
        ),
        on=["trade_date", "ticker"],
        how="inner",
    ).with_columns(
        _percentile(pl.col("predicted_excess_return")).alias("ml_percentile"),
        _percentile(pl.col("prediction_uncertainty")).alias("uncertainty_percentile"),
    ).with_columns(
        (pl.col("predicted_excess_return") - 0.5 * pl.col("prediction_uncertainty")).alias(
            "signal_ml"
        ),
        pl.col("rank_momentum_12_1").fill_null(0.5).alias("signal_momentum"),
        pl.col("quant_score").fill_null(0.5).alias("signal_quant"),
        (
            0.65 * pl.col("ml_percentile")
            + 0.25 * pl.col("quant_score").fill_null(0.5)
            + 0.10 * (1.0 - pl.col("uncertainty_percentile"))
        ).alias("signal_research_v2"),
    )


def _features_after_training(features: pl.DataFrame, trained_until: object) -> pl.DataFrame:
    future = features.filter(pl.col("trade_date") > pl.lit(trained_until))
    if future.is_empty():
        raise SystemExit(
            f"No feature rows after model trained_until={trained_until}. "
            "Ingest/build newer data before running this backtest."
        )
    return future


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
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-trained-model",
        action="store_true",
        help="load MODEL_DIR/signal_model.joblib and evaluate only dates after trained_until",
    )
    args = parser.parse_args()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    fundamental_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    technical_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    feature_path = fundamental_path if fundamental_path.exists() else technical_path
    if not feature_path.exists():
        raise SystemExit("Need a monthly feature panel")
    features = pl.read_parquet(feature_path)
    feature_names = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in features.columns]
    print(f"feature_panel={feature_path} features={len(feature_names)}")
    prices, price_source = _load_prices(data_dir)
    print(f"price_source={price_source}")

    print("\nDATA QUALITY")
    print("------------")
    print(f"Rows                     {prices.height:,}")
    print(f"Tickers                  {prices['ticker'].n_unique():,}")
    print(f"Date range               {prices['trade_date'].min()} -> {prices['trade_date'].max()}")
    invalid_prices = invalid_price_rows(prices)
    print(f"Invalid source prices    {invalid_prices.height:,}")
    for row in invalid_prices.select("ticker", "trade_date", "close").head(10).iter_rows(named=True):
        print(f"- {row['ticker']} {row['trade_date']} close={row['close']}")
    duplicate_prices = prices.group_by(["ticker", "trade_date"]).len().filter(pl.col("len") > 1)
    print(f"Duplicate ticker/date    {duplicate_prices.height:,}")
    if "adjusted_close" in prices.columns:
        invalid_adjusted = invalid_price_rows(prices, price_column="adjusted_close")
        print(f"Invalid adjusted prices  {invalid_adjusted.height:,}")

    print("\nFEATURE COVERAGE")
    print("----------------")
    coverage: dict[str, float] = {}
    for name in feature_names:
        ratio = float(
            features.select((pl.col(name).is_not_null() & pl.col(name).is_finite()).mean()).item()
        )
        coverage[name] = ratio
        print(f"{name:30s} {ratio:7.2%}")
    technical_coverage = [coverage[name] for name in DEFAULT_FEATURES]
    fundamental_coverage = [coverage[name] for name in FUNDAMENTAL_FEATURES if name in coverage]
    print(f"technical median               {float(pl.Series(technical_coverage).median()):7.2%}")
    if fundamental_coverage:
        print(f"fundamental median             {float(pl.Series(fundamental_coverage).median()):7.2%}")

    disabled_by_fold: dict[int, tuple[str, ...]] = {}
    training_rows = 0
    validation_rows = 0
    model_path: Path | None = None
    trained_until = None
    if args.use_trained_model:
        model_path = model_dir / "signal_model.joblib"
        if not model_path.exists():
            raise SystemExit(f"Trained model not found: {model_path}. Run make train first.")
        model = load_signal_model(str(model_path))
        trained_until = model.trained_until
        missing_features = sorted(set(model.feature_names) - set(features.columns))
        if missing_features:
            raise SystemExit(f"Feature panel is missing trained model features: {missing_features}")
        future = _features_after_training(features, trained_until)
        scored = composite_quant_score(future)
        signals = _strategy_signals(scored, model.predict(future)).with_columns(
            pl.col("trade_date").dt.year().alias("oos_year")
        )
        print(f"loaded_model={model_path} trained_until={trained_until}")
    else:
        years = sorted(set(features["trade_date"].dt.year().to_list()))
        prediction_frames: list[pl.DataFrame] = []
        for year in years[3:]:
            train = features.filter(
                (pl.col("trade_date").dt.year() < year)
                & pl.col("target_excess_return").is_not_null()
            )
            test = features.filter(pl.col("trade_date").dt.year() == year)
            if train.height < 500 or test.height < 50:
                continue
            prev = year - 1
            fit = train.filter(pl.col("trade_date").dt.year() < prev)
            valid = train.filter(pl.col("trade_date").dt.year() == prev)
            if fit.height < 400 or valid.height < 50:
                continue
            model, metrics = train_once(fit, valid, feature_names=feature_names, seed=year)
            disabled_by_fold[year] = model.disabled_feature_names
            training_rows += fit.height
            validation_rows += valid.height
            scored = composite_quant_score(test)
            combined = _strategy_signals(scored, model.predict(test)).with_columns(
                pl.lit(year).alias("oos_year")
            )
            print(
                f"fold={year} validation_rank_ic={metrics.rank_ic:.4f} "
                f"top_minus_bottom={metrics.top_minus_bottom:.4f}"
            )
            prediction_frames.append(combined)
        if not prediction_frames:
            raise SystemExit("Insufficient history for walk-forward predictions")
        signals = pl.concat(prediction_frames, how="vertical_relaxed")

    print("\nMODEL")
    print("-----")
    print(f"mode                     {'trained artifact' if args.use_trained_model else 'walk-forward'}")
    if not args.use_trained_model:
        print(f"training rows            {training_rows:,}")
        print(f"validation rows          {validation_rows:,}")
    disabled_union = sorted({name for values in disabled_by_fold.values() for name in values})
    print(f"features disabled        {len(disabled_union)}")
    for year, disabled in disabled_by_fold.items():
        print(f"- fold {year}: {', '.join(disabled) if disabled else 'none'}")
    print("preprocessing            median -> variance threshold -> StandardScaler")

    methodology = (
        "saved trained artifact; evaluated only after trained_until"
        if args.use_trained_model
        else "strict walk-forward OOS; no tuning on reported OOS years"
    )

    strategies = {
        "ml_top10": ("signal_ml", BacktestConfig(top_k=10, transaction_cost_bps=15.0)),
        "momentum_top10": (
            "signal_momentum",
            BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5),
        ),
        "multifactor_top10": (
            "signal_quant",
            BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5),
        ),
        "research_v2": (
            "signal_research_v2",
            BacktestConfig(top_k=10, transaction_cost_bps=15.0, selection_buffer=5),
        ),
    }
    results: dict[str, dict[str, object]] = {}
    for name, (column, config) in strategies.items():
        curve, summary = run_monthly_topk_backtest(
            prices, signals, score_column=column, config=config
        )
        metadata = {
            "strategy": name,
            "score_column": column,
            "price_source": price_source,
            "selection_buffer": config.selection_buffer,
            "transaction_cost_bps": config.transaction_cost_bps,
            "methodology": methodology,
            "model_path": str(model_path) if model_path else None,
            "model_trained_until": str(trained_until) if trained_until else None,
        }
        output_name = f"trained_model_{name}" if args.use_trained_model else name
        _write_result(data_dir, output_name, curve, summary, metadata=metadata)
        results[name] = asdict(summary)
        print(
            f"{name:20s} CAGR={summary.cagr:8.2%} Sharpe={summary.sharpe:7.3f} "
            f"MDD={summary.max_drawdown:8.2%} turnover/y={summary.annualized_turnover:6.2f}x"
        )

    research = results["research_v2"]
    # Pre-registered gate: it reports pass/fail; it never changes the strategy to force a pass.
    gate = {
        "positive_cagr": float(research["cagr"]) > 0.0,
        "positive_sharpe": float(research["sharpe"]) > 0.0,
        "max_drawdown_better_than_minus_65pct": float(research["max_drawdown"]) > -0.65,
        "turnover_below_6x_per_year": float(research["annualized_turnover"]) < 6.0,
        "beats_momentum_cagr": float(research["cagr"]) > float(results["momentum_top10"]["cagr"]),
    }
    report = {
        "price_source": price_source,
        "feature_panel": str(feature_path),
        "methodology": methodology,
        "model_path": str(model_path) if model_path else None,
        "model_trained_until": str(trained_until) if trained_until else None,
        "strategies": results,
        "research_v2_gate": gate,
        "accepted": all(gate.values()),
        "warning": (
            "PASS means this historical OOS protocol met pre-registered criteria; it is not a "
            "guarantee of future positive returns."
        ),
    }
    report_name = "comparison_trained_model.json" if args.use_trained_model else "comparison.json"
    report_path = data_dir / "gold" / "backtests" / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"comparison={report_path} accepted={report['accepted']}")


if __name__ == "__main__":
    main()
