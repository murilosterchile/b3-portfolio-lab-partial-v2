from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl

from portfolio_core.data_quality import filter_labels_known_by
from portfolio_core.features.fundamental import FUNDAMENTAL_FEATURES
from portfolio_core.ml.diagnostics import (
    feature_drift_report,
    feature_ic_report,
    model_importance,
    yearly_topk_report,
)
from portfolio_core.ml.protocol import ResearchProtocol
from portfolio_core.ml.walk_forward import (
    DEFAULT_FEATURES,
    TrainingPolicy,
    train_once,
    train_ranker_once,
)


def _candidate_policies() -> list[tuple[str, TrainingPolicy]]:
    policies = [("expanding_equal", TrainingPolicy())]
    policies.extend(
        (f"rolling_{years}y_equal", TrainingPolicy(window_years=years))
        for years in (3, 5, 7, 10)
    )
    policies.extend(
        (f"expanding_half_life_{years}y", TrainingPolicy(half_life_years=float(years)))
        for years in (1, 2, 3, 5)
    )
    return policies


def _evaluate_policy(
    frame: pl.DataFrame,
    *,
    features: list[str],
    policy: TrainingPolicy,
    model_config: dict | None,
    model_kind: str = "regression",
) -> tuple[dict[str, object], pl.DataFrame, pl.DataFrame]:
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pl.DataFrame] = []
    importance_frames: list[pl.DataFrame] = []
    for year in range(2019, 2026):
        train = frame.filter(pl.col("trade_date").dt.year() < year)
        valid = frame.filter(pl.col("trade_date").dt.year() == year)
        if train.height < 500 or valid.height < 100:
            continue
        trainer = train_ranker_once if model_kind == "ranking" else train_once
        model, metrics = trainer(
            train,
            valid,
            feature_names=features,
            seed=year,
            model_config=model_config,
            training_policy=policy,
        )
        pred = model.predict(valid)
        prediction_frames.append(
            valid.select("trade_date", "ticker", policy.target_column).join(
                pred.select("trade_date", "ticker", "predicted_excess_return"),
                on=["trade_date", "ticker"],
                how="inner",
            ).with_columns(pl.lit(year).alias("fold_year"))
        )
        fold_rows.append(
            {
                "year": year,
                **asdict(metrics),
                "disabled_features": list(model.disabled_feature_names),
                "training_feature_coverage": model.training_feature_coverage,
            }
        )
        if model_kind == "regression":
            importance_frames.append(model_importance(model, fold_year=year))
    if not fold_rows:
        raise RuntimeError("No eligible development folds for policy")
    fold = pl.DataFrame(fold_rows)
    predictions = pl.concat(prediction_frames, how="vertical_relaxed")
    topk = yearly_topk_report(
        predictions,
        predictions["predicted_excess_return"].to_numpy(),
        target_column=policy.target_column,
    )
    top10 = topk.filter(pl.col("k") == 10)
    rank = fold["rank_ic"].to_numpy()
    recent_fold = fold.filter(pl.col("year").is_between(2022, 2025))
    recent_rank = recent_fold["rank_ic"].to_numpy()
    spread = top10["top_minus_bottom"].to_numpy() if not top10.is_empty() else np.asarray([])
    previous: set[str] = set()
    turnovers: list[float] = []
    for cross in predictions.partition_by("trade_date", maintain_order=True):
        selected = set(
            cross.sort("predicted_excess_return", descending=True).head(10)["ticker"].to_list()
        )
        if previous:
            turnovers.append(1.0 - len(previous & selected) / max(len(selected), 1))
        previous = selected
    mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    gross_spread = float(np.mean(spread)) if len(spread) else None
    summary = {
        "mean_rank_ic": float(np.mean(rank)),
        "median_rank_ic": float(np.median(rank)),
        "rank_ic_std": float(np.std(rank, ddof=1)) if len(rank) > 1 else 0.0,
        "rank_ic_worst_fold": float(np.min(rank)),
        "recent_mean_rank_ic": float(np.mean(recent_rank)) if len(recent_rank) else None,
        "mean_precision_at_10": float(top10["precision"].mean()) if not top10.is_empty() else None,
        "mean_ndcg_at_10": float(top10["ndcg"].mean()) if not top10.is_empty() else None,
        "mean_top10_spread": gross_spread,
        "mean_top10_turnover": mean_turnover,
        "mean_top10_net_spread_15bps": (
            gross_spread - mean_turnover * 15.0 / 10_000.0
            if gross_spread is not None
            else None
        ),
        "model_kind": model_kind,
        "folds": fold_rows,
        "training_policy": {
            "window_years": policy.window_years,
            "half_life_years": policy.half_life_years,
            "target_column": policy.target_column,
            "label_end_column": policy.label_end_column,
            "pre_validation_embargo_days": policy.pre_validation_embargo_days,
        },
    }
    importance = pl.concat(importance_frames, how="vertical_relaxed") if importance_frames else pl.DataFrame()
    return summary, topk, importance


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Development-only diagnostics for the post-2021 B3 signal deterioration"
    )
    parser.add_argument("--skip-policy-grid", action="store_true")
    args = parser.parse_args()
    protocol = ResearchProtocol()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    panel = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    if not panel.exists():
        panel = data_dir / "gold" / "features" / "monthly_features.parquet"
    if not panel.exists():
        raise SystemExit("No feature panel is available")
    frame = pl.read_parquet(panel)
    required = {
        "target_end_date",
        "target_excess_return",
        "target_cross_sectional_rank",
        "universe_eligible",
    }
    if missing := required - set(frame.columns):
        raise SystemExit(
            "Rebuild features first: investible PIT targets are required for "
            f"leakage-safe diagnostics. Missing: {sorted(missing)}"
        )
    development = filter_labels_known_by(
        frame, knowledge_cutoff=protocol.development_knowledge_cutoff
    )
    diagnostic = frame.filter(pl.col("trade_date").dt.year() == protocol.diagnostic_year)
    features = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in frame.columns]
    out_dir = data_dir / "gold" / "research_diagnostics" / "post_2021_degradation"
    out_dir.mkdir(parents=True, exist_ok=True)

    drift = feature_drift_report(development, features)
    drift.write_csv(out_dir / "feature_drift_regime_a_vs_b.csv")
    annual_ic, regime_ic = feature_ic_report(development, features)
    annual_ic.write_csv(out_dir / "feature_ic_by_year.csv")
    regime_ic.write_csv(out_dir / "feature_ic_regime_comparison.csv")

    # Policy selection deliberately uses the pre-registered default model. A
    # final tuned config has seen later development folds and cannot be replayed
    # into earlier policy folds without data snooping.
    model_config = None

    policy_results: dict[str, object] = {}
    ranking_comparison_policy = TrainingPolicy()
    if not args.skip_policy_grid:
        for name, policy in _candidate_policies():
            print(f"policy={name}")
            summary, topk, importance = _evaluate_policy(
                development,
                features=features,
                policy=policy,
                model_config=model_config,
            )
            policy_results[name] = summary
            topk.write_csv(out_dir / f"topk_{name}.csv")
            if not importance.is_empty():
                importance.write_csv(out_dir / f"model_importance_{name}.csv")
        _write_json(out_dir / "training_policy_comparison.json", policy_results)
        eligible: list[tuple[float, str, dict[str, object]]] = []
        for name, raw in policy_results.items():
            summary = dict(raw)
            recent = summary.get("recent_mean_rank_ic")
            spread = summary.get("mean_top10_net_spread_15bps")
            if recent is None or spread is None:
                continue
            score = float(recent) + float(spread) - 0.50 * float(summary["rank_ic_std"])
            summary["selection_score"] = score
            # Pre-registered stability gate. A challenger is not selected merely
            # because it is the least bad one.
            if (
                float(summary["mean_rank_ic"]) > 0
                and float(recent) > 0
                and float(summary["rank_ic_worst_fold"]) > -0.10
                and float(spread) > 0
            ):
                eligible.append((score, name, summary))
        if eligible:
            score, name, selected = max(eligible, key=lambda item: item[0])
            ranking_comparison_policy = TrainingPolicy(**selected["training_policy"])
            model_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                model_dir / "selected_training_policy.json",
                {
                    "version": 1,
                    "selected_name": name,
                    "selection_score": score,
                    "training_policy": selected["training_policy"],
                    "development_end_year": protocol.development_end_year,
                    "diagnostic_year": protocol.diagnostic_year,
                    "selection_data_max": str(development["trade_date"].max()),
                    "all_challengers": policy_results,
                    "warning": protocol.warning,
                },
            )

    target_results: dict[str, object] = {}
    for target in ("future_return", "target_excess_return", "target_cross_sectional_rank"):
        if target not in development.columns:
            continue
        name = f"target_{target}"
        print(name)
        summary, topk, _ = _evaluate_policy(
            development,
            features=features,
            policy=TrainingPolicy(target_column=target),
            model_config=model_config,
        )
        target_results[target] = summary
        topk.write_csv(out_dir / f"topk_{name}.csv")
    _write_json(out_dir / "target_comparison.json", target_results)

    ranking_results: dict[str, object] = {}
    for model_kind in ("regression", "ranking"):
        summary, topk, _ = _evaluate_policy(
            development,
            features=features,
            policy=ranking_comparison_policy,
            model_config=model_config,
            model_kind=model_kind,
        )
        ranking_results[model_kind] = summary
        topk.write_csv(out_dir / f"topk_model_{model_kind}.csv")
    _write_json(out_dir / "regression_vs_ranking.json", ranking_results)

    manifest = {
        "feature_panel": str(panel),
        "features": features,
        "regime_a": "2014-2021",
        "regime_b": "2022-2025",
        "diagnostic_year": protocol.diagnostic_year,
        "diagnostic_rows_present_but_unused_for_selection": diagnostic.height,
        "warning": protocol.warning,
        "outputs": [path.name for path in sorted(out_dir.iterdir())],
        "selection_status": (
            "No automatic policy promotion. Inspect development-fold stability, lock a candidate, "
            "then train it with a <=2025 knowledge cutoff before the one-way 2026 diagnostic."
        ),
    }
    _write_json(out_dir / "manifest.json", manifest)
    print(f"diagnostics={out_dir}")
    print(protocol.warning)


if __name__ == "__main__":
    main()
