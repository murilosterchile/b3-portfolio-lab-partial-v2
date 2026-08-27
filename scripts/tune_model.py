from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl
from portfolio_core.data_quality import filter_labels_known_by
from portfolio_core.features.fundamental import FUNDAMENTAL_FEATURES
from portfolio_core.ml.protocol import ResearchProtocol
from portfolio_core.ml.tuning import tune_ensemble
from portfolio_core.ml.walk_forward import DEFAULT_FEATURES, TrainingPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--validation-years", type=int, default=3)
    parser.add_argument("--outer-years", type=int, default=3)
    parser.add_argument("--feature-path", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    protocol = ResearchProtocol()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    fundamental_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    technical_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    feature_path = Path(args.feature_path) if args.feature_path else (
        fundamental_path if fundamental_path.exists() else technical_path
    )
    if not feature_path.exists():
        raise SystemExit("No feature panel. Ingest B3 history first.")
    frame = pl.read_parquet(feature_path)
    required = {"target_excess_return", "target_end_date"}
    if missing := required - set(frame.columns):
        raise SystemExit(f"Feature panel predates leakage-safe labels; rebuild it. Missing: {sorted(missing)}")
    development = filter_labels_known_by(
        frame, knowledge_cutoff=protocol.development_knowledge_cutoff
    )
    horizon_values = frame.get_column("target_horizon_bars").unique().to_list() if "target_horizon_bars" in frame.columns else []
    if len(horizon_values) != 1:
        raise SystemExit(f"Expected one target horizon in the feature panel, got {horizon_values}")
    feature_names = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in frame.columns]
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
    result = tune_ensemble(
        development,
        feature_names=feature_names,
        n_trials=args.trials,
        n_validation_years=args.validation_years,
        n_outer_years=args.outer_years,
        training_policy=policy,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "outer_mean_rank_ic": result.best_score,
        "outer_fold_scores": result.outer_fold_scores,
        "outer_fold_configs": result.outer_fold_configs,
        "validation_years": result.validation_years,
        "development_end_year": protocol.development_end_year,
        "diagnostic_year": protocol.diagnostic_year,
        "n_trials": result.n_trials,
        "number_of_experiments": result.number_of_experiments,
        "trial_history": result.trial_history,
        "features": feature_names,
        "config": result.best_config,
        "training_policy": selected_policy,
        "warning": protocol.warning,
        "selection_rule": (
            "Nested walk-forward: Optuna uses inner folds only; outer folds are untouched reporting-only; "
            "the final configuration is selected by an inner study through 2025."
        ),
        "target_horizon_bars": horizon_values[0],
    }
    out = Path(args.output) if args.output else model_dir / "tuned_hyperparameters.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved={out} outer_mean_rank_ic={result.best_score:.6f}")
    print(protocol.warning)


if __name__ == "__main__":
    main()
