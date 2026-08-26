from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl
from portfolio_core.features.fundamental import FUNDAMENTAL_FEATURES
from portfolio_core.ml.protocol import ResearchProtocol
from portfolio_core.ml.tuning import tune_ensemble
from portfolio_core.ml.walk_forward import DEFAULT_FEATURES, TrainingPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--validation-years", type=int, default=3)
    args = parser.parse_args()
    protocol = ResearchProtocol()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    fundamental_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    technical_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    feature_path = fundamental_path if fundamental_path.exists() else technical_path
    if not feature_path.exists():
        raise SystemExit("No feature panel. Ingest B3 history first.")
    frame = pl.read_parquet(feature_path)
    required = {"target_excess_return", "target_end_date"}
    if missing := required - set(frame.columns):
        raise SystemExit(f"Feature panel predates leakage-safe labels; rebuild it. Missing: {sorted(missing)}")
    development = frame.filter(pl.col("trade_date").dt.year() <= protocol.development_end_year)
    feature_names = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in frame.columns]
    result = tune_ensemble(
        development,
        feature_names=feature_names,
        n_trials=args.trials,
        n_validation_years=args.validation_years,
        training_policy=TrainingPolicy(),
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_score_rank_ic": result.best_score,
        "validation_years": result.validation_years,
        "development_end_year": protocol.development_end_year,
        "diagnostic_year": protocol.diagnostic_year,
        "n_trials": result.n_trials,
        "features": feature_names,
        "config": result.best_config,
        "warning": protocol.warning,
        "selection_rule": "No diagnostic-year observation was used for hyperparameter selection.",
    }
    out = model_dir / "tuned_hyperparameters.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved={out} best_rank_ic={result.best_score:.6f}")
    print(protocol.warning)


if __name__ == "__main__":
    main()
