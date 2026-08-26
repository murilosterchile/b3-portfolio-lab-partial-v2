from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import polars as pl

from portfolio_core.features.fundamental import FUNDAMENTAL_FEATURES
from portfolio_core.ml.tuning import tune_ensemble
from portfolio_core.ml.walk_forward import DEFAULT_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--validation-years", type=int, default=3)
    args = parser.parse_args()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    fundamental_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    technical_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    feature_path = fundamental_path if fundamental_path.exists() else technical_path
    if not feature_path.exists():
        raise SystemExit("No feature panel. Ingest B3 history first.")
    frame = pl.read_parquet(feature_path).filter(pl.col("target_excess_return").is_not_null())
    years = sorted(set(frame["trade_date"].dt.year().to_list()))
    if len(years) < 5:
        raise SystemExit("At least five years are recommended: tuning keeps the latest year untouched.")
    holdout_year = years[-1]
    tuning_frame = frame.filter(pl.col("trade_date").dt.year() < holdout_year)
    feature_names = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in frame.columns]
    result = tune_ensemble(
        tuning_frame,
        feature_names=feature_names,
        n_trials=args.trials,
        n_validation_years=args.validation_years,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_score_rank_ic": result.best_score,
        "validation_years": result.validation_years,
        "untouched_holdout_year": holdout_year,
        "n_trials": result.n_trials,
        "features": feature_names,
        "config": result.best_config,
        "warning": "Hyperparameters were tuned only before the untouched holdout year.",
    }
    out = model_dir / "tuned_hyperparameters.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved={out} best_rank_ic={result.best_score:.6f} holdout={holdout_year}")


if __name__ == "__main__":
    main()
