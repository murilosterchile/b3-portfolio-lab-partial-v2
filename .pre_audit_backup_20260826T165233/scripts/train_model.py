from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import polars as pl

from portfolio_core.features.fundamental import FUNDAMENTAL_FEATURES
from portfolio_core.ml.walk_forward import DEFAULT_FEATURES, save_signal_model, train_once, walk_forward_evaluate
from portfolio_core.quant.factors import composite_quant_score


def main() -> None:
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    fundamental_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    technical_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    feature_path = fundamental_path if fundamental_path.exists() else technical_path
    if not feature_path.exists():
        raise SystemExit("No feature panel. Run B3 ingestion for multiple years first.")
    frame = pl.read_parquet(feature_path).filter(pl.col("target_excess_return").is_not_null())
    feature_names = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in frame.columns]
    print(f"feature_panel={feature_path} features={len(feature_names)}")
    years = sorted(set(frame["trade_date"].dt.year().to_list()))
    if len(years) < 4:
        raise SystemExit("At least four calendar years are recommended for a meaningful prototype training run.")

    first_validation = years[max(2, len(years) // 2)]
    evaluation = walk_forward_evaluate(
        frame, first_validation_year=first_validation, feature_names=feature_names
    )
    print("walk_forward:")
    for year, metrics in evaluation:
        print(year, metrics)

    latest_year = years[-1]
    train = frame.filter(pl.col("trade_date").dt.year() < latest_year)
    valid = frame.filter(pl.col("trade_date").dt.year() == latest_year)
    tuned_path = model_dir / "tuned_hyperparameters.json"
    model_config = None
    tuning_metadata = None
    if tuned_path.exists():
        tuning_metadata = json.loads(tuned_path.read_text(encoding="utf-8"))
        if int(tuning_metadata.get("untouched_holdout_year", -1)) == latest_year:
            model_config = tuning_metadata.get("config")
            print(f"using_tuned_config={tuned_path}")
        else:
            print("tuned_config_ignored=holdout_year_mismatch")
    model, metrics = train_once(
        train, valid, feature_names=feature_names, seed=42, model_config=model_config
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "signal_model.joblib"
    save_signal_model(model, str(model_path))
    metadata = {
        "trained_until": str(model.trained_until),
        "validation_year": latest_year,
        "rank_ic": metrics.rank_ic,
        "top_minus_bottom": metrics.top_minus_bottom,
        "hit_rate": metrics.hit_rate,
        "features": feature_names,
        "feature_panel": str(feature_path),
        "tuned_config_used": model_config is not None,
        "tuning": tuning_metadata if model_config is not None else None,
        "research_warning": "Model promotion requires repeated OOS evidence; this file is not a profit guarantee.",
    }
    (model_dir / "signal_model.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved={model_path}")

    # Optional: publish a latest inference snapshot into PostgreSQL for the web UI.
    try:
        sys.path.insert(0, "/workspace/services/api")
        from app.db import Base, SessionLocal, engine
        from app.models import AssetSnapshot
        from sqlalchemy import delete

        latest_date = frame["trade_date"].max()
        latest = composite_quant_score(frame.filter(pl.col("trade_date") == latest_date))
        predictions = model.predict(latest)
        joined = latest.join(predictions.select("ticker", "predicted_excess_return", "prediction_uncertainty"), on="ticker")
        max_abs = max(float(joined["predicted_excess_return"].abs().max() or 1.0), 1e-6)
        Base.metadata.create_all(engine)
        with SessionLocal() as db:
            as_of = str(latest_date)
            db.execute(delete(AssetSnapshot).where(AssetSnapshot.as_of == as_of))
            for row in joined.iter_rows(named=True):
                ml_score = 50.0 + 45.0 * float(row["predicted_excess_return"]) / max_abs
                db.add(AssetSnapshot(
                    as_of=as_of,
                    ticker=str(row["ticker"]), company=str(row["ticker"]),
                    sector=str(row.get("sector") or "Unknown"),
                    price=float(row["close"]),
                    predicted_excess_return=float(row["predicted_excess_return"]),
                    prediction_uncertainty=float(row["prediction_uncertainty"]),
                    volatility_annual=float(row.get("volatility_63d") or 0.02) * (252.0 ** 0.5),
                    ml_score=max(0.0, min(100.0, ml_score)),
                    quant_score=float(row.get("quant_score") or 0.5) * 100.0,
                    liquidity_score=float(row.get("rank_log_volume_21d") or 0.5) * 100.0,
                    explanation=(
                        "Generated from B3 history with point-in-time CVM fundamentals."
                        if "net_margin" in feature_names
                        else "Generated from B3 history; add a reviewed CVM issuer bridge to enable fundamentals."
                    ),
                ))
            db.commit()
        print(f"published_db_snapshot={latest_date}")
    except Exception as exc:
        print(f"db_publish_skipped={exc}")


if __name__ == "__main__":
    main()
