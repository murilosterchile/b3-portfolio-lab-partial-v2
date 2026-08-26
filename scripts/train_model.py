from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import polars as pl

from portfolio_core.data_quality import filter_labels_known_by
from portfolio_core.features.fundamental import FUNDAMENTAL_FEATURES
from portfolio_core.ml.protocol import ResearchProtocol
from portfolio_core.ml.walk_forward import (
    DEFAULT_FEATURES,
    TrainingPolicy,
    fit_final_model,
    save_signal_model,
    walk_forward_evaluate,
)
from portfolio_core.quant.factors import composite_quant_score, learn_factor_sleeve_weights


def main() -> None:
    protocol = ResearchProtocol()
    data_dir = Path(os.getenv("DATA_DIR", "data"))
    model_dir = Path(os.getenv("MODEL_DIR", "models"))
    fundamental_path = data_dir / "gold" / "features" / "monthly_features_with_fundamentals.parquet"
    technical_path = data_dir / "gold" / "features" / "monthly_features.parquet"
    feature_path = fundamental_path if fundamental_path.exists() else technical_path
    if not feature_path.exists():
        raise SystemExit("No feature panel. Run the research data refresh first.")
    frame = pl.read_parquet(feature_path)
    if "target_end_date" not in frame.columns:
        raise SystemExit("Feature panel predates target_end_date. Rebuild features before training.")
    feature_names = DEFAULT_FEATURES + [name for name in FUNDAMENTAL_FEATURES if name in frame.columns]
    development = filter_labels_known_by(
        frame, knowledge_cutoff=protocol.development_knowledge_cutoff
    )
    print(f"feature_panel={feature_path} features={len(feature_names)}")
    print(f"development_end_year={protocol.development_end_year}")
    print(protocol.warning)
    tuned_path = model_dir / "tuned_hyperparameters.json"
    model_config = None
    tuning_metadata = None
    if tuned_path.exists():
        tuning_metadata = json.loads(tuned_path.read_text(encoding="utf-8"))
        if (
            int(tuning_metadata.get("development_end_year", -1)) == protocol.development_end_year
            and int(tuning_metadata.get("diagnostic_year", -1)) == protocol.diagnostic_year
            and int(tuning_metadata.get("target_horizon_bars", -1)) == protocol.label_horizon_bars
        ):
            model_config = tuning_metadata.get("config")
            print(f"using_tuned_config={tuned_path}")
        else:
            print("tuned_config_ignored=research_protocol_mismatch")

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
        raise SystemExit("Selected training policy does not match the registered research protocol")
    policy = TrainingPolicy(**selected_policy["training_policy"])
    factor_development = development
    if policy.window_years is not None:
        factor_development = factor_development.filter(
            pl.col("trade_date").dt.year()
            >= protocol.development_end_year - policy.window_years + 1
        )
    factor_sleeve_weights = learn_factor_sleeve_weights(factor_development)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "factor_sleeve_challenger.json").write_text(
        json.dumps(
            {
                "weights": factor_sleeve_weights,
                "shrinkage_to_equal": 0.80,
                "turnover_penalty": 0.10,
                "knowledge_cutoff": str(protocol.development_knowledge_cutoff),
                "diagnostic_year_excluded": protocol.diagnostic_year,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    evaluation = walk_forward_evaluate(
        development,
        first_validation_year=2019,
        last_validation_year=protocol.development_end_year,
        feature_names=feature_names,
        model_config=model_config,
        training_policy=policy,
    )
    print("development_walk_forward:")
    for year, metrics in evaluation:
        print(year, metrics)

    model = fit_final_model(
        development,
        knowledge_cutoff=protocol.development_knowledge_cutoff,
        feature_names=feature_names,
        seed=42,
        model_config=model_config,
        training_policy=policy,
    )
    if model.target_end_max is None or model.target_end_max > protocol.development_knowledge_cutoff:
        raise SystemExit("Refusing to save model: a training label extends beyond knowledge cutoff")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "signal_model.joblib"
    save_signal_model(model, str(model_path))
    metadata = {
        "trained_until_feature_date": str(model.trained_until),
        "training_target_end_max": str(model.target_end_max),
        "knowledge_cutoff": str(model.knowledge_cutoff),
        "development_end_year": protocol.development_end_year,
        "diagnostic_year": protocol.diagnostic_year,
        "features": model.feature_names,
        "feature_panel": str(feature_path),
        "disabled_features": list(model.disabled_feature_names),
        "training_policy": {
            "window_years": policy.window_years,
            "half_life_years": policy.half_life_years,
            "target_column": policy.target_column,
            "pre_validation_embargo_days": policy.pre_validation_embargo_days,
        },
        "tuned_config_used": model_config is not None,
        "selected_policy_metadata": selected_policy,
        "effective_model_config": model.model.effective_config(),
        "feature_coverage_by_year": model.feature_coverage_by_year,
        "disagreement_calibration": {
            "enabled": model.disagreement_calibration.enabled,
            "reason": model.disagreement_calibration.reason,
            "rank_correlation": model.disagreement_calibration.rank_correlation,
            "bins": list(model.disagreement_calibration.bins),
        },
        "tuning": tuning_metadata if model_config is not None else None,
        "warning": protocol.warning,
        "research_warning": "Promotion requires development-fold evidence and cannot use diagnostic-year results.",
    }
    (model_dir / "signal_model.metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"saved={model_path}")
    print(
        f"trained_until_feature_date={model.trained_until} "
        f"target_end_max={model.target_end_max} knowledge_cutoff={model.knowledge_cutoff}"
    )

    # Publishing a current snapshot is operational only. It does not feed back
    # into model selection or acceptance gates.
    try:
        sys.path.insert(0, "/workspace/services/api")
        from app.db import Base, SessionLocal, engine
        from app.models import AssetSnapshot
        from sqlalchemy import delete

        latest_date = frame["trade_date"].max()
        latest = composite_quant_score(frame.filter(pl.col("trade_date") == latest_date))
        predictions = model.predict(latest)
        joined = latest.join(
            predictions.select("ticker", "predicted_excess_return", "prediction_uncertainty"),
            on="ticker",
        )
        max_abs = max(float(joined["predicted_excess_return"].abs().max() or 1.0), 1e-6)
        Base.metadata.create_all(engine)
        with SessionLocal() as db:
            as_of = str(latest_date)
            db.execute(delete(AssetSnapshot).where(AssetSnapshot.as_of == as_of))
            for row in joined.iter_rows(named=True):
                ml_score = 50.0 + 45.0 * float(row["predicted_excess_return"]) / max_abs
                db.add(
                    AssetSnapshot(
                        as_of=as_of,
                        ticker=str(row["ticker"]),
                        issuer_id=str(
                            row.get("CD_CVM")
                            or row.get("issuer_identifier")
                            or str(row["ticker"])[:4]
                        ),
                        company=str(row["ticker"]),
                        sector=str(row.get("sector") or "Unknown"),
                        price=float(row["close"]),
                        predicted_excess_return=float(row["predicted_excess_return"]),
                        prediction_uncertainty=float(row["prediction_uncertainty"]),
                        volatility_annual=float(row.get("volatility_63d") or 0.02) * (252.0 ** 0.5),
                        ml_score=max(0.0, min(100.0, ml_score)),
                        quant_score=float(row.get("quant_score") or 0.5) * 100.0,
                        liquidity_score=float(row.get("rank_log_volume_21d") or 0.5) * 100.0,
                        explanation="Leakage-safe model; current snapshot is not a model-selection observation.",
                    )
                )
            db.commit()
        print(f"published_db_snapshot={latest_date}")
    except Exception as exc:
        print(f"db_publish_skipped={exc}")


if __name__ == "__main__":
    main()
