from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import joblib
import polars as pl
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from portfolio_core.data_quality import DataQualityError, validate_temporal_split

from .evaluation import CrossSectionalMetrics, panel_cross_sectional_metrics
from .model import EnsembleRegressor


DEFAULT_FEATURES = [
    "return_5d",
    "return_21d",
    "return_63d",
    "return_126d",
    "return_252d",
    "momentum_12_1",
    "volatility_21d",
    "volatility_63d",
    "distance_sma_21",
    "distance_sma_63",
    "distance_sma_252",
    "log_volume_21d",
    "log_trades_21d",
    "rank_return_21d",
    "rank_return_63d",
    "rank_return_126d",
    "rank_momentum_12_1",
    "rank_volatility_63d",
    "rank_distance_sma_63",
    "rank_log_volume_21d",
]


@dataclass
class TrainedSignalModel:
    preprocessing: Pipeline
    model: EnsembleRegressor
    feature_names: list[str]
    trained_until: date
    disabled_feature_names: tuple[str, ...] = ()
    training_feature_coverage: dict[str, float] = field(default_factory=dict)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        x = frame.select(self.feature_names).to_numpy()
        xp = self.preprocessing.transform(x)
        pred = self.model.predict(xp)
        unc = self.model.uncertainty(xp)
        return frame.select(["trade_date", "ticker", "close"]).with_columns(
            pl.Series("predicted_excess_return", pred),
            pl.Series("prediction_uncertainty", unc),
        ).with_columns(
            pl.col("predicted_excess_return")
            .rank(method="average", descending=True)
            .over("trade_date")
            .alias("predicted_rank")
        )


def _preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("non_degenerate", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
        ]
    )


def select_available_features(
    train: pl.DataFrame,
    requested_features: list[str],
    *,
    minimum_coverage: float = 0.01,
) -> tuple[list[str], list[str], dict[str, float]]:
    """Choose fold features from training data only and record the decision."""
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in [0, 1]")
    missing_columns = set(requested_features) - set(train.columns)
    if missing_columns:
        raise DataQualityError(f"training frame missing features: {sorted(missing_columns)}")
    if train.is_empty():
        raise DataQualityError("cannot select features from an empty training frame")
    coverage = {
        name: float(
            train.select(
                (pl.col(name).is_not_null() & pl.col(name).is_finite()).mean()
            ).item()
        )
        for name in requested_features
    }
    enabled = [name for name in requested_features if coverage[name] >= minimum_coverage]
    disabled = [name for name in requested_features if name not in enabled]
    if not enabled:
        raise DataQualityError(
            f"all requested features are below minimum coverage {minimum_coverage:.1%}"
        )
    return enabled, disabled, coverage


def train_once(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    *,
    feature_names: list[str] | None = None,
    seed: int = 42,
    model_config: dict | None = None,
    minimum_feature_coverage: float = 0.01,
) -> tuple[TrainedSignalModel, CrossSectionalMetrics]:
    requested_features = list(feature_names or DEFAULT_FEATURES)
    train = train.filter(pl.col("target_excess_return").is_finite())
    valid = valid.filter(pl.col("target_excess_return").is_finite())
    validate_temporal_split(train, valid)
    features, disabled, coverage = select_available_features(
        train, requested_features, minimum_coverage=minimum_feature_coverage
    )
    preprocessing = _preprocessor()
    x_train = preprocessing.fit_transform(train.select(features).to_numpy())
    y_train = train["target_excess_return"].to_numpy()
    config = model_config or {}
    model = EnsembleRegressor.create_default(
        random_seed=seed,
        lightgbm_overrides=config.get("lightgbm"),
        catboost_overrides=config.get("catboost"),
        ridge_alpha=float(config.get("ridge_alpha", 10.0)),
        weights=tuple(config.get("weights", (0.45, 0.45, 0.10))),
    ).fit(x_train, y_train)
    x_valid = preprocessing.transform(valid.select(features).to_numpy())
    pred = model.predict(x_valid)
    metrics = panel_cross_sectional_metrics(valid, pred)
    trained_until = train["trade_date"].max()
    if trained_until is None:
        raise ValueError("Empty training data")
    return (
        TrainedSignalModel(
            preprocessing,
            model,
            features,
            trained_until,
            tuple(disabled),
            coverage,
        ),
        metrics,
    )


def walk_forward_evaluate(
    frame: pl.DataFrame,
    *,
    first_validation_year: int,
    feature_names: list[str] | None = None,
    model_config: dict | None = None,
) -> list[tuple[int, CrossSectionalMetrics]]:
    results: list[tuple[int, CrossSectionalMetrics]] = []
    max_year = int(frame["trade_date"].dt.year().max())
    for year in range(first_validation_year, max_year + 1):
        train = frame.filter(pl.col("trade_date").dt.year() < year)
        valid = frame.filter(pl.col("trade_date").dt.year() == year)
        if train.height < 500 or valid.height < 100:
            continue
        _, metrics = train_once(
            train, valid, feature_names=feature_names, seed=year, model_config=model_config
        )
        results.append((year, metrics))
    return results


def save_signal_model(model: TrainedSignalModel, path: str) -> None:
    joblib.dump(model, path, compress=3)


def load_signal_model(path: str) -> TrainedSignalModel:
    return joblib.load(path)
