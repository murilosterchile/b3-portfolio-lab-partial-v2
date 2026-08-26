from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import log

import joblib
import numpy as np
import polars as pl
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from portfolio_core.data_quality import (
    DataQualityError,
    filter_labels_known_by,
    purge_overlapping_labels,
    validate_temporal_split,
)

from .evaluation import CrossSectionalMetrics, panel_cross_sectional_metrics
from .model import EnsembleRegressor


DEFAULT_FEATURES = [
    "return_5d", "return_21d", "return_63d", "return_126d", "return_252d",
    "momentum_12_1", "volatility_21d", "volatility_63d", "distance_sma_21",
    "distance_sma_63", "distance_sma_252", "log_volume_21d", "log_trades_21d",
    "rank_return_21d", "rank_return_63d", "rank_return_126d", "rank_momentum_12_1",
    "rank_volatility_63d", "rank_distance_sma_63", "rank_log_volume_21d",
]


@dataclass(frozen=True)
class TrainingPolicy:
    """Training-history policy selected only on development folds."""

    window_years: int | None = None
    half_life_years: float | None = None
    target_column: str = "target_excess_return"
    label_end_column: str = "target_end_date"
    pre_validation_embargo_days: int = 0

    def __post_init__(self) -> None:
        if self.window_years is not None and self.window_years <= 0:
            raise ValueError("window_years must be positive or None for expanding")
        if self.half_life_years is not None and self.half_life_years <= 0:
            raise ValueError("half_life_years must be positive")
        if self.pre_validation_embargo_days < 0:
            raise ValueError("pre_validation_embargo_days must be non-negative")


@dataclass
class TrainedSignalModel:
    preprocessing: Pipeline
    model: EnsembleRegressor
    feature_names: list[str]
    trained_until: date
    disabled_feature_names: tuple[str, ...] = ()
    training_feature_coverage: dict[str, float] = field(default_factory=dict)
    knowledge_cutoff: date | None = None
    target_end_max: date | None = None
    target_column: str = "target_excess_return"
    training_policy: TrainingPolicy = field(default_factory=TrainingPolicy)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise DataQualityError(f"prediction frame is missing model features: {missing}")
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
    if not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("minimum_coverage must be in [0, 1]")
    missing_columns = set(requested_features) - set(train.columns)
    if missing_columns:
        raise DataQualityError(f"training frame missing features: {sorted(missing_columns)}")
    if train.is_empty():
        raise DataQualityError("cannot select features from an empty training frame")
    coverage = {
        name: float(
            train.select((pl.col(name).is_not_null() & pl.col(name).is_finite()).mean()).item()
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


def exponential_recency_weights(
    dates: list[date], *, reference_date: date, half_life_years: float
) -> np.ndarray:
    """Exponential sample weights with an interpretable half-life."""
    if half_life_years <= 0:
        raise ValueError("half_life_years must be positive")
    ages = np.asarray([max(0, (reference_date - value).days) / 365.25 for value in dates])
    weights = np.exp(-log(2.0) * ages / half_life_years)
    mean = float(weights.mean())
    return weights / mean if mean > 0 else np.ones_like(weights)


def _apply_window(train: pl.DataFrame, *, reference_date: date, policy: TrainingPolicy) -> pl.DataFrame:
    if policy.window_years is None:
        return train
    first_year = reference_date.year - policy.window_years
    return train.filter(pl.col("trade_date").dt.year() >= first_year)


def _prepare_train_for_validation(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    *,
    policy: TrainingPolicy,
) -> pl.DataFrame:
    target = policy.target_column
    if target not in train.columns or target not in valid.columns:
        raise DataQualityError(f"missing target column {target!r}")
    train = train.filter(pl.col(target).is_not_null() & pl.col(target).is_finite())
    valid = valid.filter(pl.col(target).is_not_null() & pl.col(target).is_finite())
    if valid.is_empty():
        raise DataQualityError("validation has no finite targets")
    train = purge_overlapping_labels(
        train,
        valid,
        label_end_column=policy.label_end_column,
        pre_validation_embargo_days=policy.pre_validation_embargo_days,
    )
    validation_start = valid["trade_date"].min()
    if validation_start is None:
        raise DataQualityError("validation start is unavailable")
    train = _apply_window(train, reference_date=validation_start, policy=policy)
    validate_temporal_split(
        train, valid, label_end_column=policy.label_end_column
    )
    return train


def _fit_model(
    train: pl.DataFrame,
    *,
    requested_features: list[str],
    target_column: str,
    seed: int,
    model_config: dict | None,
    minimum_feature_coverage: float,
    policy: TrainingPolicy,
    knowledge_cutoff: date | None,
) -> TrainedSignalModel:
    features, disabled, coverage = select_available_features(
        train, requested_features, minimum_coverage=minimum_feature_coverage
    )
    preprocessing = _preprocessor()
    x_train = preprocessing.fit_transform(train.select(features).to_numpy())
    y_train = train[target_column].to_numpy()
    config = model_config or {}
    model = EnsembleRegressor.create_default(
        random_seed=seed,
        lightgbm_overrides=config.get("lightgbm"),
        catboost_overrides=config.get("catboost"),
        ridge_alpha=float(config.get("ridge_alpha", 10.0)),
        weights=tuple(config.get("weights", (0.45, 0.45, 0.10))),
    )
    sample_weight = None
    if policy.half_life_years is not None:
        reference = knowledge_cutoff or train["trade_date"].max()
        if reference is None:
            raise DataQualityError("training reference date is unavailable")
        sample_weight = exponential_recency_weights(
            train["trade_date"].to_list(),
            reference_date=reference,
            half_life_years=policy.half_life_years,
        )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    trained_until = train["trade_date"].max()
    target_end_max = train[policy.label_end_column].max()
    if trained_until is None or target_end_max is None:
        raise DataQualityError("training dates are unavailable")
    return TrainedSignalModel(
        preprocessing=preprocessing,
        model=model,
        feature_names=features,
        trained_until=trained_until,
        disabled_feature_names=tuple(disabled),
        training_feature_coverage=coverage,
        knowledge_cutoff=knowledge_cutoff,
        target_end_max=target_end_max,
        target_column=target_column,
        training_policy=policy,
    )


def train_once(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    *,
    feature_names: list[str] | None = None,
    seed: int = 42,
    model_config: dict | None = None,
    minimum_feature_coverage: float = 0.01,
    training_policy: TrainingPolicy | None = None,
) -> tuple[TrainedSignalModel, CrossSectionalMetrics]:
    policy = training_policy or TrainingPolicy()
    requested_features = list(feature_names or DEFAULT_FEATURES)
    valid = valid.filter(
        pl.col(policy.target_column).is_not_null() & pl.col(policy.target_column).is_finite()
    )
    prepared = _prepare_train_for_validation(train, valid, policy=policy)
    validation_start = valid["trade_date"].min()
    if validation_start is None:
        raise DataQualityError("validation start is unavailable")
    model = _fit_model(
        prepared,
        requested_features=requested_features,
        target_column=policy.target_column,
        seed=seed,
        model_config=model_config,
        minimum_feature_coverage=minimum_feature_coverage,
        policy=policy,
        knowledge_cutoff=validation_start,
    )
    x_valid = model.preprocessing.transform(valid.select(model.feature_names).to_numpy())
    pred = model.model.predict(x_valid)
    metrics = panel_cross_sectional_metrics(
        valid, pred, target_column=policy.target_column
    )
    return model, metrics


def fit_final_model(
    frame: pl.DataFrame,
    *,
    knowledge_cutoff: date,
    feature_names: list[str] | None = None,
    seed: int = 42,
    model_config: dict | None = None,
    minimum_feature_coverage: float = 0.01,
    training_policy: TrainingPolicy | None = None,
) -> TrainedSignalModel:
    """Fit an artifact using only labels that were fully known at cutoff."""
    policy = training_policy or TrainingPolicy()
    requested_features = list(feature_names or DEFAULT_FEATURES)
    known = filter_labels_known_by(
        frame,
        knowledge_cutoff=knowledge_cutoff,
        label_end_column=policy.label_end_column,
    ).filter(
        pl.col(policy.target_column).is_not_null() & pl.col(policy.target_column).is_finite()
    )
    known = _apply_window(known, reference_date=knowledge_cutoff, policy=policy)
    return _fit_model(
        known,
        requested_features=requested_features,
        target_column=policy.target_column,
        seed=seed,
        model_config=model_config,
        minimum_feature_coverage=minimum_feature_coverage,
        policy=policy,
        knowledge_cutoff=knowledge_cutoff,
    )


def walk_forward_evaluate(
    frame: pl.DataFrame,
    *,
    first_validation_year: int,
    last_validation_year: int | None = None,
    feature_names: list[str] | None = None,
    model_config: dict | None = None,
    training_policy: TrainingPolicy | None = None,
) -> list[tuple[int, CrossSectionalMetrics]]:
    policy = training_policy or TrainingPolicy()
    results: list[tuple[int, CrossSectionalMetrics]] = []
    max_year = int(frame["trade_date"].dt.year().max())
    end_year = min(max_year, last_validation_year or max_year)
    for year in range(first_validation_year, end_year + 1):
        train = frame.filter(pl.col("trade_date").dt.year() < year)
        valid = frame.filter(pl.col("trade_date").dt.year() == year)
        if train.height < 500 or valid.height < 100:
            continue
        _, metrics = train_once(
            train,
            valid,
            feature_names=feature_names,
            seed=year,
            model_config=model_config,
            training_policy=policy,
        )
        results.append((year, metrics))
    return results


def save_signal_model(model: TrainedSignalModel, path: str) -> None:
    joblib.dump(model, path, compress=3)


def load_signal_model(path: str) -> TrainedSignalModel:
    model = joblib.load(path)
    if not hasattr(model, "knowledge_cutoff") or model.knowledge_cutoff is None:
        raise DataQualityError(
            "legacy signal artifact has no knowledge_cutoff; retrain it with the leakage-safe protocol"
        )
    return model
