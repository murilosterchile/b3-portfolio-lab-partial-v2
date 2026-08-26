from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import log

import joblib
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from portfolio_core.data_quality import (
    DataQualityError,
    filter_labels_known_by,
    purge_overlapping_labels,
    validate_temporal_split,
)

from .evaluation import CrossSectionalMetrics, panel_cross_sectional_metrics
from .model import EnsembleRegressor, LearningToRankModel


DEFAULT_FEATURES = [
    "return_5d", "return_21d", "return_63d", "return_126d", "return_252d",
    "momentum_12_1", "volatility_21d", "volatility_63d", "distance_sma_21",
    "distance_sma_63", "distance_sma_252", "log_volume_21d", "log_trades_21d",
    "rank_return_21d", "rank_return_63d", "rank_return_126d", "rank_momentum_12_1",
    "rank_volatility_63d", "rank_distance_sma_63", "rank_log_volume_21d",
    "short_term_reversal_5d", "medium_term_momentum", "residual_volatility_63d",
    "log_traded_value_21d", "amihud_illiquidity_21d",
    "rank_short_term_reversal_5d", "rank_medium_term_momentum",
    "rank_residual_volatility_63d", "rank_log_traded_value_21d",
    "rank_amihud_illiquidity_21d",
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
    linear_preprocessing: Pipeline
    model: EnsembleRegressor
    feature_names: list[str]
    trained_until: date
    disabled_feature_names: tuple[str, ...] = ()
    training_feature_coverage: dict[str, float] = field(default_factory=dict)
    feature_coverage_by_year: dict[int, dict[str, float]] = field(default_factory=dict)
    disagreement_calibration: "DisagreementCalibration" = field(
        default_factory=lambda: DisagreementCalibration.disabled("not_fitted")
    )
    knowledge_cutoff: date | None = None
    target_end_max: date | None = None
    target_column: str = "target_excess_return"
    training_policy: TrainingPolicy = field(default_factory=TrainingPolicy)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise DataQualityError(f"prediction frame is missing model features: {missing}")
        x = frame.select(self.feature_names).to_numpy()
        x_tree = _sanitize_tree_matrix(x)
        x_linear = self.linear_preprocessing.transform(x)
        pred = self.model.predict(x_tree, x_linear)
        disagreement = self.model.ensemble_disagreement(x_tree, x_linear)
        calibrated = self.disagreement_calibration.predict(disagreement)
        return frame.select(["trade_date", "ticker", "close"]).with_columns(
            pl.Series("predicted_excess_return", pred),
            pl.Series("ensemble_disagreement", disagreement),
            pl.Series("calibrated_uncertainty", calibrated),
            # Compatibility at API boundaries. This is calibrated expected absolute
            # error, or zero when development calibration fails its monotonicity gate.
            pl.Series("prediction_uncertainty", calibrated),
        ).with_columns(
            pl.col("predicted_excess_return")
            .rank(method="average", descending=True)
            .over("trade_date")
            .alias("predicted_rank")
        )


    @property
    def preprocessing(self) -> Pipeline:
        """Compatibility alias for consumers inspecting the Ridge branch."""
        return self.linear_preprocessing


@dataclass
class DisagreementCalibration:
    model: IsotonicRegression | None
    enabled: bool
    reason: str
    rank_correlation: float | None = None
    bins: tuple[dict[str, float], ...] = ()

    @classmethod
    def disabled(cls, reason: str) -> "DisagreementCalibration":
        return cls(model=None, enabled=False, reason=reason)

    def predict(self, disagreement: np.ndarray) -> np.ndarray:
        values = np.asarray(disagreement, dtype=float)
        if not self.enabled or self.model is None:
            return np.zeros_like(values)
        return np.maximum(np.asarray(self.model.predict(values), dtype=float), 0.0)


@dataclass
class TrainedRankingSignalModel:
    model: LearningToRankModel
    feature_names: list[str]
    trained_until: date
    disabled_feature_names: tuple[str, ...] = ()
    training_feature_coverage: dict[str, float] = field(default_factory=dict)

    def predict(self, frame: pl.DataFrame) -> pl.DataFrame:
        missing = sorted(set(self.feature_names) - set(frame.columns))
        if missing:
            raise DataQualityError(f"prediction frame is missing ranker features: {missing}")
        score = self.model.predict(
            _sanitize_tree_matrix(frame.select(self.feature_names).to_numpy())
        )
        return frame.select(["trade_date", "ticker", "close"]).with_columns(
            pl.Series("predicted_excess_return", score),
            pl.lit(0.0).alias("ensemble_disagreement"),
            pl.lit(0.0).alias("calibrated_uncertainty"),
            pl.lit(0.0).alias("prediction_uncertainty"),
        ).with_columns(
            pl.col("predicted_excess_return")
            .rank(method="average", descending=True)
            .over("trade_date")
            .alias("predicted_rank")
        )


def _linear_preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("non_degenerate", VarianceThreshold(threshold=0.0)),
            ("scale", StandardScaler()),
        ]
    )


def _preprocessor() -> Pipeline:
    """Backward-compatible name for the Ridge-only preprocessing branch."""
    return _linear_preprocessor()


def _sanitize_tree_matrix(values: np.ndarray) -> np.ndarray:
    """Preserve native missing values for trees while eliminating +/- infinity."""
    clean = np.asarray(values, dtype=float).copy()
    clean[~np.isfinite(clean)] = np.nan
    return clean


def _coverage_by_year(train: pl.DataFrame, features: list[str]) -> dict[int, dict[str, float]]:
    output: dict[int, dict[str, float]] = {}
    for year in sorted(set(train["trade_date"].dt.year().to_list())):
        subset = train.filter(pl.col("trade_date").dt.year() == year)
        output[int(year)] = {
            name: float(
                subset.select(
                    (pl.col(name).is_not_null() & pl.col(name).is_finite()).mean()
                ).item()
            )
            for name in features
        }
    return output


def select_available_features(
    train: pl.DataFrame,
    requested_features: list[str],
    *,
    minimum_coverage: float = 0.60,
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
    robust_date_fraction: dict[str, float] = {}
    for name in requested_features:
        required = max(minimum_coverage, 0.80) if name in DEFAULT_FEATURES else minimum_coverage
        if "trade_date" not in train.columns:
            robust_date_fraction[name] = 1.0
        else:
            per_date = train.group_by("trade_date").agg(
                (pl.col(name).is_not_null() & pl.col(name).is_finite()).mean().alias("coverage")
            )
            robust_date_fraction[name] = float(
                per_date.select((pl.col("coverage") >= required).mean()).item()
            )
    enabled = [
        name
        for name in requested_features
        if coverage[name] >= (max(minimum_coverage, 0.80) if name in DEFAULT_FEATURES else minimum_coverage)
        and robust_date_fraction[name] >= (0.80 if name in DEFAULT_FEATURES else 0.60)
    ]
    disabled = [name for name in requested_features if name not in enabled]
    if not enabled:
        raise DataQualityError(
            f"all requested features are below minimum coverage {minimum_coverage:.1%}"
        )
    return enabled, disabled, coverage


def cross_section_sample_weights(
    frame: pl.DataFrame,
    *,
    reference_date: date,
    half_life_years: float | None,
) -> np.ndarray:
    """Give each date equal mass, then optionally decay it and split issuer mass."""
    if frame.is_empty():
        raise DataQualityError("cannot weight an empty training frame")
    issuer_column = next(
        (name for name in ("CD_CVM", "issuer_id", "issuer_identifier") if name in frame.columns),
        None,
    )
    if issuer_column is None:
        counts = frame.select(pl.len().over("trade_date").alias("count"))["count"].to_numpy()
        base = 1.0 / counts
    else:
        issuer = frame.select("trade_date", issuer_column).with_row_index("__row").with_columns(
            pl.when(pl.col(issuer_column).is_null())
            .then(pl.lit("__ticker_row__") + pl.col("__row").cast(pl.Utf8))
            .otherwise(pl.col(issuer_column).cast(pl.Utf8))
            .alias("__issuer")
        ).with_columns(
            pl.col("__issuer").n_unique().over("trade_date").alias("__issuer_count"),
            pl.len().over(["trade_date", "__issuer"]).alias("__class_count"),
        )
        base = 1.0 / (
            issuer["__issuer_count"].to_numpy() * issuer["__class_count"].to_numpy()
        )
    if half_life_years is not None:
        recency = exponential_recency_weights(
            frame["trade_date"].to_list(),
            reference_date=reference_date,
            half_life_years=half_life_years,
        )
        base = base * recency
    mean = float(np.mean(base))
    return base / mean if mean > 0 else np.ones(frame.height, dtype=float)


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


def _fit_disagreement_calibration(
    train: pl.DataFrame,
    *,
    features: list[str],
    target_column: str,
    seed: int,
    model_config: dict,
    policy: TrainingPolicy,
    reference_date: date,
) -> DisagreementCalibration:
    """Time-aware inner OOF calibration from disagreement to absolute error."""
    dates = sorted(set(train["trade_date"].to_list()))
    if len(dates) < 24:
        return DisagreementCalibration.disabled("insufficient_inner_dates")
    split = max(1, int(len(dates) * 0.80))
    if split >= len(dates):
        return DisagreementCalibration.disabled("insufficient_calibration_dates")
    fit_dates = set(dates[:split])
    fit_frame = train.filter(pl.col("trade_date").is_in(fit_dates))
    calibration_frame = train.filter(~pl.col("trade_date").is_in(fit_dates))
    fit_frame = purge_overlapping_labels(
        fit_frame,
        calibration_frame,
        label_end_column=policy.label_end_column,
        pre_validation_embargo_days=policy.pre_validation_embargo_days,
    )
    if fit_frame.height < 100 or calibration_frame.height < 50:
        return DisagreementCalibration.disabled("insufficient_calibration_rows")

    raw_fit = fit_frame.select(features).to_numpy()
    raw_calibration = calibration_frame.select(features).to_numpy()
    preprocessing = _linear_preprocessor()
    x_fit_linear = preprocessing.fit_transform(raw_fit)
    x_calibration_linear = preprocessing.transform(raw_calibration)
    temporary = EnsembleRegressor.create_default(
        random_seed=seed + 10_000,
        lightgbm_overrides=model_config.get("lightgbm"),
        catboost_overrides=model_config.get("catboost"),
        ridge_alpha=float(model_config.get("ridge_alpha", 10.0)),
        weights=tuple(model_config.get("weights", (0.45, 0.45, 0.10))),
    )
    weights = cross_section_sample_weights(
        fit_frame,
        reference_date=reference_date,
        half_life_years=policy.half_life_years,
    )
    temporary.fit(
        _sanitize_tree_matrix(raw_fit),
        x_fit_linear,
        fit_frame[target_column].to_numpy(),
        sample_weight=weights,
    )
    x_calibration_tree = _sanitize_tree_matrix(raw_calibration)
    prediction = temporary.predict(x_calibration_tree, x_calibration_linear)
    disagreement = temporary.ensemble_disagreement(
        x_calibration_tree, x_calibration_linear
    )
    error = np.abs(calibration_frame[target_column].to_numpy() - prediction)
    mask = np.isfinite(disagreement) & np.isfinite(error)
    disagreement, error = disagreement[mask], error[mask]
    if len(disagreement) < 50 or np.unique(disagreement).size < 5:
        return DisagreementCalibration.disabled("insufficient_calibration_variation")
    statistic = spearmanr(disagreement, error).statistic
    correlation = float(statistic) if statistic is not None else float("nan")
    quantiles = np.unique(np.quantile(disagreement, np.linspace(0.0, 1.0, 6)))
    bins: list[dict[str, float]] = []
    if len(quantiles) >= 3:
        quantiles[0], quantiles[-1] = -np.inf, np.inf
        for lower, upper in zip(quantiles[:-1], quantiles[1:], strict=True):
            selected = (disagreement > lower) & (disagreement <= upper)
            if np.any(selected):
                bins.append(
                    {
                        "disagreement_mean": float(np.mean(disagreement[selected])),
                        "absolute_error_mean": float(np.mean(error[selected])),
                        "count": float(np.sum(selected)),
                    }
                )
    increasing = len(bins) >= 3 and all(
        right["absolute_error_mean"] >= left["absolute_error_mean"]
        for left, right in zip(bins[:-1], bins[1:])
    )
    if not np.isfinite(correlation) or correlation <= 0.0 or not increasing:
        return DisagreementCalibration(
            model=None,
            enabled=False,
            reason="oof_error_not_monotonic",
            rank_correlation=correlation if np.isfinite(correlation) else None,
            bins=tuple(bins),
        )
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0)
    isotonic.fit(disagreement, error)
    return DisagreementCalibration(
        model=isotonic,
        enabled=True,
        reason="oof_monotonic_calibration_passed",
        rank_correlation=correlation,
        bins=tuple(bins),
    )


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
    raw_train = train.select(features).to_numpy()
    x_tree = _sanitize_tree_matrix(raw_train)
    linear_preprocessing = _linear_preprocessor()
    x_linear = linear_preprocessing.fit_transform(raw_train)
    y_train = train[target_column].to_numpy()
    config = model_config or {}
    model = EnsembleRegressor.create_default(
        random_seed=seed,
        lightgbm_overrides=config.get("lightgbm"),
        catboost_overrides=config.get("catboost"),
        ridge_alpha=float(config.get("ridge_alpha", 10.0)),
        weights=tuple(config.get("weights", (0.45, 0.45, 0.10))),
    )
    reference = knowledge_cutoff or train["trade_date"].max()
    if reference is None:
        raise DataQualityError("training reference date is unavailable")
    sample_weight = cross_section_sample_weights(
        train,
        reference_date=reference,
        half_life_years=policy.half_life_years,
    )
    calibration = _fit_disagreement_calibration(
        train,
        features=features,
        target_column=target_column,
        seed=seed,
        model_config=config,
        policy=policy,
        reference_date=reference,
    )
    model.fit(x_tree, x_linear, y_train, sample_weight=sample_weight)
    trained_until = train["trade_date"].max()
    target_end_max = train[policy.label_end_column].max()
    if trained_until is None or target_end_max is None:
        raise DataQualityError("training dates are unavailable")
    return TrainedSignalModel(
        linear_preprocessing=linear_preprocessing,
        model=model,
        feature_names=features,
        trained_until=trained_until,
        disabled_feature_names=tuple(disabled),
        training_feature_coverage=coverage,
        feature_coverage_by_year=_coverage_by_year(train, requested_features),
        disagreement_calibration=calibration,
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
    minimum_feature_coverage: float = 0.60,
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
    pred = model.predict(valid)["predicted_excess_return"].to_numpy()
    metrics = panel_cross_sectional_metrics(
        valid, pred, target_column=policy.target_column
    )
    return model, metrics


def _cross_sectional_relevance(frame: pl.DataFrame, target_column: str) -> np.ndarray:
    """Five ordinal relevance levels using only each training cross-section."""
    ranked = frame.with_columns(
        (
            pl.col(target_column).rank(method="average").over("trade_date")
            / pl.len().over("trade_date")
        ).alias("__percentile")
    )
    return (
        ranked["__percentile"].to_numpy() * 5.0
    ).astype(int).clip(0, 4)


def train_ranker_once(
    train: pl.DataFrame,
    valid: pl.DataFrame,
    *,
    feature_names: list[str] | None = None,
    seed: int = 42,
    model_config: dict | None = None,
    minimum_feature_coverage: float = 0.60,
    training_policy: TrainingPolicy | None = None,
) -> tuple[TrainedRankingSignalModel, CrossSectionalMetrics]:
    """Fit the LightGBM ranking challenger on purged development data only."""
    policy = training_policy or TrainingPolicy()
    requested = list(feature_names or DEFAULT_FEATURES)
    valid = valid.filter(
        pl.col(policy.target_column).is_not_null() & pl.col(policy.target_column).is_finite()
    )
    prepared = _prepare_train_for_validation(train, valid, policy=policy).sort(
        ["trade_date", "ticker"]
    )
    features, disabled, coverage = select_available_features(
        prepared, requested, minimum_coverage=minimum_feature_coverage
    )
    groups = prepared.group_by("trade_date", maintain_order=True).len()["len"].to_list()
    reference = valid["trade_date"].min()
    if reference is None:
        raise DataQualityError("ranking validation start is unavailable")
    weights = cross_section_sample_weights(
        prepared,
        reference_date=reference,
        half_life_years=policy.half_life_years,
    )
    ranker = LearningToRankModel.create_default(
        random_seed=seed,
        overrides=(model_config or {}).get("ranker"),
    ).fit(
        _sanitize_tree_matrix(prepared.select(features).to_numpy()),
        _cross_sectional_relevance(prepared, policy.target_column),
        group=[int(value) for value in groups],
        sample_weight=weights,
    )
    trained_until = prepared["trade_date"].max()
    if trained_until is None:
        raise DataQualityError("ranking training dates are unavailable")
    artifact = TrainedRankingSignalModel(
        model=ranker,
        feature_names=features,
        trained_until=trained_until,
        disabled_feature_names=tuple(disabled),
        training_feature_coverage=coverage,
    )
    prediction = artifact.predict(valid)["predicted_excess_return"].to_numpy()
    return artifact, panel_cross_sectional_metrics(
        valid, prediction, target_column=policy.target_column
    )


def fit_final_model(
    frame: pl.DataFrame,
    *,
    knowledge_cutoff: date,
    feature_names: list[str] | None = None,
    seed: int = 42,
    model_config: dict | None = None,
    minimum_feature_coverage: float = 0.60,
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
    if not hasattr(model, "linear_preprocessing") or not hasattr(
        model, "disagreement_calibration"
    ):
        raise DataQualityError(
            "legacy signal artifact uses shared preprocessing or uncalibrated disagreement; retrain it"
        )
    return model
