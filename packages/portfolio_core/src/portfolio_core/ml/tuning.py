from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import optuna
import polars as pl

from .walk_forward import DEFAULT_FEATURES, TrainingPolicy, train_once


@dataclass(frozen=True)
class TuningResult:
    best_score: float
    best_config: dict
    validation_years: list[int]
    n_trials: int


def _suggest_config(trial: optuna.Trial) -> dict:
    lightgbm = {
        "n_estimators": trial.suggest_int("lgb_n_estimators", 300, 1000, step=100),
        "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.08, log=True),
        "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 63),
        "min_child_samples": trial.suggest_int("lgb_min_child_samples", 20, 120, step=10),
        "subsample": trial.suggest_float("lgb_subsample", 0.65, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("lgb_colsample", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-3, 5.0, log=True),
        "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-2, 10.0, log=True),
    }
    catboost = {
        "iterations": trial.suggest_int("cat_iterations", 300, 1000, step=100),
        "learning_rate": trial.suggest_float("cat_learning_rate", 0.01, 0.08, log=True),
        "depth": trial.suggest_int("cat_depth", 4, 8),
        "l2_leaf_reg": trial.suggest_float("cat_l2_leaf_reg", 1.0, 20.0, log=True),
        "random_strength": trial.suggest_float("cat_random_strength", 0.0, 2.0),
    }
    raw_weights = np.asarray(
        [
            trial.suggest_float("weight_lightgbm", 0.1, 1.0),
            trial.suggest_float("weight_catboost", 0.1, 1.0),
            trial.suggest_float("weight_ridge", 0.02, 0.4),
        ],
        dtype=float,
    )
    weights = tuple((raw_weights / raw_weights.sum()).tolist())
    return {
        "lightgbm": lightgbm,
        "catboost": catboost,
        "ridge_alpha": trial.suggest_float("ridge_alpha", 1.0, 100.0, log=True),
        "weights": weights,
    }


def tune_ensemble(
    frame: pl.DataFrame,
    *,
    feature_names: list[str] | None = None,
    n_trials: int = 25,
    n_validation_years: int = 3,
    seed: int = 42,
    training_policy: TrainingPolicy | None = None,
) -> TuningResult:
    """Tune only on supplied development data using purged time-ordered folds."""
    features = feature_names or DEFAULT_FEATURES
    policy = training_policy or TrainingPolicy()
    years = sorted(set(frame["trade_date"].dt.year().to_list()))
    if len(years) < 4:
        raise ValueError("At least four development years are required before tuning")
    validation_years = years[-min(n_validation_years, len(years) - 2) :]

    def objective(trial: optuna.Trial) -> float:
        config = _suggest_config(trial)
        trial.set_user_attr("config", config)
        scores: list[float] = []
        for year in validation_years:
            train = frame.filter(pl.col("trade_date").dt.year() < year)
            valid = frame.filter(pl.col("trade_date").dt.year() == year)
            if train.height < 500 or valid.height < 100:
                continue
            _, metrics = train_once(
                train,
                valid,
                feature_names=features,
                seed=seed + year,
                model_config=config,
                training_policy=policy,
            )
            if np.isfinite(metrics.rank_ic):
                scores.append(metrics.rank_ic)
        if not scores:
            raise optuna.TrialPruned("No eligible purged development folds")
        score = float(np.mean(scores))
        trial.set_user_attr("fold_rank_ic", scores)
        return score

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return TuningResult(
        best_score=float(study.best_value),
        best_config=study.best_trial.user_attrs["config"],
        validation_years=validation_years,
        n_trials=n_trials,
    )
