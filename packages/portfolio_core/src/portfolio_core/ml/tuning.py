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
    outer_fold_scores: dict[int, float]
    outer_fold_configs: dict[int, dict]
    trial_history: list[dict[str, object]]
    number_of_experiments: int


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
    n_outer_years: int = 3,
    seed: int = 42,
    training_policy: TrainingPolicy | None = None,
) -> TuningResult:
    """Nested walk-forward tuning confined to supplied development data.

    Each outer year is evaluated exactly once with a configuration selected on
    earlier inner folds. A final inner study on all development data produces
    the deployable configuration; outer scores are reporting-only.
    """
    features = feature_names or DEFAULT_FEATURES
    policy = training_policy or TrainingPolicy()
    years = sorted(set(frame["trade_date"].dt.year().to_list()))
    if len(years) < 6:
        raise ValueError("At least six development years are required for nested tuning")
    if n_outer_years <= 0:
        raise ValueError("n_outer_years must be positive")
    outer_years = years[-min(n_outer_years, len(years) - 4) :]
    trial_history: list[dict[str, object]] = []

    def run_inner_study(inner_frame: pl.DataFrame, *, study_seed: int, scope: str) -> optuna.Study:
        inner_years = sorted(set(inner_frame["trade_date"].dt.year().to_list()))
        validation_years = inner_years[-min(n_validation_years, len(inner_years) - 2) :]

        def objective(trial: optuna.Trial) -> float:
            config = _suggest_config(trial)
            trial.set_user_attr("config", config)
            scores: list[float] = []
            for validation_year in validation_years:
                train = inner_frame.filter(pl.col("trade_date").dt.year() < validation_year)
                valid = inner_frame.filter(pl.col("trade_date").dt.year() == validation_year)
                if train.height < 500 or valid.height < 100:
                    continue
                _, metrics = train_once(
                    train,
                    valid,
                    feature_names=features,
                    seed=study_seed + validation_year,
                    model_config=config,
                    training_policy=policy,
                )
                if np.isfinite(metrics.rank_ic):
                    scores.append(metrics.rank_ic)
            if not scores:
                raise optuna.TrialPruned("No eligible purged inner folds")
            trial.set_user_attr("fold_rank_ic", scores)
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=study_seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        for trial in study.trials:
            trial_history.append(
                {
                    "scope": scope,
                    "trial_number": trial.number,
                    "state": trial.state.name,
                    "score": float(trial.value) if trial.value is not None else None,
                    "config": trial.user_attrs.get("config"),
                    "fold_rank_ic": trial.user_attrs.get("fold_rank_ic", []),
                }
            )
        return study

    outer_fold_scores: dict[int, float] = {}
    outer_fold_configs: dict[int, dict] = {}
    for outer_year in outer_years:
        outer_train = frame.filter(pl.col("trade_date").dt.year() < outer_year)
        outer_test = frame.filter(pl.col("trade_date").dt.year() == outer_year)
        if outer_train.height < 500 or outer_test.height < 100:
            continue
        inner_study = run_inner_study(
            outer_train, study_seed=seed + outer_year, scope=f"outer_{outer_year}_inner"
        )
        outer_config = inner_study.best_trial.user_attrs["config"]
        _, outer_metrics = train_once(
            outer_train,
            outer_test,
            feature_names=features,
            seed=seed + outer_year,
            model_config=outer_config,
            training_policy=policy,
        )
        outer_fold_scores[outer_year] = float(outer_metrics.rank_ic)
        outer_fold_configs[outer_year] = outer_config
    if not outer_fold_scores:
        raise ValueError("No eligible untouched outer folds")

    study = run_inner_study(frame, study_seed=seed, scope="final_development_inner")
    validation_years = years[-min(n_validation_years, len(years) - 2) :]
    return TuningResult(
        best_score=float(np.mean(list(outer_fold_scores.values()))),
        best_config=study.best_trial.user_attrs["config"],
        validation_years=validation_years,
        n_trials=n_trials,
        outer_fold_scores=outer_fold_scores,
        outer_fold_configs=outer_fold_configs,
        trial_history=trial_history,
        number_of_experiments=len(trial_history),
    )
