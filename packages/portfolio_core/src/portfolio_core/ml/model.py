from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRanker, LGBMRegressor
from sklearn.linear_model import Ridge


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs: object) -> object: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass
class EnsembleRegressor:
    lightgbm: LGBMRegressor
    catboost: CatBoostRegressor
    ridge: Ridge
    weights: tuple[float, float, float] = (0.45, 0.45, 0.10)

    @classmethod
    def create_default(
        cls,
        *,
        random_seed: int = 42,
        lightgbm_overrides: dict | None = None,
        catboost_overrides: dict | None = None,
        ridge_alpha: float = 10.0,
        weights: tuple[float, float, float] = (0.45, 0.45, 0.10),
    ) -> "EnsembleRegressor":
        raw_weights = np.asarray(weights, dtype=float)
        if (
            raw_weights.shape != (3,)
            or not np.all(np.isfinite(raw_weights))
            or np.any(raw_weights < 0)
            or raw_weights.sum() <= 0
        ):
            raise ValueError("ensemble weights must be three finite non-negative values with positive sum")
        normalized = tuple((raw_weights / raw_weights.sum()).tolist())
        lightgbm_params = {
            "objective": "huber",
            "n_estimators": 600,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "random_state": random_seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        lightgbm_params.update(lightgbm_overrides or {})
        # LightGBM ignores ``subsample`` unless bagging is scheduled explicitly.
        # Persist the effective setting so tuned artifacts are auditable.
        if float(lightgbm_params.get("subsample", 1.0)) < 1.0:
            lightgbm_params.setdefault("subsample_freq", 1)
        catboost_params = {
            "loss_function": "Huber:delta=1.0",
            "iterations": 600,
            "learning_rate": 0.025,
            "depth": 6,
            "l2_leaf_reg": 5.0,
            "random_seed": random_seed,
            "verbose": False,
            "allow_writing_files": False,
        }
        catboost_params.update(catboost_overrides or {})
        return cls(
            lightgbm=LGBMRegressor(**lightgbm_params),
            catboost=CatBoostRegressor(**catboost_params),
            ridge=Ridge(alpha=ridge_alpha),
            weights=normalized,
        )

    def fit(
        self,
        x_tree: np.ndarray,
        x_linear: np.ndarray,
        y: np.ndarray | None = None,
        *,
        sample_weight: np.ndarray | None = None,
    ) -> "EnsembleRegressor":
        if y is None:
            y = np.asarray(x_linear)
            x_linear = x_tree
        kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}
        self.lightgbm.fit(x_tree, y, **kwargs)
        self.catboost.fit(x_tree, y, **kwargs)
        self.ridge.fit(x_linear, y, **kwargs)
        return self

    def predict_components(
        self, x_tree: np.ndarray, x_linear: np.ndarray | None = None
    ) -> np.ndarray:
        if x_linear is None:
            x_linear = x_tree
        return np.column_stack(
            [
                self.lightgbm.predict(x_tree),
                self.catboost.predict(x_tree),
                self.ridge.predict(x_linear),
            ]
        )

    def predict(self, x_tree: np.ndarray, x_linear: np.ndarray | None = None) -> np.ndarray:
        components = self.predict_components(x_tree, x_linear)
        return components @ np.asarray(self.weights)

    def ensemble_disagreement(
        self, x_tree: np.ndarray, x_linear: np.ndarray | None = None
    ) -> np.ndarray:
        """Raw component dispersion; this is not predictive uncertainty by itself."""
        return np.std(self.predict_components(x_tree, x_linear), axis=1, ddof=0)

    def uncertainty(self, x_tree: np.ndarray, x_linear: np.ndarray | None = None) -> np.ndarray:
        """Compatibility alias; callers must not treat this as calibrated uncertainty."""
        return self.ensemble_disagreement(x_tree, x_linear)

    def effective_config(self) -> dict[str, object]:
        """Serializable configuration, including whether bagging is truly active."""
        lightgbm = dict(self.lightgbm.get_params())
        return {
            "lightgbm": lightgbm,
            "catboost": dict(self.catboost.get_params()),
            "ridge": dict(self.ridge.get_params()),
            "weights": list(self.weights),
            "lightgbm_bagging_active": (
                float(lightgbm.get("subsample", 1.0)) < 1.0
                and int(lightgbm.get("subsample_freq", 0)) > 0
            ),
        }


@dataclass
class LearningToRankModel:
    """Development challenger aligned with cross-sectional stock selection."""

    ranker: LGBMRanker

    @classmethod
    def create_default(
        cls,
        *,
        random_seed: int = 42,
        overrides: dict | None = None,
    ) -> "LearningToRankModel":
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "n_estimators": 600,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "random_state": random_seed,
            "n_jobs": -1,
            "verbosity": -1,
        }
        params.update(overrides or {})
        if float(params.get("subsample", 1.0)) < 1.0:
            params.setdefault("subsample_freq", 1)
        return cls(LGBMRanker(**params))

    def fit(
        self,
        x: np.ndarray,
        relevance: np.ndarray,
        *,
        group: list[int],
        sample_weight: np.ndarray | None = None,
    ) -> "LearningToRankModel":
        kwargs: dict[str, object] = {"group": group}
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight
        self.ranker.fit(x, relevance, **kwargs)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self.ranker.predict(x), dtype=float)
