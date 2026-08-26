from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray) -> object: ...
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
            weights=weights,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EnsembleRegressor":
        self.lightgbm.fit(x, y)
        self.catboost.fit(x, y)
        self.ridge.fit(x, y)
        return self

    def predict_components(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack(
            [self.lightgbm.predict(x), self.catboost.predict(x), self.ridge.predict(x)]
        )

    def predict(self, x: np.ndarray) -> np.ndarray:
        components = self.predict_components(x)
        return components @ np.asarray(self.weights)

    def uncertainty(self, x: np.ndarray) -> np.ndarray:
        return np.std(self.predict_components(x), axis=1, ddof=0)
