import numpy as np

from portfolio_core.ml.model import EnsembleRegressor


def _model(seed: int) -> EnsembleRegressor:
    return EnsembleRegressor.create_default(
        random_seed=seed,
        lightgbm_overrides={"n_estimators": 20, "n_jobs": 1},
        catboost_overrides={"iterations": 20, "thread_count": 1},
    )


def test_same_data_and_seed_produce_same_predictions() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(120, 4))
    y = rng.normal(size=120)
    first = _model(42).fit(x, y).predict(x)
    second = _model(42).fit(x, y).predict(x)
    assert np.allclose(first, second, rtol=1e-10, atol=1e-10)
