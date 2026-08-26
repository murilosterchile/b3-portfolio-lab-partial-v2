import warnings

import numpy as np
import polars as pl
from portfolio_core.ml.walk_forward import _preprocessor, select_available_features
from scipy.linalg import LinAlgWarning
from sklearn.linear_model import Ridge


def test_completely_missing_feature_is_disabled_from_training_only() -> None:
    train = pl.DataFrame({"available": [1.0, 2.0, 3.0], "empty": [None, None, None]})

    enabled, disabled, coverage = select_available_features(
        train, ["available", "empty"], minimum_coverage=0.01
    )

    assert enabled == ["available"]
    assert disabled == ["empty"]
    assert coverage == {"available": 1.0, "empty": 0.0}


def test_ridge_preprocessing_handles_very_different_feature_scales() -> None:
    small = np.linspace(1.0, 100.0, 200)
    raw = np.column_stack([small, small * 1e15, np.ones_like(small)])
    target = np.sin(small)
    transformed = _preprocessor().fit_transform(raw)

    with warnings.catch_warnings():
        warnings.simplefilter("error", LinAlgWarning)
        model = Ridge(alpha=10.0).fit(transformed, target)

    assert np.isfinite(transformed).all()
    assert np.isfinite(model.coef_).all()
