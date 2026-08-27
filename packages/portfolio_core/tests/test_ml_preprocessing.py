import warnings
from datetime import date

import numpy as np
import polars as pl
from portfolio_core.ml.model import EnsembleRegressor
from portfolio_core.ml.walk_forward import (
    _preprocessor,
    _sanitize_tree_matrix,
    cross_section_sample_weights,
    select_available_features,
)
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


def test_tree_branch_preserves_missing_and_lightgbm_bagging_is_active() -> None:
    raw = np.asarray([[1.0, np.inf], [np.nan, 2.0]])
    tree = _sanitize_tree_matrix(raw)
    assert np.isnan(tree[0, 1]) and np.isnan(tree[1, 0])
    model = EnsembleRegressor.create_default(lightgbm_overrides={"subsample": 0.75})
    config = model.effective_config()
    assert config["lightgbm_bagging_active"] is True
    assert config["lightgbm"]["subsample_freq"] == 1


def test_cross_sections_have_equal_mass_before_recency() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [date(2020, 1, 31)] * 2 + [date(2020, 2, 28)] * 4,
            "CD_CVM": ["A", "B", "A", "A", "B", "C"],
        }
    )
    weights = cross_section_sample_weights(
        frame, reference_date=date(2020, 2, 28), half_life_years=None
    )
    weighted = frame.with_columns(pl.Series("weight", weights)).group_by("trade_date").agg(
        pl.col("weight").sum()
    )
    assert np.allclose(weighted["weight"].to_numpy(), weighted["weight"][0])
