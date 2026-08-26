from datetime import date

import numpy as np
import polars as pl
import pytest
from portfolio_core.data_quality import DataQualityError, validate_prices
from portfolio_core.features.technical import build_technical_features
from portfolio_core.quant.risk import ledoit_wolf_covariance


@pytest.mark.parametrize("invalid_close", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_invalid_close_fails_with_ticker_and_date(invalid_close: float) -> None:
    prices = pl.DataFrame(
        {"ticker": ["TEST3"], "trade_date": [date(2024, 1, 2)], "close": [invalid_close]}
    )

    with pytest.raises(DataQualityError, match=r"TEST3.*2024"):
        validate_prices(prices)


def test_duplicate_ticker_date_fails() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["TEST3", "TEST3"],
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "close": [10.0, 10.0],
        }
    )

    with pytest.raises(DataQualityError, match="duplicate"):
        validate_prices(prices)


def test_ledoit_wolf_rejects_non_finite_returns() -> None:
    returns = np.array([[0.01, 0.02], [0.03, np.inf], [0.02, 0.01]])

    with pytest.raises(DataQualityError, match="Ledoit-Wolf returns"):
        ledoit_wolf_covariance(returns)


def test_technical_features_reject_invalid_price_input() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["TEST3"],
            "trade_date": [date(2024, 1, 2)],
            "close": [0.0],
            "volume": [1000.0],
            "trades": [10],
        }
    )

    with pytest.raises(DataQualityError, match="technical feature prices"):
        build_technical_features(prices)


def test_technical_features_mask_corporate_action_price_transition() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["TEST3"] * 4,
            "trade_date": [date(2024, 1, day) for day in range(2, 6)],
            "close": [1.0, 1.1, 11.0, 11.5],
            "volume": [1000.0] * 4,
            "trades": [10] * 4,
            "distribution_number": [1, 1, 2, 2],
        }
    )

    features = build_technical_features(prices, horizon_days=1)

    assert features.filter(pl.col("trade_date") == date(2024, 1, 4))["ret_1d"].item() is None
    assert features.filter(pl.col("trade_date") == date(2024, 1, 3))["future_return"].item() is None
