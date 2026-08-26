from datetime import date, timedelta

import polars as pl
import pytest

from portfolio_core.data.corporate_actions import adjust_prices_for_total_return
from portfolio_core.features.technical import build_technical_features


def _prices() -> pl.DataFrame:
    start = date(2024, 1, 1)
    dates = [start + timedelta(days=i) for i in range(15)]
    return pl.DataFrame(
        {
            "trade_date": dates,
            "ticker": ["TEST3"] * len(dates),
            "isin": ["BRTESTACNOR0"] * len(dates),
            "close": [100.0 + i for i in range(len(dates))],
            "volume": [1000.0] * len(dates),
            "trades": [100] * len(dates),
        }
    )


def _action_schema() -> dict[str, pl.DataType]:
    return {
        "isin": pl.Utf8,
        "action_type": pl.Utf8,
        "last_date_prior": pl.Date,
        "cash_amount": pl.Float64,
        "share_factor": pl.Float64,
    }


def test_future_action_does_not_change_pre_event_trailing_return_feature() -> None:
    prices = _prices()
    no_actions = pl.DataFrame(schema=_action_schema())
    future_action = pl.DataFrame(
        {
            "isin": ["BRTESTACNOR0"],
            "action_type": ["cash_dividend"],
            "last_date_prior": [date(2024, 1, 12)],
            "cash_amount": [1.0],
            "share_factor": [None],
        },
        schema=_action_schema(),
    )
    baseline, _ = adjust_prices_for_total_return(prices, no_actions)
    adjusted, _ = adjust_prices_for_total_return(prices, future_action)
    baseline_features = build_technical_features(baseline, horizon_days=2)
    adjusted_features = build_technical_features(adjusted, horizon_days=2)
    feature_date = date(2024, 1, 8)
    before = baseline_features.filter(pl.col("trade_date") == feature_date)["return_5d"].item()
    after = adjusted_features.filter(pl.col("trade_date") == feature_date)["return_5d"].item()
    assert after == pytest.approx(before)
