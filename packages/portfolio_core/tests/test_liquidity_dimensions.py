from datetime import date, timedelta

import numpy as np
import polars as pl

from portfolio_core.features.technical import build_technical_features


def test_liquidity_features_use_financial_traded_value_not_quantity_or_price_times_value() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=index) for index in range(22)]
    rows = []
    for ticker, traded_value_brl in (("LOWV3", 1_000.0), ("HIGH3", 2_000.0)):
        for index, trade_date in enumerate(dates):
            rows.append(
                {
                    "trade_date": trade_date,
                    "ticker": ticker,
                    "close": 10.0 * (1.01**index),
                    "quantity": 7,
                    "traded_value_brl": traded_value_brl,
                    "trades": 10,
                }
            )

    last = build_technical_features(pl.DataFrame(rows), horizon_days=1).filter(
        pl.col("trade_date") == dates[-1]
    ).sort("ticker")

    assert np.isclose(
        last.filter(pl.col("ticker") == "LOWV3")["log_traded_value_21d"].item(),
        np.log1p(1_000.0),
    )
    assert np.isclose(
        last.filter(pl.col("ticker") == "HIGH3")["log_traded_value_21d"].item(),
        np.log1p(2_000.0),
    )
    low_amihud = last.filter(pl.col("ticker") == "LOWV3")["amihud_illiquidity_21d"].item()
    high_amihud = last.filter(pl.col("ticker") == "HIGH3")["amihud_illiquidity_21d"].item()
    assert np.isclose(low_amihud, 0.01 / 1_000.0)
    assert np.isclose(low_amihud / high_amihud, 2.0)
