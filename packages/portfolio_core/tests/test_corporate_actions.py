from datetime import date

import polars as pl
import pytest

from portfolio_core.data.corporate_actions import adjust_prices_for_total_return, parse_company_actions


def _prices(values: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2 + index) for index in range(len(values))],
            "ticker": ["TEST3"] * len(values),
            "isin": ["BRTESTACNOR0"] * len(values),
            "close": values,
            "volume": [1_000_000.0] * len(values),
            "trades": [100] * len(values),
        }
    )


def test_b3_cash_dividend_is_backward_total_return_adjusted() -> None:
    prices = _prices([100.0, 95.0, 96.0])
    payload = [
        {
            "cashDividends": [
                {
                    "label": "DIVIDENDO",
                    "assetIssued": "BRTESTACNOR0",
                    "lastDatePrior": "02/01/2024",
                    "rate": "5,00",
                }
            ],
            "stockDividends": [],
        }
    ]
    actions = parse_company_actions("TEST", payload)
    adjusted, quarantine = adjust_prices_for_total_return(prices, actions)

    assert quarantine.is_empty()
    assert adjusted["adjusted_close"].to_list() == pytest.approx([95.0, 95.0, 96.0])


def test_b3_split_is_backward_adjusted() -> None:
    prices = _prices([100.0, 102.0, 51.0, 52.0])
    payload = [
        {
            "cashDividends": [],
            "stockDividends": [
                {
                    "label": "DESDOBRAMENTO",
                    "assetIssued": "BRTESTACNOR0",
                    "lastDatePrior": "03/01/2024",
                    "factor": "100,00",
                }
            ],
        }
    ]
    actions = parse_company_actions("TEST", payload)
    adjusted, _ = adjust_prices_for_total_return(prices, actions)

    assert adjusted["adjusted_close"].to_list() == pytest.approx([50.0, 51.0, 51.0, 52.0])


def test_invalid_raw_close_is_quarantined_not_zero_filled() -> None:
    prices = _prices([10.0, 0.0, 10.5])
    actions = pl.DataFrame(
        schema={
            "issuer_root": pl.Utf8,
            "isin": pl.Utf8,
            "action_type": pl.Utf8,
            "label": pl.Utf8,
            "last_date_prior": pl.Date,
            "approved_on": pl.Date,
            "payment_date": pl.Date,
            "cash_amount": pl.Float64,
            "share_factor": pl.Float64,
            "source": pl.Utf8,
        }
    )
    adjusted, quarantine = adjust_prices_for_total_return(prices, actions)

    assert quarantine.height == 1
    assert quarantine["close"].to_list() == [0.0]
    assert adjusted["close"].to_list() == [10.0, 10.5]
