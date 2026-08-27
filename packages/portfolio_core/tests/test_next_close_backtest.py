from datetime import date

import polars as pl
import pytest

from portfolio_core.backtest import BacktestConfig, run_monthly_topk_backtest
from portfolio_core.data_quality import DataQualityError


def _prices(closes: list[float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, day) for day in range(2, 2 + len(closes))],
            "ticker": ["TEST3"] * len(closes),
            "close": closes,
            "adjusted_close": closes,
        }
    )


def test_signal_close_to_next_close_return_is_not_captured() -> None:
    prices = _prices([10.0, 20.0, 22.0])
    signals = pl.DataFrame(
        {"trade_date": [date(2024, 1, 2)], "ticker": ["TEST3"], "signal": [1.0]}
    )
    curve, summary = run_monthly_topk_backtest(
        prices,
        signals,
        score_column="signal",
        config=BacktestConfig(top_k=1, transaction_cost_bps=0.0),
    )
    assert curve["portfolio_value"].to_list() == pytest.approx([100_000.0, 100_000.0, 110_000.0])
    assert summary.end_value == pytest.approx(110_000.0)


def test_initial_transaction_cost_is_charged_at_next_close() -> None:
    prices = _prices([10.0, 10.0, 10.0])
    signals = pl.DataFrame(
        {"trade_date": [date(2024, 1, 2)], "ticker": ["TEST3"], "signal": [1.0]}
    )
    _, summary = run_monthly_topk_backtest(
        prices,
        signals,
        score_column="signal",
        config=BacktestConfig(top_k=1, transaction_cost_bps=10.0),
    )
    assert summary.end_value == pytest.approx(99_900.0)
    assert summary.turnover == pytest.approx(1.0)


def test_mixed_monthly_signal_dates_are_rejected() -> None:
    prices = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 3)],
            "ticker": ["A3", "B3", "A3", "B3"],
            "close": [10.0, 20.0, 11.0, 21.0],
        }
    )
    signals = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 3)],
            "ticker": ["A3", "B3"],
            "signal": [1.0, 0.5],
        }
    )
    with pytest.raises(DataQualityError, match="mix different trade dates"):
        run_monthly_topk_backtest(prices, signals, score_column="signal")


def test_missing_execution_quote_is_rejected_not_forward_filled() -> None:
    from portfolio_core.backtest import run_monthly_topk_backtest_detailed

    prices = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2), date(2024, 1, 2), date(2024, 1, 3),
                date(2024, 1, 4), date(2024, 1, 4),
            ],
            "ticker": ["A3", "B3", "B3", "A3", "B3"],
            "close": [10.0, 20.0, 20.0, 11.0, 20.0],
            "adjusted_close": [10.0, 20.0, 20.0, 11.0, 20.0],
        }
    )
    signals = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 2), date(2024, 1, 2)],
            "ticker": ["A3", "B3"],
            "signal": [2.0, 1.0],
        }
    )
    detail = run_monthly_topk_backtest_detailed(
        prices,
        signals,
        score_column="signal",
        config=BacktestConfig(top_k=1, transaction_cost_bps=0.0),
    )
    assert detail.selections["ticker"].to_list() == ["B3"]
    assert (
        detail.rejected_orders.filter(pl.col("ticker") == "A3")["reason"].item()
        == "no_observed_execution_quote"
    )
