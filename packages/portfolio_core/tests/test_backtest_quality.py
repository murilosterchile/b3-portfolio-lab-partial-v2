from datetime import date

import numpy as np
import polars as pl
import pytest
from portfolio_core.backtest.engine import (
    BacktestConfig,
    _metrics,
    _one_way_turnover,
    run_monthly_topk_backtest,
)
from portfolio_core.data_quality import DataQualityError


def _prices(closes: list[float], distributions: list[int] | None = None) -> pl.DataFrame:
    dates = [date(2024, 1, day) for day in range(2, 2 + len(closes))]
    data: dict[str, object] = {
        "trade_date": dates,
        "ticker": ["TEST3"] * len(closes),
        "close": closes,
    }
    if distributions is not None:
        data["distribution_number"] = distributions
    return pl.DataFrame(data)


def _signals(signal_dates: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        {"trade_date": signal_dates, "ticker": ["TEST3"] * len(signal_dates), "signal": [1.0] * len(signal_dates)}
    )


def test_valid_price_sequence_produces_only_finite_returns() -> None:
    prices = _prices([10.0, 10.5, 10.25, 10.75], [1, 1, 1, 1])
    signals = _signals([date(2024, 1, 2)])
    curve, summary = run_monthly_topk_backtest(
        prices,
        signals,
        score_column="signal",
        config=BacktestConfig(top_k=1, transaction_cost_bps=0),
    )
    values = curve["portfolio_value"].to_numpy()
    assert np.isfinite(values).all()
    assert np.isfinite(values[1:] / values[:-1] - 1.0).all()
    assert np.isfinite(summary.sharpe)


def test_corporate_action_transition_is_explicitly_excluded_when_held() -> None:
    prices = _prices([10.0, 10.0, 10.5, 11.0], [7, 7, 8, 8])
    signals = _signals([date(2024, 1, 2)])
    _, summary = run_monthly_topk_backtest(
        prices,
        signals,
        score_column="signal",
        config=BacktestConfig(top_k=1, transaction_cost_bps=0),
    )
    assert summary.corporate_action_transitions_excluded == 1
    assert summary.end_value == pytest.approx(100_000 * 11.0 / 10.5)


def test_unexplained_extreme_held_return_fails() -> None:
    prices = _prices([1.0, 1.0, 3.0], [1, 1, 1])
    signals = _signals([date(2024, 1, 2)])
    with pytest.raises(DataQualityError, match="suspicious unadjusted returns"):
        run_monthly_topk_backtest(prices, signals, score_column="signal")


def test_unmodeled_action_in_adjusted_prices_is_explicitly_excluded() -> None:
    prices = _prices([1.0, 1.0, 7.0, 7.7], [125, 125, 126, 126]).with_columns(
        pl.col("close").alias("adjusted_close")
    )
    signals = _signals([date(2024, 1, 2)])
    _, summary = run_monthly_topk_backtest(
        prices,
        signals,
        score_column="signal",
        config=BacktestConfig(top_k=1, transaction_cost_bps=0),
    )
    assert summary.corporate_action_transitions_excluded == 1
    assert summary.end_value == pytest.approx(110_000.0)


def test_normal_adjusted_return_across_action_is_preserved() -> None:
    prices = _prices([10.0, 10.5, 11.0], [7, 7, 8]).with_columns(
        pl.col("close").alias("adjusted_close")
    )
    signals = _signals([date(2024, 1, 2)])
    _, summary = run_monthly_topk_backtest(
        prices,
        signals,
        score_column="signal",
        config=BacktestConfig(top_k=1, transaction_cost_bps=0),
    )
    assert summary.corporate_action_transitions_excluded == 0
    assert summary.end_value == pytest.approx(100_000 * 11.0 / 10.5)


def test_realistic_extreme_adjusted_return_is_preserved() -> None:
    prices = _prices([10.0, 17.5, 43.75], [137, 137, 137]).with_columns(
        pl.col("close").alias("adjusted_close")
    )
    signals = _signals([date(2024, 1, 2)])
    _, summary = run_monthly_topk_backtest(
        prices,
        signals,
        score_column="signal",
        config=BacktestConfig(top_k=1, transaction_cost_bps=0),
    )
    assert summary.end_value == pytest.approx(250_000)


def test_unexplained_implausible_adjusted_return_fails() -> None:
    prices = _prices([1.0, 1.0, 7.0], [1, 1, 1]).with_columns(
        pl.col("close").alias("adjusted_close")
    )
    signals = _signals([date(2024, 1, 2)])
    with pytest.raises(DataQualityError, match="suspicious adjusted returns"):
        run_monthly_topk_backtest(prices, signals, score_column="signal")


def test_sharpe_uses_signed_net_return_series() -> None:
    returns = np.array([-0.01, -0.02, -0.03])
    equity = 100_000.0 * np.cumprod(np.r_[1.0, 1.0 + returns])
    summary = _metrics(equity, returns, 0.0)
    expected = returns.mean() * 252 / (returns.std(ddof=1) * np.sqrt(252))
    assert summary.sharpe == pytest.approx(expected)
    assert summary.sharpe < 0


def test_half_l1_turnover_matches_simple_switch() -> None:
    current = np.array([0.5, 0.5, 0.0])
    target = np.array([0.0, 0.5, 0.5])
    assert _one_way_turnover(current, target) == pytest.approx(0.5)
    assert _one_way_turnover(
        np.zeros(3), current, current_cash=1.0, target_cash=0.0
    ) == pytest.approx(1.0)


def test_multiple_ticker_signal_dates_in_one_month_are_rejected() -> None:
    prices = pl.DataFrame(
        {
            "trade_date": [
                date(2024, 1, 2), date(2024, 1, 2),
                date(2024, 1, 3), date(2024, 1, 3),
                date(2024, 1, 4), date(2024, 1, 4),
            ],
            "ticker": ["A3", "B3"] * 3,
            "close": [10.0, 20.0, 10.1, 20.1, 10.2, 20.2],
            "distribution_number": [1] * 6,
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
        run_monthly_topk_backtest(
            prices,
            signals,
            score_column="signal",
            config=BacktestConfig(top_k=2, transaction_cost_bps=0),
        )


def test_rank_buffer_keeps_existing_holding_near_cutoff() -> None:
    from portfolio_core.backtest.engine import _buffered_selection

    selected = _buffered_selection(
        ["NEW3", "KEEP3", "OTHER3"],
        previous={"KEEP3"},
        top_k=1,
        buffer=1,
    )
    assert selected == ["KEEP3"]
