from datetime import date, timedelta

import polars as pl

from portfolio_core.data.universe import (
    UniverseConfig,
    apply_point_in_time_universe,
    build_point_in_time_universe,
)


def test_universe_uses_only_past_sessions_and_rejects_sparse_names() -> None:
    dates = [date(2024, 1, 1) + timedelta(days=offset) for offset in range(8)]
    rows = []
    for index, trade_date in enumerate(dates):
        rows.append(
            {"trade_date": trade_date, "ticker": "LIQD3", "close": 10.0, "volume": 2_000_000.0}
        )
        if index % 2 == 0:
            rows.append(
                {"trade_date": trade_date, "ticker": "SPRS3", "close": 10.0, "volume": 2_000_000.0}
            )
    universe = build_point_in_time_universe(
        pl.DataFrame(rows),
        config=UniverseConfig(
            participation_window_sessions=4,
            min_session_participation=0.75,
            adv_window_sessions=2,
            min_adv_brl=1_000_000.0,
            min_history_sessions=2,
        ),
    )
    latest = universe.filter(pl.col("trade_date") == dates[-1])
    assert latest.filter(pl.col("ticker") == "LIQD3")["universe_eligible"].item()
    assert latest.filter(pl.col("ticker") == "SPRS3").is_empty()


def test_ineligible_ticker_cannot_change_targets_of_eligible_tickers() -> None:
    signal_date = date(2024, 1, 31)
    signals = pl.DataFrame(
        {
            "trade_date": [signal_date] * 3,
            "ticker": ["ELIG3", "ELIG4", "INEL3"],
            "future_return": [0.10, 0.30, 9.00],
        }
    )
    universe = pl.DataFrame(
        {
            "trade_date": [signal_date] * 3,
            "ticker": ["ELIG3", "ELIG4", "INEL3"],
            "quote_observed": [True] * 3,
            "universe_eligible": [True, True, False],
            "universe_rejection_reason": [None, None, "low_adv"],
        }
    )
    baseline = apply_point_in_time_universe(
        signals.filter(pl.col("ticker") != "INEL3"),
        universe.filter(pl.col("ticker") != "INEL3"),
    ).sort("ticker")
    with_ineligible = apply_point_in_time_universe(signals, universe).sort("ticker")

    assert with_ineligible["ticker"].to_list() == ["ELIG3", "ELIG4"]
    assert with_ineligible.select(
        "ticker", "target_excess_return", "target_cross_sectional_rank"
    ).equals(
        baseline.select(
            "ticker", "target_excess_return", "target_cross_sectional_rank"
        )
    )
