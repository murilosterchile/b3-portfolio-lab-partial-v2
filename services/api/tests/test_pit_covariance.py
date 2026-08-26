from datetime import date, timedelta

import numpy as np
import polars as pl

from app.research import historical_correlation


def _write_prices(root, future_multiplier: float) -> None:
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(120)]
    rows = []
    for i, dt in enumerate(dates):
        a = 100.0 + i
        b = 80.0 + 0.5 * i
        if i > 90:
            b *= future_multiplier
        rows.extend(
            [
                {"trade_date": dt, "ticker": "A3", "close": a, "adjusted_close": a},
                {"trade_date": dt, "ticker": "B3", "close": b, "adjusted_close": b},
            ]
        )
    path = root / "silver" / "b3_prices_adjusted" / "year=2024" / "part-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def test_future_prices_do_not_change_asof_covariance(tmp_path) -> None:
    cutoff = date(2024, 1, 1) + timedelta(days=90)
    _write_prices(tmp_path, future_multiplier=1.0)
    first = historical_correlation(
        tickers=["A3", "B3"], data_dir=str(tmp_path), as_of=cutoff, lookback_observations=80
    )
    _write_prices(tmp_path, future_multiplier=100.0)
    second = historical_correlation(
        tickers=["A3", "B3"], data_dir=str(tmp_path), as_of=cutoff, lookback_observations=80
    )
    assert first is not None and second is not None
    assert np.allclose(first, second)
