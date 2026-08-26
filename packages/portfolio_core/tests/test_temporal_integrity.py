from datetime import date

import numpy as np
import polars as pl

from portfolio_core.data_quality import filter_labels_known_by, purge_overlapping_labels
from portfolio_core.features.technical import build_technical_features, monthly_snapshots
from portfolio_core.ml.walk_forward import exponential_recency_weights


def _prices() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "trade_date": [date(2024, 1, day) for day in range(2, 7)],
            "ticker": ["TEST3"] * 5,
            "close": [10.0, 20.0, 30.0, 40.0, 50.0],
            "adjusted_close": [10.0, 20.0, 30.0, 40.0, 50.0],
            "volume": [1000.0] * 5,
            "trades": [100] * 5,
        }
    )


def test_target_starts_after_signal_close() -> None:
    features = build_technical_features(_prices(), horizon_days=2, execution_lag_bars=1)
    first = features.filter(pl.col("trade_date") == date(2024, 1, 2))
    assert first["target_start_date"].item() == date(2024, 1, 3)
    assert first["target_end_date"].item() == date(2024, 1, 5)
    assert first["future_return"].item() == 1.0  # 40 / 20 - 1, not 30 / 10 - 1


def test_purge_removes_labels_that_cross_validation_start() -> None:
    train = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 1), date(2024, 1, 15)],
            "target_end_date": [date(2024, 1, 20), date(2024, 2, 10)],
        }
    )
    valid = pl.DataFrame({"trade_date": [date(2024, 2, 1)]})
    purged = purge_overlapping_labels(train, valid)
    assert purged["trade_date"].to_list() == [date(2024, 1, 1)]


def test_knowledge_cutoff_uses_label_end_not_feature_date() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [date(2025, 9, 1), date(2025, 12, 1)],
            "target_end_date": [date(2025, 12, 1), date(2026, 3, 1)],
        }
    )
    known = filter_labels_known_by(frame, knowledge_cutoff=date(2025, 12, 31))
    assert known["trade_date"].to_list() == [date(2025, 9, 1)]


def test_monthly_snapshot_uses_one_common_market_date() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [date(2024, 1, 30), date(2024, 1, 31), date(2024, 1, 30)],
            "ticker": ["A3", "A3", "B3"],
            "x": [1.0, 2.0, 3.0],
        }
    )
    monthly = monthly_snapshots(frame)
    assert monthly["trade_date"].unique().to_list() == [date(2024, 1, 31)]
    assert monthly["ticker"].to_list() == ["A3"]


def test_recency_weights_are_monotone_and_normalized() -> None:
    dates = [date(2021, 1, 1), date(2023, 1, 1), date(2025, 1, 1)]
    weights = exponential_recency_weights(
        dates, reference_date=date(2025, 1, 1), half_life_years=2.0
    )
    assert np.all(np.diff(weights) > 0)
    assert weights.mean() == 1.0
