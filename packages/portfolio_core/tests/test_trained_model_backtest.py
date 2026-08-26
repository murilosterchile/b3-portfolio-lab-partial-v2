import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.run_backtest import _features_after_training


def test_trained_model_backtest_uses_only_strictly_future_rows() -> None:
    features = pl.DataFrame(
        {
            "trade_date": [date(2024, 12, 31), date(2025, 1, 31)],
            "ticker": ["AAAA3", "AAAA3"],
        }
    )

    future = _features_after_training(features, date(2024, 12, 31))

    assert future["trade_date"].to_list() == [date(2025, 1, 31)]


def test_trained_model_backtest_rejects_absence_of_future_rows() -> None:
    features = pl.DataFrame(
        {"trade_date": [date(2024, 12, 31)], "ticker": ["AAAA3"]}
    )

    with pytest.raises(SystemExit, match="No feature rows after model trained_until"):
        _features_after_training(features, date(2024, 12, 31))
