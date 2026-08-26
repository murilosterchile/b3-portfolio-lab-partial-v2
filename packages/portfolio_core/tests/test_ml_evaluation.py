from datetime import date

import numpy as np
import polars as pl

from portfolio_core.ml.evaluation import panel_cross_sectional_metrics


def test_panel_metrics_rank_within_each_date() -> None:
    frame = pl.DataFrame(
        {
            "trade_date": [date(2025, 1, 31)] * 10 + [date(2025, 2, 28)] * 10,
            "target_excess_return": list(np.arange(10.0)) + list(np.arange(10.0)),
        }
    )
    # Perfect ranking on each date, despite an arbitrary level shift in predictions.
    predictions = np.r_[np.arange(10.0), np.arange(10.0) + 1000.0]
    metrics = panel_cross_sectional_metrics(frame, predictions)
    assert abs(metrics.rank_ic - 1.0) < 1e-12
    assert metrics.top_minus_bottom > 0
