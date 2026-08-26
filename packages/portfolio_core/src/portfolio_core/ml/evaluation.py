from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import spearmanr


@dataclass(frozen=True)
class CrossSectionalMetrics:
    rank_ic: float
    top_decile_mean: float
    bottom_decile_mean: float
    top_minus_bottom: float
    hit_rate: float


def cross_sectional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> CrossSectionalMetrics:
    """Metrics for a *single* cross-section.

    Callers evaluating a time panel should use ``panel_cross_sectional_metrics`` so assets from
    different prediction dates are never ranked against one another.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    if len(yt) < 10:
        return CrossSectionalMetrics(float("nan"), *(float("nan"),) * 4)
    statistic = spearmanr(yt, yp).statistic
    rank_ic = float(statistic) if statistic is not None else float("nan")
    order = np.argsort(yp)
    k = max(1, len(order) // 10)
    bottom = float(np.mean(yt[order[:k]]))
    top = float(np.mean(yt[order[-k:]]))
    hit = float(np.mean((yt > 0) == (yp > 0)))
    return CrossSectionalMetrics(rank_ic, top, bottom, top - bottom, hit)


def panel_cross_sectional_metrics(
    frame: pl.DataFrame,
    predictions: np.ndarray,
    *,
    target_column: str = "target_excess_return",
    date_column: str = "trade_date",
) -> CrossSectionalMetrics:
    """Average single-date cross-sectional metrics over a validation panel."""
    if frame.height != len(predictions):
        raise ValueError("frame and predictions must have the same number of rows")
    evaluated = frame.select([date_column, target_column]).with_columns(
        pl.Series("__prediction", predictions)
    )
    per_date: list[CrossSectionalMetrics] = []
    for group in evaluated.partition_by(date_column, maintain_order=True):
        metrics = cross_sectional_metrics(
            group[target_column].to_numpy(), group["__prediction"].to_numpy()
        )
        if np.isfinite(metrics.rank_ic):
            per_date.append(metrics)
    if not per_date:
        return CrossSectionalMetrics(float("nan"), *(float("nan"),) * 4)
    matrix = np.asarray(
        [
            [m.rank_ic, m.top_decile_mean, m.bottom_decile_mean, m.top_minus_bottom, m.hit_rate]
            for m in per_date
        ],
        dtype=float,
    )
    means = np.nanmean(matrix, axis=0)
    return CrossSectionalMetrics(*(float(value) for value in means))
