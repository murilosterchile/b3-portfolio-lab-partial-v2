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


@dataclass(frozen=True)
class TopKMetrics:
    k: int
    precision: float
    recall: float
    ndcg: float
    mean_future_return: float
    median_future_return: float
    hit_rate: float
    top_minus_universe: float
    top_minus_bottom: float
    top_minus_equal_weight_universe: float


def cross_sectional_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> CrossSectionalMetrics:
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


def top_k_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, k: int) -> TopKMetrics:
    if k <= 0:
        raise ValueError("k must be positive")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = np.asarray(y_true, dtype=float)[mask]
    yp = np.asarray(y_pred, dtype=float)[mask]
    if len(yt) < max(3, k):
        return TopKMetrics(k, *(float("nan"),) * 9)
    kk = min(k, len(yt))
    predicted = np.argsort(yp)[::-1][:kk]
    actual = np.argsort(yt)[::-1][:kk]
    bottom = np.argsort(yp)[:kk]
    overlap = len(set(predicted.tolist()) & set(actual.tolist()))
    precision = overlap / kk
    recall = overlap / len(actual)
    actual_set = set(actual.tolist())
    relevance = np.asarray([1.0 if index in actual_set else 0.0 for index in predicted])
    discounts = 1.0 / np.log2(np.arange(2, kk + 2, dtype=float))
    dcg = float(np.sum(relevance * discounts))
    idcg = float(np.sum(discounts))
    ndcg = dcg / idcg if idcg > 0 else float("nan")
    top_returns = yt[predicted]
    universe = float(np.mean(yt))
    top_mean = float(np.mean(top_returns))
    bottom_mean = float(np.mean(yt[bottom]))
    return TopKMetrics(
        k=kk,
        precision=float(precision),
        recall=float(recall),
        ndcg=float(ndcg),
        mean_future_return=top_mean,
        median_future_return=float(np.median(top_returns)),
        hit_rate=float(np.mean(top_returns > 0)),
        top_minus_universe=top_mean - universe,
        top_minus_bottom=top_mean - bottom_mean,
        top_minus_equal_weight_universe=top_mean - universe,
    )


def panel_cross_sectional_metrics(
    frame: pl.DataFrame,
    predictions: np.ndarray,
    *,
    target_column: str = "target_excess_return",
    date_column: str = "trade_date",
) -> CrossSectionalMetrics:
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
        [[m.rank_ic, m.top_decile_mean, m.bottom_decile_mean, m.top_minus_bottom, m.hit_rate] for m in per_date],
        dtype=float,
    )
    means = np.nanmean(matrix, axis=0)
    return CrossSectionalMetrics(*(float(value) for value in means))


def panel_top_k_metrics(
    frame: pl.DataFrame,
    predictions: np.ndarray,
    *,
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
    target_column: str = "target_excess_return",
    date_column: str = "trade_date",
) -> dict[int, TopKMetrics]:
    if frame.height != len(predictions):
        raise ValueError("frame and predictions must have the same number of rows")
    evaluated = frame.select([date_column, target_column]).with_columns(
        pl.Series("__prediction", predictions)
    )
    buckets: dict[int, list[TopKMetrics]] = {k: [] for k in ks}
    for group in evaluated.partition_by(date_column, maintain_order=True):
        for k in ks:
            metric = top_k_metrics(
                group[target_column].to_numpy(), group["__prediction"].to_numpy(), k=k
            )
            if np.isfinite(metric.ndcg):
                buckets[k].append(metric)
    result: dict[int, TopKMetrics] = {}
    fields = [
        "precision", "recall", "ndcg", "mean_future_return", "median_future_return",
        "hit_rate", "top_minus_universe", "top_minus_bottom", "top_minus_equal_weight_universe",
    ]
    for k, values in buckets.items():
        if not values:
            result[k] = TopKMetrics(k, *(float("nan"),) * 9)
            continue
        means = [float(np.nanmean([getattr(item, field) for item in values])) for field in fields]
        result[k] = TopKMetrics(k, *means)
    return result
