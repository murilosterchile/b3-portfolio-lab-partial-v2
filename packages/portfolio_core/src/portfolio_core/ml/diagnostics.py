from __future__ import annotations

from dataclasses import asdict
import numpy as np
import polars as pl
from scipy.stats import ks_2samp, spearmanr, wasserstein_distance

from .evaluation import panel_top_k_metrics
from .walk_forward import TrainedSignalModel


def _finite(values: pl.Series) -> np.ndarray:
    arr = values.cast(pl.Float64, strict=False).to_numpy()
    return arr[np.isfinite(arr)]


def population_stability_index(reference: np.ndarray, current: np.ndarray, *, bins: int = 10) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins or len(current) < bins:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    eps = 1e-6
    ref_pct = np.maximum(ref_hist / max(ref_hist.sum(), 1), eps)
    cur_pct = np.maximum(cur_hist / max(cur_hist.sum(), 1), eps)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def feature_drift_report(
    frame: pl.DataFrame,
    features: list[str],
    *,
    regime_a: tuple[int, int] = (2014, 2021),
    regime_b: tuple[int, int] = (2022, 2025),
) -> pl.DataFrame:
    year = pl.col("trade_date").dt.year()
    a = frame.filter(year.is_between(*regime_a))
    b = frame.filter(year.is_between(*regime_b))
    rows: list[dict[str, object]] = []
    for feature in features:
        av = _finite(a[feature])
        bv = _finite(b[feature])
        if len(av) == 0 or len(bv) == 0:
            continue
        ks = ks_2samp(av, bv, method="auto")
        rows.append(
            {
                "feature": feature,
                "a_mean": float(np.mean(av)),
                "a_median": float(np.median(av)),
                "a_std": float(np.std(av, ddof=1)) if len(av) > 1 else 0.0,
                "a_q05": float(np.quantile(av, 0.05)),
                "a_q25": float(np.quantile(av, 0.25)),
                "a_q75": float(np.quantile(av, 0.75)),
                "a_q95": float(np.quantile(av, 0.95)),
                "a_missingness": float(1.0 - a[feature].is_not_null().mean()),
                "b_mean": float(np.mean(bv)),
                "b_median": float(np.median(bv)),
                "b_std": float(np.std(bv, ddof=1)) if len(bv) > 1 else 0.0,
                "b_q05": float(np.quantile(bv, 0.05)),
                "b_q25": float(np.quantile(bv, 0.25)),
                "b_q75": float(np.quantile(bv, 0.75)),
                "b_q95": float(np.quantile(bv, 0.95)),
                "b_missingness": float(1.0 - b[feature].is_not_null().mean()),
                "psi": population_stability_index(av, bv),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "wasserstein": float(wasserstein_distance(av, bv)),
            }
        )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(
        ["psi", "ks_statistic", "wasserstein"], descending=[True, True, True]
    )


def monthly_feature_ic(
    frame: pl.DataFrame,
    features: list[str],
    *,
    target_column: str = "target_excess_return",
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for group in frame.partition_by("trade_date", maintain_order=True):
        trade_date = group["trade_date"].item(0)
        target = group[target_column].to_numpy()
        for feature in features:
            values = group[feature].to_numpy()
            mask = np.isfinite(values) & np.isfinite(target)
            if mask.sum() < 10:
                continue
            stat = spearmanr(values[mask], target[mask]).statistic
            if stat is not None and np.isfinite(stat):
                rows.append({"trade_date": trade_date, "feature": feature, "ic": float(stat)})
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={"trade_date": pl.Date, "feature": pl.Utf8, "ic": pl.Float64}
    )


def feature_ic_report(
    frame: pl.DataFrame,
    features: list[str],
    *,
    target_column: str = "target_excess_return",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    monthly = monthly_feature_ic(frame, features, target_column=target_column)
    if monthly.is_empty():
        return monthly, pl.DataFrame()
    annual = monthly.with_columns(pl.col("trade_date").dt.year().alias("year")).group_by(
        ["feature", "year"]
    ).agg(
        pl.col("ic").mean().alias("ic_mean"),
        pl.col("ic").median().alias("ic_median"),
        pl.col("ic").std().alias("ic_std"),
        pl.len().alias("months"),
    ).with_columns(
        (pl.col("ic_mean") / pl.col("ic_std")).alias("icir"),
        (pl.col("ic_std") / pl.col("months").cast(pl.Float64).sqrt()).alias("ic_se"),
    ).with_columns((pl.col("ic_mean") / pl.col("ic_se")).alias("ic_tstat"))

    regime = monthly.with_columns(pl.col("trade_date").dt.year().alias("year")).group_by("feature").agg(
        pl.col("ic").filter(pl.col("year").is_between(2014, 2021)).mean().alias("ic_2014_2021"),
        pl.col("ic").filter(pl.col("year").is_between(2022, 2025)).mean().alias("ic_2022_2025"),
    ).with_columns(
        (pl.col("ic_2022_2025") - pl.col("ic_2014_2021")).alias("delta_ic"),
        (
            pl.col("ic_2014_2021").sign() != pl.col("ic_2022_2025").sign()
        ).alias("sign_flip"),
    ).with_columns(
        pl.when(pl.col("sign_flip"))
        .then(pl.lit("sign_flip"))
        .when(pl.col("delta_ic") > 0.02)
        .then(pl.lit("gained_relevance"))
        .when(pl.col("delta_ic") < -0.02)
        .then(pl.lit("deteriorated"))
        .when((pl.col("ic_2014_2021").abs() < 0.02) & (pl.col("ic_2022_2025").abs() < 0.02))
        .then(pl.lit("never_material"))
        .otherwise(pl.lit("stable"))
        .alias("classification")
    ).sort("delta_ic")
    return annual.sort(["feature", "year"]), regime


def yearly_topk_report(
    frame: pl.DataFrame,
    predictions: np.ndarray,
    *,
    target_column: str = "target_excess_return",
    ks: tuple[int, ...] = (5, 10, 20, 30, 50),
) -> pl.DataFrame:
    evaluated = frame.with_columns(pl.Series("__prediction", predictions))
    rows: list[dict[str, object]] = []
    for year in sorted(set(evaluated["trade_date"].dt.year().to_list())):
        subset = evaluated.filter(pl.col("trade_date").dt.year() == year)
        metrics = panel_top_k_metrics(
            subset,
            subset["__prediction"].to_numpy(),
            ks=ks,
            target_column=target_column,
        )
        for k, metric in metrics.items():
            rows.append({"year": year, **asdict(metric)})
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def model_importance(model: TrainedSignalModel, *, fold_year: int | None = None) -> pl.DataFrame:
    try:
        names = list(model.preprocessing.get_feature_names_out(model.feature_names))
    except Exception:
        n = len(model.model.ridge.coef_)
        names = [f"transformed_feature_{i}" for i in range(n)]
    lgb = model.model.lightgbm.booster_
    gain = lgb.feature_importance(importance_type="gain")
    split = lgb.feature_importance(importance_type="split")
    cat = np.asarray(model.model.catboost.get_feature_importance(type="PredictionValuesChange"))
    ridge = np.asarray(model.model.ridge.coef_, dtype=float)
    rows: list[dict[str, object]] = []
    for feature, value in zip(model.feature_names, gain, strict=True):
        rows.append({"model": "lightgbm", "feature": feature, "metric": "gain", "value": float(value)})
    for feature, value in zip(model.feature_names, split, strict=True):
        rows.append({"model": "lightgbm", "feature": feature, "metric": "split", "value": float(value)})
    for feature, value in zip(model.feature_names, cat, strict=True):
        rows.append({"model": "catboost", "feature": feature, "metric": "prediction_values_change", "value": float(value)})
    for feature, value in zip(names, ridge, strict=True):
        rows.append({"model": "ridge", "feature": feature, "metric": "standardized_coefficient", "value": float(value)})
    frame = pl.DataFrame(rows)
    if fold_year is not None:
        frame = frame.with_columns(pl.lit(fold_year).alias("fold_year"))
    return frame
